"""Game aggregate root for the Tractor game.

Wraps sm state machines, manages 4 Player instances, drives the sync
round-robin bidding, and provides receive(), snapshot(), is_over(),
set_on_game_over(), get_player(), and resolve_cards() interfaces.

Game lifecycle: WAITING (confirm to start) → DEAL_BID → STIRRING →
PLAYING → WAITING (confirm for next round) → ...
Game over is represented by winning_team, not by a phase value.

Push model: every state change triggers exactly one broadcast push with
seq increment. This includes each card dealt during DEAL_BID, each
bid/skip, each stir pass, each play, and each WAITING confirmation.
Error pushes are unicast to the acting player only (seq unchanged).
process_round_result is called immediately when a round ends (PLAYING →
WAITING) so players see scoring + level changes before confirming.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Callable

from server.actions import (
    BidAction,
    DiscardAction,
    NextRoundAction,
    PlayAction,
    SkipBidAction,
    SkipStirAction,
    StirAction,
)
from server.game_protocol import (
    GameAction,
    action_kind,
    bid_event_from_action,
    parse_player_message,
)
from server.game_protocol import (
    resolve_cards as resolve_player_cards,
)
from server.game_push import GameStatePublisher
from server.game_snapshot import (
    build_state_snapshot,
    trump_rank_for_round,
)
from server.player import Player
from server.protocol import (
    PlayerMessage,
    RoundPhase,
    StateMessage,
    StateSnapshot,
)
from server.result import Ok, Rejected
from server.rules.cards import Card
from server.sm import game_sm, round_sm
from server.sm.constants import next_player_ccw
from server.sm.rejections.turn import (
    DuplicateNextRoundConfirmationRejected,
    PlayerActionNotAllowedInRoundPhaseRejected,
    WrongBidTurnRejected,
)

logger = logging.getLogger(__name__)


class Game:
    """
    Aggregate root that orchestrates game lifecycle using sm state
    machines.

    Manages 4 Player instances, drives sync round-robin bidding, and
    provides
    the public API for the server layer.
    """

    def __init__(self, players: Sequence[Player]) -> None:
        self._game_state = game_sm.create_game()
        self._round_state: round_sm.RoundState | None = None
        self._publisher = GameStatePublisher(
            players=players,
            owner=self,
            snapshot_for=self.snapshot,
        )
        self._on_game_over: Callable[[Game], None] | None = None
        self._next_round_confirmed: set[int] = set()
        self._bid_turn: int = 0
        self._act_lock = asyncio.Lock()

    async def _deal_one_and_push(self) -> None:
        """Deal one card and push to all players.

        The player who received the card sees awaiting_action='bid'
        and must act (bid or skip) before the next card is dealt.
        Their action message calls this method again, forming the chain:
        deal → bid/skip → deal → bid/skip → ...
        """
        rs = self._round_state
        assert rs is not None
        if rs.phase != "DEAL_BID":
            return
        if (
            rs.deal_bid_state is None
            or rs.deal_bid_state.phase != "DEALING"
        ):
            return
        if rs.deal_bid_state.all_dealt:
            # All 100 cards dealt — waiting for last recipient to act.
            # Finalization happens after their bid/skip.
            return
        # Remember who receives this card (deal_target before advance)
        recipient = rs.deal_bid_state.deal_target
        match round_sm.deal_next_card(rs):
            case Ok(value=new_rs):
                self._bid_turn = recipient
                self._round_state = new_rs
                await self._push_state_to_all()
            case Rejected() as rejected:
                logger.warning(
                    "deal_next_card rejected: %s", rejected.reason
                )

    async def _run_and_push(self) -> None:
        """
        Start the game after WAITING confirmation: create first round,
        deal, push.

        Called internally when all 4 players confirm in WAITING phase
        and _round_state is None (game has not started yet).
        """
        if self._round_state is not None:
            raise RuntimeError(
                "Game already started; _run_and_push() called with"
                "existing round"
            )
        match game_sm.start_game(self._game_state):
            case Ok(value=new_gs):
                self._game_state = new_gs
            case Rejected() as rejected:
                raise RuntimeError(
                    f"game_sm.start_game rejected: {rejected.reason}"
                )
        self._round_state = round_sm.create_round(
            round_sm.RoundInput(
                declarer_team=self._game_state.declarer_team,
                trump_rank=trump_rank_for_round(self._game_state),
                next_declarer_player=self._game_state.next_declarer_player,
                team0_level=self._game_state.team0_level,
                team1_level=self._game_state.team1_level,
            )
        )
        self._bid_turn = 0

        # Deal the first card — the recipient must bid/skip before
        # the next card is dealt. Their next action calls
        # _deal_one_and_push.
        await self._deal_one_and_push()

    async def receive(
        self, player_index: int, message: PlayerMessage
    ) -> None:
        """Receive one player message through the aggregate root.

        Seq is the protocol gate. If it is unknown (0) or does not match
        the
        current state seq, the server returns the current state without
        interpreting any action fields.
        """
        async with self._act_lock:
            if message.seq == 0 or not self._publisher.accepts_seq(
                message.seq
            ):
                await self._send_state_to_player(
                    player_index, error=None
                )
                return

            parse_result = parse_player_message(
                round_state=self._round_state,
                player_index=player_index,
                message=message,
            )
            if isinstance(parse_result, Rejected):
                await self._send_state_to_player(
                    player_index, error=parse_result.reason
                )
                return

            await self._act_unlocked(player_index, parse_result.value)

    async def _act_unlocked(
        self, player_index: int, action: GameAction
    ) -> None:
        """Dispatch an already seq-validated action by current phase.

        Two push paths, strictly separated:
        - State push: state changed → broadcast to all, seq increments
        - Error push: action rejected → unicast to acting player, seq
        unchanged

        Every state change (including intermediate WAITING
        confirmations)
        triggers a broadcast push + seq increment. No special cases.

        WAITING identity validation rejects duplicate confirmations.

        All runtime action rejections are communicated through the error
        channel instead of exceptions.  Programming errors (e.g. player
        index out of range) propagate as IndexError from the underlying
        list.

        Seq validation happens in receive() before action parsing.
        """
        rs = self._round_state
        phase = self._current_phase()
        logger.debug(
            "Game.receive: player=%d action=%s phase=%s",
            player_index,
            action_kind(action),
            phase,
        )

        error: Rejected | None = None

        if phase == "DEAL_BID" and isinstance(action, BidAction):
            assert rs is not None
            if player_index != self._bid_turn:
                error = WrongBidTurnRejected(self._bid_turn)
            else:
                match bid_event_from_action(
                    player_index=player_index, action=action
                ):
                    case Ok(value=bid_event):
                        match round_sm.reveal(rs, bid_event):
                            case Ok(value=new_state):
                                rs = new_state
                            case Rejected() as rejected:
                                error = rejected
                    case Rejected() as rejected:
                        error = rejected

            if error is not None:
                # Bid rejected — unicast error, no state change, no turn
                # advance.
                # The player must re-decide (choose different cards or
                # pass).
                self._round_state = rs
                await self._send_state_to_player(
                    player_index, error=error.reason
                )
                return
            # Bid succeeded — advance turn
            self._bid_turn = next_player_ccw(self._bid_turn)
            self._round_state = rs
            if (
                rs.deal_bid_state is not None
                and rs.deal_bid_state.all_dealt
            ):
                # Last card recipient bid — finalize deal-bid phase
                match round_sm.finalize_deal_bid(rs):
                    case Ok(value=new_state):
                        self._round_state = new_state
                        await self._push_state_to_all()
                    case Rejected() as rejected:
                        logger.error(
                            "finalize_deal_bid rejected after bid: %s",
                            rejected.reason,
                        )
            else:
                await self._deal_one_and_push()
            return

        elif phase == "DEAL_BID" and isinstance(action, SkipBidAction):
            assert rs is not None
            if player_index != self._bid_turn:
                error = WrongBidTurnRejected(self._bid_turn)
                self._round_state = rs
                await self._send_state_to_player(
                    player_index, error=error.reason
                )
                return
            # Skip succeeded — advance turn
            self._bid_turn = next_player_ccw(self._bid_turn)
            self._round_state = rs
            if (
                rs.deal_bid_state is not None
                and rs.deal_bid_state.all_dealt
            ):
                # Last card recipient skipped — finalize deal-bid phase
                match round_sm.finalize_deal_bid(rs):
                    case Ok(value=new_state):
                        self._round_state = new_state
                        await self._push_state_to_all()
                    case Rejected() as rejected:
                        logger.error(
                            "finalize_deal_bid rejected after skip: %s",
                            rejected.reason,
                        )
            else:
                await self._deal_one_and_push()
            return

        elif phase == "STIRRING" and isinstance(action, SkipStirAction):
            assert rs is not None
            match round_sm.pass_stir(rs, player_index):
                case Ok(value=new_state):
                    rs = new_state
                case Rejected() as rejected:
                    error = rejected

        elif phase == "STIRRING" and isinstance(action, StirAction):
            assert rs is not None
            match round_sm.stir(rs, player_index, action.cards):
                case Ok(value=new_state):
                    rs = new_state
                case Rejected() as rejected:
                    error = rejected

        elif phase == "STIRRING" and isinstance(action, DiscardAction):
            assert rs is not None
            match round_sm.stir_discard(rs, player_index, action.cards):
                case Ok(value=new_state):
                    rs = new_state
                case Rejected() as rejected:
                    error = rejected

        elif phase == "PLAYING" and isinstance(action, PlayAction):
            assert rs is not None
            match round_sm.play(rs, player_index, action.cards):
                case Ok(value=new_state):
                    rs = new_state
                    # If round ended, process result immediately so
                    # players
                    # see scoring + level changes in this push.
                    if rs.phase == "WAITING" and rs.result is not None:
                        round_result = rs.result
                        match game_sm.process_round_result(
                            self._game_state, round_result
                        ):
                            case Ok(value=new_gs):
                                self._game_state = new_gs
                            case Rejected() as rejected:
                                logger.error(
                                    "process_round_result rejected"
                                    "after round completion: %s",
                                    rejected.reason,
                                )
                case Rejected() as rejected:
                    error = rejected
            # Check if game ended after processing the play.
            if (
                error is None
                and rs.phase == "WAITING"
                and self.is_over()
            ):
                self._round_state = rs
                await self._push_state_to_all()
                if self._on_game_over is not None:
                    self._on_game_over(self)
                return

        elif phase == "WAITING" and isinstance(action, NextRoundAction):
            if player_index in self._next_round_confirmed:
                error = DuplicateNextRoundConfirmationRejected()
            else:
                self._next_round_confirmed.add(player_index)
                if len(self._next_round_confirmed) == 4:
                    # All 4 confirmed
                    self._next_round_confirmed.clear()

                    if self._round_state is None:
                        # Game start: create first round, deal, push
                        await self._run_and_push()
                        return

                    # Between rounds: _game_state was already updated
                    # when
                    # the round ended (PLAYING branch calls
                    # process_round_result).
                    if self.is_over():
                        self._round_state = rs
                        await self._push_state_to_all()
                        if self._on_game_over is not None:
                            self._on_game_over(self)
                        return

                    # Create new round and deal cards
                    rs = round_sm.create_round(
                        round_sm.RoundInput(
                            declarer_team=self._game_state.declarer_team,
                            trump_rank=trump_rank_for_round(
                                self._game_state
                            ),
                            next_declarer_player=self._game_state.next_declarer_player,
                            team0_level=self._game_state.team0_level,
                            team1_level=self._game_state.team1_level,
                        )
                    )
                    self._bid_turn = 0
                    self._round_state = rs
                    # Deal the first card — the recipient must bid/skip
                    # before the next card is dealt. No intermediate
                    # push
                    # needed; _deal_one_and_push will broadcast once a
                    # card is dealt. Same pattern as _run_and_push().
                    await self._deal_one_and_push()
                    return
                # else: intermediate confirmation — fall through to
                # _push_state_to_all(). next_round_confirmed changed,
                # that's a state change like any other.

        else:
            error = PlayerActionNotAllowedInRoundPhaseRejected(
                action_kind(action), phase
            )

        self._round_state = rs

        if error is not None:
            # Unicast error to acting player. Error pushes do NOT
            # increment _seq because the game state has not changed.
            await self._send_state_to_player(
                player_index, error=error.reason
            )
        else:
            await self._push_state_to_all()

    def snapshot(self, for_player: int) -> StateSnapshot:
        return build_state_snapshot(
            for_player=for_player,
            game_state=self._game_state,
            round_state=self._round_state,
            bid_turn=self._bid_turn,
            next_round_confirmed=self._next_round_confirmed,
        )

    def is_over(self) -> bool:
        """Return True if the game is over."""
        return self._game_state.winning_team is not None

    def is_started(self) -> bool:
        """Return True after the first round has been created."""
        return self._round_state is not None

    def _current_phase(self) -> RoundPhase:
        """Return the current round/player-visible phase.

        WAITING: game not started yet (_round_state is None) or round
        complete (rs.phase == "WAITING"). Both use the same WAITING
        phase with next_round confirmation mechanism.
        Game over is represented by winning_team, not by a phase value.
        """
        if self._round_state is None:
            return "WAITING"
        return self._round_state.phase

    def set_on_game_over(
        self, callback: Callable[[Game], None]
    ) -> None:
        """
        Register a callback for when the game reaches a winning team.
        """
        self._on_game_over = callback

    def get_player(self, index: int) -> Player:
        """Return the Player at the given index.

        Raises IndexError if the index is out of range.
        """
        return self._publisher.player(index)

    def resolve_cards(
        self, player_index: int, card_ids: list[str]
    ) -> Ok[list[Card]] | Rejected:
        return resolve_player_cards(
            round_state=self._round_state,
            player_index=player_index,
            card_ids=card_ids,
        )

    def _state_message_for(
        self, player_index: int, error: str | None = None
    ) -> StateMessage:
        return self._publisher.state_message_for(
            player_index, error=error
        )

    async def _send_state_to_player(
        self, player_index: int, error: str | None = None
    ) -> None:
        await self._publisher.send_to_player(player_index, error=error)

    async def _push_state_to_all(self) -> None:
        """Push state to all players."""
        await self._publisher.push_to_all()
