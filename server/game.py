"""Game aggregate root for the Tractor game.

Wraps sm state machines, manages 4 Player instances, drives the dealing loop,
and provides act(), run(), snapshot(), is_over(), get_phase(), set_on_game_over(),
get_player(), cancel(), and resolve_cards() interfaces.
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
    SkipStirAction,
    StirAction,
)
from server.player import Player
from server.sm import game_sm, play_rules, round_sm
from server.sm.card_model import Card
from server.sm.result import Ok, Rejected
from server.sm.types import BidEvent
from server.snapshot import (
    ExchangeStateSnapshot,
    ScoringSnapshot,
    StateSnapshot,
    StirringStateSnapshot,
    TrickSnapshot,
    TrickSlotSnapshot,
)

logger = logging.getLogger(__name__)


class Game:
    """Aggregate root that orchestrates game lifecycle using sm state machines.

    Manages 4 Player instances, drives the dealing loop, and provides
    the public API for the server layer.
    """

    def __init__(self, players: Sequence[Player], *, deal_delay: float = 0.5) -> None:
        self._game_state = game_sm.create_game()
        self._round_state: round_sm.RoundState | None = None
        self._players = list(players)
        self._dealing_task: asyncio.Task[None] | None = None
        self._on_game_over: Callable[['Game'], None] | None = None
        self._cancelled: bool = False
        self._next_round_confirmed: set[int] = set()
        self._deal_delay = deal_delay

    async def run(self) -> None:
        """Start the game: transition to IN_ROUND, create round, start dealing loop.

        Raises RuntimeError if called more than once.
        Raises RuntimeError if game_sm.start_game rejects (should never happen).
        """
        if self._round_state is not None:
            raise RuntimeError("Game already started; run() can only be called once")
        match game_sm.start_game(self._game_state):
            case Ok(value=new_gs):
                self._game_state = new_gs
            case Rejected(reason=reason):
                raise RuntimeError(f"game_sm.start_game rejected: {reason}")
        self._round_state = round_sm.create_round(round_sm.RoundInput(
            declarer_team=self._game_state.declarer_team,
            trump_rank=self._game_state.team0_level,  # trump rank starts at team0_level
            last_declarer_player=self._game_state.last_declarer_player,
            team0_level=self._game_state.team0_level,
            team1_level=self._game_state.team1_level,
        ))
        self._cancelled = False
        self._dealing_task = asyncio.create_task(self._dealing_loop())

    async def act(self, player_index: int, action: BidAction | PlayAction | StirAction | SkipStirAction | DiscardAction | NextRoundAction) -> None:
        """Unified action entry point. Dispatches based on current phase and action type.

        After applying the action, pushes state to all players.
        Any rejection (invalid action, race condition, etc.) sends an error
        message to the acting player via WebSocket, then still pushes state
        so the game never deadlocks.

        All runtime action rejections are communicated through the error
        channel instead of exceptions.  Programming errors (e.g. player
        index out of range) propagate as IndexError from the underlying list.
        """
        rs = self._round_state
        assert rs is not None, "act() called before run()"
        phase = self.get_phase()
        logger.debug("Game.act: player=%d action=%s phase=%s", player_index, type(action).__name__, phase)

        error_msg: str | None = None
        should_push = True

        if phase == "DEAL_BID" and isinstance(action, BidAction):
            match self._convert_bid_action(player_index, action):
                case Ok(value=bid_event):
                    match round_sm.reveal(rs, bid_event):
                        case Ok(value=new_state):
                            rs = new_state
                        case Rejected(reason=reason):
                            error_msg = reason
                case Rejected(reason=reason):
                    error_msg = reason
            # Dealing loop already pushes state every 0.5s; avoid extra push
            # here to prevent AutoPlayer bid cascades.
            should_push = False

        elif phase == "STIRRING" and isinstance(action, SkipStirAction):
            match round_sm.pass_stir(rs):
                case Ok(value=new_state):
                    rs = new_state
                case Rejected(reason=reason):
                    error_msg = reason

        elif phase == "STIRRING" and isinstance(action, StirAction):
            match round_sm.stir(rs, action.cards):
                case Ok(value=new_state):
                    rs = new_state
                case Rejected(reason=reason):
                    error_msg = reason

        elif phase == "EXCHANGE" and isinstance(action, DiscardAction):
            match round_sm.discard(rs, action.cards):
                case Ok(value=new_state):
                    rs = new_state
                case Rejected(reason=reason):
                    error_msg = reason

        elif phase == "PLAYING" and isinstance(action, PlayAction):
            match round_sm.play(rs, player_index, action.cards):
                case Ok(value=new_state):
                    rs = new_state
                case Rejected(reason=reason):
                    error_msg = reason

        elif phase == "COMPLETE" and isinstance(action, NextRoundAction):
            self._next_round_confirmed.add(player_index)

            if len(self._next_round_confirmed) == 4:
                # All 4 players confirmed: proceed to next round
                self._next_round_confirmed.clear()
                round_result = round_sm.get_round_result(rs)
                assert round_result is not None, "Round result is None in COMPLETE phase; this indicates an sm layer bug"
                match game_sm.process_round_result(self._game_state, round_result):
                    case Ok(value=new_gs):
                        self._game_state = new_gs
                    case Rejected(reason=reason):
                        error_msg = f"处理回合结果失败：{reason}"

                if error_msg is None and self._game_state.phase == "GAME_OVER":
                    self._round_state = rs
                    await self._push_state_to_all()
                    if self._on_game_over is not None:
                        self._on_game_over(self)
                    return
                else:
                    self._cancelled = False
                    rs = round_sm.create_round(round_sm.RoundInput(
                        declarer_team=self._game_state.declarer_team,
                        trump_rank=self._game_state.team0_level,
                        last_declarer_player=self._game_state.last_declarer_player,
                        team0_level=self._game_state.team0_level,
                        team1_level=self._game_state.team1_level,
                    ))
                    # Start a new dealing loop for the next round
                    self._dealing_task = asyncio.create_task(self._dealing_loop())

        else:
            error_msg = f"无效的操作：{type(action).__name__} 不能在 {phase} 阶段使用"

        self._round_state = rs

        if error_msg:
            await self._send_error_to_player(player_index, error_msg)

        if should_push:
            await self._push_state_to_all()

    def snapshot(self, for_player: int) -> StateSnapshot:
        """Build a StateSnapshot for the given player.

        Raises IndexError if for_player is out of range.
        """
        rs = self._round_state
        assert rs is not None, "snapshot() called before run()"
        gs = self._game_state

        # player_hand
        player_hand = list(rs.players_hand[for_player]) if for_player < len(rs.players_hand) else []

        # player_hand_counts: card count for each player (for game table display)
        player_hand_counts = [len(h) for h in rs.players_hand]

        # current_player derivation
        current_player = for_player  # default
        if rs.phase == "DEAL_BID" and rs.deal_bid_state is not None:
            current_player = rs.deal_bid_state.deal_target
        elif rs.phase == "STIRRING" and rs.stirring_state is not None:
            current_player = rs.stirring_state.current_player
        elif rs.phase == "EXCHANGE" and rs.exchange_state is not None:
            current_player = rs.exchange_state.declarer_player
        elif rs.phase == "PLAYING" and rs.trick_state is not None:
            current_player = rs.trick_state.cur
        elif rs.phase == "COMPLETE":
            current_player = rs.declarer_player if rs.declarer_player is not None else 0

        # legal_actions
        legal_actions: list[list[Card]] = []
        can_act_in_playing = False  # whether current player can act in PLAYING
        if rs.phase == "PLAYING" and rs.trick_state is not None:
            is_leading = rs.trick_state.phase == "LEADING"
            lead_cards = None
            if is_leading:
                can_act_in_playing = True
            else:
                # Following: only compute legal actions if lead cards exist
                lead_slots = rs.trick_state.slots
                if lead_slots:
                    lead_cards = lead_slots[rs.trick_state.lead_player].cards
                    if lead_cards:
                        can_act_in_playing = True
                    # else: lead player hasn't played yet, followers must wait
            if can_act_in_playing:
                # Compute other_hands: all cards not in current player's hand
                other_hands: list[Card] = []
                for i in range(4):
                    if i != for_player:
                        other_hands.extend(rs.players_hand[i])
                legal_actions = play_rules.get_legal_plays(
                    hand=player_hand,
                    is_leading=is_leading,
                    lead_cards=lead_cards,
                    trump_suit=rs.trump_suit,
                    trump_rank=rs.trump_rank,
                    other_hands=other_hands,
                )
                # Safety: if legal_actions is empty despite can_act, no valid play yet
                if not legal_actions:
                    can_act_in_playing = False

        # awaiting_action
        awaiting_action = None
        if rs.phase == "STIRRING":
            awaiting_action = "stir"
        elif rs.phase == "EXCHANGE":
            awaiting_action = "discard"
        elif rs.phase == "PLAYING" and can_act_in_playing:
            awaiting_action = "play"
        elif rs.phase == "COMPLETE":
            awaiting_action = "next_round"

        # trick
        trick: TrickSnapshot | None = None
        if rs.phase == "PLAYING" and rs.trick_state is not None:
            ts = rs.trick_state
            trick = TrickSnapshot(
                lead_player=ts.lead_player,
                slots=[TrickSlotSnapshot(player=slot.player, cards=slot.cards) for slot in ts.slots],
                current_player=ts.cur,
            )

        # bid_events and bid_winner
        bid_events: list[BidEvent] = []
        bid_winner: BidEvent | None = None
        if rs.deal_bid_state is not None:
            bid_events = list(rs.deal_bid_state.bid_events)
            bid_winner = rs.deal_bid_state.bid_winner

        # stirring_state
        stirring_state_snap: StirringStateSnapshot | None = None
        if rs.stirring_state is not None and rs.phase == "STIRRING":
            stirring_state_snap = StirringStateSnapshot(
                phase=rs.stirring_state.phase,
                trump_suit=rs.stirring_state.trump_suit,
                current_player=rs.stirring_state.current_player,
            )

        # exchange_state
        exchange_state_snap: ExchangeStateSnapshot | None = None
        if rs.exchange_state is not None and rs.phase == "EXCHANGE":
            exchange_state_snap = ExchangeStateSnapshot(
                phase=rs.exchange_state.phase,
                declarer_player=rs.exchange_state.declarer_player,
                count=rs.exchange_state.count,
            )

        # scoring
        scoring_snap: ScoringSnapshot | None = None
        if rs.result is not None:
            scoring_snap = ScoringSnapshot(
                declarer_team=rs.declarer_team,
                defender_points=rs.defender_points,
                bottom_cards=list(rs.bottom_cards),
            )

        return StateSnapshot(
            phase=self.get_phase(),
            player_hand=player_hand,
            player_hand_counts=player_hand_counts,
            bottom_cards=list(rs.bottom_cards),
            trump_suit=rs.trump_suit,
            trump_rank=rs.trump_rank,
            declarer_team=rs.declarer_team,
            declarer_player=rs.declarer_player,
            current_player=current_player,
            defender_points=rs.defender_points,
            trick=trick,
            trick_history=list(rs.trick_history),
            legal_actions=legal_actions,
            awaiting_action=awaiting_action,
            scoring=scoring_snap,
            winning_team=gs.winning_team,
            team0_level=gs.team0_level,
            team1_level=gs.team1_level,
            bid_events=bid_events,
            bid_winner=bid_winner,
            stirring_state=stirring_state_snap,
            exchange_state=exchange_state_snap,
            next_round_confirmed=sorted(self._next_round_confirmed),
        )

    def is_over(self) -> bool:
        """Return True if the game is over."""
        return self._game_state.phase == "GAME_OVER"

    def get_phase(self) -> str:
        """Return the current phase.

        GAME_OVER takes priority over round-level phases.
        """
        if self._game_state.phase == "GAME_OVER":
            return "GAME_OVER"
        if self._round_state is not None:
            return self._round_state.phase
        return self._game_state.phase

    def set_on_game_over(self, callback: Callable[['Game'], None]) -> None:
        """Register a callback for when the game transitions to GAME_OVER."""
        self._on_game_over = callback

    def get_player(self, index: int) -> Player:
        """Return the Player at the given index.

        Raises IndexError if the index is out of range.
        """
        return self._players[index]

    async def cancel(self) -> None:
        """Stop the dealing loop background task."""
        self._cancelled = True
        if self._dealing_task is not None and not self._dealing_task.done():
            self._dealing_task.cancel()
            try:
                await self._dealing_task
            except asyncio.CancelledError:
                pass

    def resolve_cards(self, player_index: int, card_ids: list[str]) -> Ok[list[Card]] | Rejected:
        """Resolve card ID strings to Card objects from the player's hand.

        Returns Rejected if any card_id is not found in the player's hand.
        Raises IndexError if player_index is out of range (programming error).
        """
        rs = self._round_state
        assert rs is not None, "resolve_cards() called before run()"
        hand = rs.players_hand[player_index]
        card_map = {c.id: c for c in hand}

        result: list[Card] = []
        for card_id in card_ids:
            if card_id not in card_map:
                return Rejected(reason=f"Card {card_id} not in hand of player {player_index}")
            result.append(card_map[card_id])
        return Ok(value=result)

    async def _dealing_loop(self) -> None:
        """Background coroutine that deals cards one at a time.

        Checks _cancelled at the start of each iteration.
        Sleeps 0.75s between deals. Pushes state to all players after each deal.
        When dealing completes, transitions to STIRRING automatically via sm.

        _push_state_to_all() is resilient to individual player failures,
        so a disconnected human player will not crash this loop.
        Any unexpected exception indicates a programming bug and will crash
        the task rather than silently leaving the game in a stuck state.
        """
        while not self._cancelled:
            if self._round_state is None:
                break
            if self._round_state.phase != "DEAL_BID":
                break
            if self._round_state.deal_bid_state is None:
                break

            match round_sm.deal_next_card(self._round_state):
                case Ok(value=new_rs):
                    self._round_state = new_rs
                case Rejected(reason=reason):
                    logger.warning("deal_next_card rejected: %s", reason)
                    break
            await self._push_state_to_all()

            if self._round_state.phase != "DEAL_BID":
                break

            await asyncio.sleep(self._deal_delay)

    def _convert_bid_action(self, player_index: int, action: BidAction) -> Ok[BidEvent] | Rejected:
        """Convert a player BidAction to an sm BidEvent."""
        cards = action.cards
        if not cards:
            return Rejected(reason="BidAction requires at least one card")
        # Determine kind, suit, joker_type from the cards
        if cards[0].is_joker:
            kind = "joker"
            joker_type = "big" if cards[0].is_big_joker else "small"
            suit = None
        else:
            kind = "trump_rank"
            suit = cards[0].suit
            joker_type = None

        return Ok(value=BidEvent(
            player=player_index,
            cards=cards,
            kind=kind,
            suit=suit,
            joker_type=joker_type,
            count=action.count,
        ))

    async def _send_error_to_player(self, player_index: int, message: str) -> None:
        """Send an error message to a specific player."""
        await self._players[player_index].send_error(message)

    async def _push_state_to_all(self) -> None:
        """Push state to all players."""
        for i in range(len(self._players)):
            await self._players[i].on_state(self)
