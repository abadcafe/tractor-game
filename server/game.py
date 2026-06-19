"""Game aggregate root for the Tractor game.

Wraps sm state machines, manages 4 Player instances, drives the sync
round-robin bidding, and provides receive(), snapshot(),
is_over(), get_phase(), set_on_game_over(), get_player(), and
resolve_cards() interfaces.

Game lifecycle: WAITING (confirm to start) → DEAL_BID → STIRRING →
PLAYING → WAITING (confirm for next round) → ... → GAME_OVER.

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
from collections import defaultdict
from collections.abc import Sequence
from typing import Callable, TypeGuard

from server.actions import (
    BidAction,
    DiscardAction,
    NextRoundAction,
    PlayAction,
    SkipBidAction,
    SkipStirAction,
    StirAction,
)
from server.messages import PlayerMessage, StateMessage
from server.player import Player
from server.sm import deal_bid_sm, game_sm, play_rules, round_sm, stirring_sm
from server.sm.comparator import bid_value
from server.sm.card_model import Card, Rank, Suit
from server.sm.result import Ok, Rejected, StateResult
from server.sm.types import BidEvent
from server.snapshot import (
    ScoringSnapshot,
    StateSnapshot,
    StirringStateSnapshot,
    TrickSnapshot,
    TrickSlotSnapshot,
)

logger = logging.getLogger(__name__)

type GameAction = BidAction | SkipBidAction | PlayAction | StirAction | SkipStirAction | DiscardAction | NextRoundAction


class Game:
    """Aggregate root that orchestrates game lifecycle using sm state machines.

    Manages 4 Player instances, drives sync round-robin bidding, and provides
    the public API for the server layer.
    """

    def __init__(self, players: Sequence[Player]) -> None:
        self._game_state = game_sm.create_game()
        self._round_state: round_sm.RoundState | None = None
        self._players = list(players)
        self._on_game_over: Callable[['Game'], None] | None = None
        self._next_round_confirmed: set[int] = set()
        self._seq: int = 1
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
        if rs.deal_bid_state is None or rs.deal_bid_state.phase != "DEALING":
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
            case Rejected(reason=reason):
                logger.warning("deal_next_card rejected: %s", reason)

    async def _run_and_push(self) -> None:
        """Start the game after WAITING confirmation: create first round, deal, push.

        Called internally when all 4 players confirm in WAITING phase
        and _round_state is None (game has not started yet).
        """
        if self._round_state is not None:
            raise RuntimeError("Game already started; _run_and_push() called with existing round")
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
        self._bid_turn = 0

        # Deal the first card — the recipient must bid/skip before
        # the next card is dealt. Their next action calls _deal_one_and_push.
        await self._deal_one_and_push()

    async def receive(self, player_index: int, message: PlayerMessage) -> None:
        """Receive one player message through the aggregate root.

        Seq is the protocol gate. If it is unknown (0) or does not match the
        current state seq, the server returns the current state without
        interpreting any action fields.
        """
        async with self._act_lock:
            if message.seq == 0 or message.seq != self._seq:
                await self._send_state_to_player(player_index, error=None)
                return

            parse_result = self._parse_player_message(player_index, message)
            if isinstance(parse_result, Rejected):
                await self._send_state_to_player(player_index, error=parse_result.reason)
                return

            await self._act_unlocked(player_index, parse_result.value)

    async def _act_unlocked(self, player_index: int, action: GameAction) -> None:
        """Dispatch an already seq-validated action by current phase.

        Two push paths, strictly separated:
        - State push: state changed → broadcast to all, seq increments
        - Error push: action rejected → unicast to acting player, seq unchanged

        Every state change (including intermediate WAITING confirmations)
        triggers a broadcast push + seq increment. No special cases.

        WAITING identity validation rejects duplicate confirmations.

        All runtime action rejections are communicated through the error
        channel instead of exceptions.  Programming errors (e.g. player
        index out of range) propagate as IndexError from the underlying list.

        Seq validation happens in receive() before action parsing.
        """
        rs = self._round_state
        phase = self.get_phase()
        logger.debug("Game.receive: player=%d action=%s phase=%s", player_index, type(action).__name__, phase)

        error_msg: str | None = None

        if phase == "DEAL_BID" and isinstance(action, BidAction):
            assert rs is not None
            if player_index != self._bid_turn:
                error_msg = f"不是你的叫牌回合（当前叫牌者：{self._bid_turn}）"
            else:
                match self._convert_bid_action(player_index, action):
                    case Ok(value=bid_event):
                        match round_sm.reveal(rs, bid_event):
                            case Ok(value=new_state):
                                rs = new_state
                            case Rejected(reason=reason):
                                error_msg = reason
                    case Rejected(reason=reason):
                        error_msg = reason

            if error_msg is not None:
                # Bid rejected — unicast error, no state change, no turn advance.
                # The player must re-decide (choose different cards or pass).
                self._round_state = rs
                await self._send_state_to_player(player_index, error=error_msg)
                return
            # Bid succeeded — advance turn
            self._bid_turn = (self._bid_turn + 1) % 4
            self._round_state = rs
            if rs.deal_bid_state is not None and rs.deal_bid_state.all_dealt:
                # Last card recipient bid — finalize deal-bid phase
                match round_sm.finalize_deal_bid(rs):
                    case Ok(value=new_state):
                        self._round_state = new_state
                        await self._push_state_to_all()
                    case Rejected(reason=reason):
                        logger.error("finalize_deal_bid rejected after bid: %s", reason)
            else:
                await self._deal_one_and_push()
            return

        elif phase == "DEAL_BID" and isinstance(action, SkipBidAction):
            assert rs is not None
            if player_index != self._bid_turn:
                error_msg = f"不是你的叫牌回合（当前叫牌者：{self._bid_turn}）"
                self._round_state = rs
                await self._send_state_to_player(player_index, error=error_msg)
                return
            # Skip succeeded — advance turn
            self._bid_turn = (self._bid_turn + 1) % 4
            self._round_state = rs
            if rs.deal_bid_state is not None and rs.deal_bid_state.all_dealt:
                # Last card recipient skipped — finalize deal-bid phase
                match round_sm.finalize_deal_bid(rs):
                    case Ok(value=new_state):
                        self._round_state = new_state
                        await self._push_state_to_all()
                    case Rejected(reason=reason):
                        logger.error("finalize_deal_bid rejected after skip: %s", reason)
            else:
                await self._deal_one_and_push()
            return

        elif phase == "STIRRING" and isinstance(action, SkipStirAction):
            assert rs is not None
            match round_sm.pass_stir(rs, player_index):
                case Ok(value=new_state):
                    rs = new_state
                case Rejected(reason=reason):
                    error_msg = reason

        elif phase == "STIRRING" and isinstance(action, StirAction):
            assert rs is not None
            match round_sm.stir(rs, player_index, action.cards):
                case Ok(value=new_state):
                    rs = new_state
                case Rejected(reason=reason):
                    error_msg = reason

        elif phase == "STIRRING" and isinstance(action, DiscardAction):
            assert rs is not None
            match round_sm.stir_discard(rs, player_index, action.cards):
                case Ok(value=new_state):
                    rs = new_state
                case Rejected(reason=reason):
                    error_msg = reason

        elif phase == "PLAYING" and isinstance(action, PlayAction):
            assert rs is not None
            match round_sm.play(rs, player_index, action.cards):
                case Ok(value=new_state):
                    rs = new_state
                    # If round ended, process result immediately so players
                    # see scoring + level changes in this push.
                    if rs.phase == "WAITING" and rs.result is not None:
                        round_result = rs.result
                        match game_sm.process_round_result(self._game_state, round_result):
                            case Ok(value=new_gs):
                                self._game_state = new_gs
                            case Rejected(reason=reason):
                                logger.error("process_round_result rejected after round completion: %s", reason)
                case Rejected(reason=reason):
                    error_msg = reason
            # Check if game ended after processing the play
            if error_msg is None and rs.phase == "WAITING" and self._game_state.phase == "GAME_OVER":
                self._round_state = rs
                await self._push_state_to_all()
                if self._on_game_over is not None:
                    self._on_game_over(self)
                return

        elif phase == "WAITING" and isinstance(action, NextRoundAction):
            if player_index in self._next_round_confirmed:
                error_msg = "你已经确认过了"
            else:
                self._next_round_confirmed.add(player_index)
                if len(self._next_round_confirmed) == 4:
                    # All 4 confirmed
                    self._next_round_confirmed.clear()

                    if self._round_state is None:
                        # Game start: create first round, deal, push
                        await self._run_and_push()
                        return

                    # Between rounds: _game_state was already updated when
                    # the round ended (PLAYING branch calls process_round_result).
                    if self._game_state.phase == "GAME_OVER":
                        self._round_state = rs
                        await self._push_state_to_all()
                        if self._on_game_over is not None:
                            self._on_game_over(self)
                        return

                    # Create new round and deal cards
                    rs = round_sm.create_round(round_sm.RoundInput(
                        declarer_team=self._game_state.declarer_team,
                        trump_rank=self._game_state.team0_level,
                        last_declarer_player=self._game_state.last_declarer_player,
                        team0_level=self._game_state.team0_level,
                        team1_level=self._game_state.team1_level,
                    ))
                    self._bid_turn = 0
                    self._round_state = rs
                    # Deal the first card — the recipient must bid/skip
                    # before the next card is dealt. No intermediate push
                    # needed; _deal_one_and_push will broadcast once a
                    # card is dealt. Same pattern as _run_and_push().
                    await self._deal_one_and_push()
                    return
                # else: intermediate confirmation — fall through to
                # _push_state_to_all(). next_round_confirmed changed,
                # that's a state change like any other.

        else:
            error_msg = f"无效的操作：{type(action).__name__} 不能在 {phase} 阶段使用"

        self._round_state = rs

        if error_msg:
            # Unicast error to acting player. Error pushes do NOT
            # increment _seq because the game state has not changed.
            await self._send_state_to_player(player_index, error=error_msg)
        else:
            await self._push_state_to_all()

    @staticmethod
    def _get_legal_stir_actions(
        hand: list[Card],
        stirring_state: stirring_sm.StirringState,
        player_index: int,
    ) -> list[list[Card]]:
        """Compute legal stir actions for a player's hand.

        Returns pairs of trump-rank cards or joker pairs that have priority
        exceeding the current trump. Returns empty list if the player is
        stirring_state.last_stir_player (can't stir own trump).
        """
        # Can't stir own trump
        if stirring_state.last_stir_player == player_index:
            return []

        trump_rank = stirring_state.trump_rank
        current_priority = stirring_state.current_priority

        # Group trump-rank cards by suit
        by_suit: dict[Suit, list[Card]] = defaultdict(list)
        small_jokers: list[Card] = []
        big_jokers: list[Card] = []

        for c in hand:
            if c.is_joker:
                if c.rank == Rank.SMALL_JOKER:
                    small_jokers.append(c)
                else:
                    big_jokers.append(c)
            elif c.rank == trump_rank:
                by_suit[c.suit].append(c)

        result: list[list[Card]] = []

        # Same-suit trump-rank pairs
        for suit_cards in by_suit.values():
            if len(suit_cards) >= 2:
                pair = [suit_cards[0], suit_cards[1]]
                if bid_value(pair, trump_rank) > current_priority:
                    result.append(pair)

        # Small joker pair
        if len(small_jokers) >= 2:
            pair = [small_jokers[0], small_jokers[1]]
            if bid_value(pair, trump_rank) > current_priority:
                result.append(pair)

        # Big joker pair
        if len(big_jokers) >= 2:
            pair = [big_jokers[0], big_jokers[1]]
            if bid_value(pair, trump_rank) > current_priority:
                result.append(pair)

        return result

    def snapshot(self, for_player: int) -> StateSnapshot:
        """Build a StateSnapshot for the given player.

        Handles WAITING phase before game start (_round_state is None)
        by returning a minimal snapshot with empty hands and no round data.

        Raises IndexError if for_player is out of range.
        """
        rs = self._round_state
        gs = self._game_state

        # WAITING phase before game start: no round state yet
        if rs is None:
            awaiting_action: str | None = "next_round" if for_player not in self._next_round_confirmed else None
            return StateSnapshot(
                phase="WAITING",
                player_hand=[],
                player_hand_counts=[0, 0, 0, 0],
                bottom_cards=[],
                trump_suit=None,
                trump_rank=gs.team0_level,
                declarer_team=None,
                declarer_player=None,
                defender_points=0,
                trick=None,
                trick_history=[],
                failed_throw=None,
                action_hints=[],
                awaiting_action=awaiting_action,
                scoring=None,
                winning_team=None,
                team0_level=gs.team0_level,
                team1_level=gs.team1_level,
                bid_events=[],
                bid_winner=None,
                stirring_state=None,
                next_round_confirmed=sorted(self._next_round_confirmed),
            )

        # player_hand
        player_hand = list(rs.players_hand[for_player]) if for_player < len(rs.players_hand) else []

        # player_hand_counts: card count for each player (for game table display)
        player_hand_counts = [len(h) for h in rs.players_hand]

        if (
            rs.phase == "STIRRING"
            and rs.stirring_state is not None
            and rs.stirring_state.phase == "EXCHANGING"
            and rs.stirring_state.exchanging_player == for_player
            and rs.stirring_state.exchange_state is not None
        ):
            player_hand = list(rs.stirring_state.exchange_state.hand_after_pickup)
            player_hand_counts[for_player] = len(player_hand)

        can_act_in_playing = False  # whether current player can act in PLAYING
        if rs.phase == "PLAYING" and rs.trick_state is not None:
            is_leading = rs.trick_state.phase == "LEADING"
            if is_leading:
                can_act_in_playing = True
            else:
                # Following: only act if lead cards exist
                lead_slots = rs.trick_state.slots
                if lead_slots:
                    lead_cards = lead_slots[rs.trick_state.lead_player].cards
                    if lead_cards:
                        can_act_in_playing = True
                    # else: lead player hasn't played yet, followers must wait

        # awaiting_action — derived directly from SM state (no current_player)
        awaiting_action = None
        if gs.phase == "GAME_OVER":
            awaiting_action = None
        elif rs.phase == "DEAL_BID":
            # Only the player who just received a card sees
            # awaiting_action='bid' and must act before the next card
            # is dealt.
            if for_player == self._bid_turn:
                awaiting_action = "bid"
            else:
                awaiting_action = None
        elif rs.phase == "STIRRING" and rs.stirring_state is not None:
            if rs.stirring_state.phase == "EXCHANGING" and for_player == rs.stirring_state.exchanging_player:
                awaiting_action = "discard"
            elif rs.stirring_state.phase == "WAITING" and for_player == rs.stirring_state.current_player:
                awaiting_action = "stir"
        elif rs.phase == "PLAYING" and can_act_in_playing and rs.trick_state is not None and for_player == rs.trick_state.cur:
            awaiting_action = "play"
        elif rs.phase == "WAITING":
            if for_player not in self._next_round_confirmed:
                awaiting_action = "next_round"
            else:
                awaiting_action = None

        action_hints = self._get_action_hints(awaiting_action, for_player, player_hand)

        # trick
        trick: TrickSnapshot | None = None
        if rs.phase == "PLAYING" and rs.trick_state is not None:
            ts = rs.trick_state
            trick = TrickSnapshot(
                lead_player=ts.lead_player,
                slots=[TrickSlotSnapshot(player=slot.player, cards=slot.cards) for slot in ts.slots],
                current_player=ts.cur,
            )
        failed_throw = rs.trick_state.failed_throw if rs.phase == "PLAYING" and rs.trick_state is not None else None

        # bid_events and bid_winner
        bid_events: list[BidEvent] = []
        bid_winner: BidEvent | None = None
        if rs.deal_bid_state is not None:
            bid_events = list(rs.deal_bid_state.bid_events)
            bid_winner = rs.deal_bid_state.bid_winner

        # stirring_state
        stirring_state_snap: StirringStateSnapshot | None = None
        if rs.stirring_state is not None and rs.phase == "STIRRING":
            exchanging_player: int | None = None
            exchange_count: int | None = None
            if rs.stirring_state.phase == "EXCHANGING":
                exchanging_player = rs.stirring_state.exchanging_player
                if rs.stirring_state.exchange_state is not None:
                    exchange_count = rs.stirring_state.exchange_state.count
            stirring_state_snap = StirringStateSnapshot(
                phase=rs.stirring_state.phase,
                trump_suit=rs.stirring_state.trump_suit,
                current_player=rs.stirring_state.current_player,
                declarer_player=rs.stirring_state.declarer_player,
                exchanging_player=exchanging_player,
                exchange_count=exchange_count,
            )

        # scoring
        scoring_snap: ScoringSnapshot | None = None
        if rs.result is not None:
            scoring_snap = ScoringSnapshot(
                declarer_team=rs.declarer_team,
                defender_points=rs.defender_points,
                total_defender_points=rs.result.total_defender_points,
                bottom_card_bonus=rs.result.bottom_card_bonus,
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
            defender_points=rs.defender_points,
            trick=trick,
            trick_history=list(rs.trick_history),
            failed_throw=failed_throw,
            action_hints=action_hints,
            awaiting_action=awaiting_action,
            scoring=scoring_snap,
            winning_team=gs.winning_team,
            team0_level=gs.team0_level,
            team1_level=gs.team1_level,
            bid_events=bid_events,
            bid_winner=bid_winner,
            stirring_state=stirring_state_snap,
            next_round_confirmed=sorted(self._next_round_confirmed),
        )

    def _get_action_hints(
        self,
        awaiting_action: str | None,
        player_index: int,
        player_hand: list[Card],
    ) -> list[list[Card]]:
        """Return advisory card-group hints for the current awaiting action."""
        rs = self._round_state
        if rs is None:
            return []

        if awaiting_action == "bid" and rs.phase == "DEAL_BID" and rs.deal_bid_state is not None:
            return deal_bid_sm.get_bid_action_hints(rs.deal_bid_state, player_index)

        if awaiting_action == "stir" and rs.phase == "STIRRING" and rs.stirring_state is not None:
            return self._get_legal_stir_actions(player_hand, rs.stirring_state, player_index)

        if awaiting_action == "play" and rs.phase == "PLAYING":
            hints = self._get_play_action_hints(player_index)
            if len(hints) > play_rules.MAX_LEGAL_PLAY_HINTS:
                return []
            return hints

        return []

    def _get_play_action_hints(self, player_index: int) -> list[list[Card]]:
        """Compute complete play hints for the player-facing snapshot."""
        rs = self._round_state
        if rs is None or rs.phase != "PLAYING" or rs.trick_state is None:
            return []
        if player_index != rs.trick_state.cur:
            return []

        player_hand = list(rs.players_hand[player_index])
        is_leading = rs.trick_state.phase == "LEADING"
        lead_cards = None
        if not is_leading:
            lead_slots = rs.trick_state.slots
            if not lead_slots:
                return []
            lead_cards = lead_slots[rs.trick_state.lead_player].cards
            if not lead_cards:
                return []

        other_hands: list[Card] = []
        for i in range(4):
            if i != player_index:
                other_hands.extend(rs.players_hand[i])

        return play_rules.get_legal_plays(
            hand=player_hand,
            is_leading=is_leading,
            lead_cards=lead_cards,
            trump_suit=rs.trump_suit,
            trump_rank=rs.trump_rank,
            other_hands=other_hands,
        )

    def is_over(self) -> bool:
        """Return True if the game is over."""
        return self._game_state.phase == "GAME_OVER"

    def get_phase(self) -> str:
        """Return the current phase.

        WAITING: game not started yet (_round_state is None) or round
        complete (rs.phase == "WAITING"). Both use the same WAITING
        phase with next_round confirmation mechanism.
        GAME_OVER takes priority over round-level phases.
        """
        if self._game_state.phase == "GAME_OVER":
            return "GAME_OVER"
        if self._round_state is None:
            return "WAITING"
        return self._round_state.phase

    def set_on_game_over(self, callback: Callable[['Game'], None]) -> None:
        """Register a callback for when the game transitions to GAME_OVER."""
        self._on_game_over = callback

    def get_player(self, index: int) -> Player:
        """Return the Player at the given index.

        Raises IndexError if the index is out of range.
        """
        return self._players[index]

    def _parse_player_message(self, player_index: int, message: PlayerMessage) -> StateResult[GameAction]:
        raw = message.raw
        t = raw.get("type")
        action_type: str | None = t if isinstance(t, str) else None
        if action_type is None:
            return Rejected(reason="missing action type")

        pass_val_raw = raw.get("pass", False)
        is_pass = isinstance(pass_val_raw, bool) and pass_val_raw

        if action_type == "bid":
            if is_pass:
                return Ok(value=SkipBidAction())
            return self._parse_card_action(player_index, raw.get("cards"), action_type)

        if action_type == "stir":
            if is_pass:
                return Ok(value=SkipStirAction())
            return self._parse_card_action(player_index, raw.get("cards"), action_type)

        if action_type in ("discard", "play"):
            return self._parse_card_action(player_index, raw.get("cards"), action_type)

        if action_type == "next_round":
            return Ok(value=NextRoundAction())

        return Rejected(reason=f"unknown action type: {action_type}")

    def _parse_card_action(
        self,
        player_index: int,
        cards_raw: object,
        action_type: str,
    ) -> StateResult[GameAction]:
        card_ids_result = _extract_card_ids(cards_raw)
        if isinstance(card_ids_result, Rejected):
            return card_ids_result
        resolved_result = self.resolve_cards(player_index, card_ids_result.value)
        if isinstance(resolved_result, Rejected):
            return resolved_result

        cards = resolved_result.value
        if action_type == "bid":
            return Ok(value=BidAction(cards=cards, count=len(cards)))
        if action_type == "stir":
            return Ok(value=StirAction(cards=cards))
        if action_type == "discard":
            return Ok(value=DiscardAction(cards=cards))
        if action_type == "play":
            return Ok(value=PlayAction(cards=cards))
        return Rejected(reason=f"unknown card action type: {action_type}")

    def resolve_cards(self, player_index: int, card_ids: list[str]) -> Ok[list[Card]] | Rejected:
        """Resolve card ID strings to Card objects from the player's hand.

        Returns Rejected if the game has not started yet, or if any
        card_id is not found in the player's hand.
        Raises IndexError if player_index is out of range (programming error).
        """
        rs = self._round_state
        if rs is None:
            return Rejected(reason="游戏尚未开始")
        hand = rs.players_hand[player_index]
        if (
            rs.phase == "STIRRING"
            and rs.stirring_state is not None
            and rs.stirring_state.phase == "EXCHANGING"
            and rs.stirring_state.exchanging_player == player_index
            and rs.stirring_state.exchange_state is not None
        ):
            hand = rs.stirring_state.exchange_state.hand_after_pickup
        card_map = {c.id: c for c in hand}

        result: list[Card] = []
        for card_id in card_ids:
            if card_id not in card_map:
                return Rejected(reason=f"Card {card_id} not in hand of player {player_index}")
            result.append(card_map[card_id])
        return Ok(value=result)

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

    def _state_message_for(self, player_index: int, error: str | None = None) -> StateMessage:
        snapshot = self.snapshot(player_index)
        return StateMessage(
            seq=self._seq,
            awaiting=snapshot.awaiting_action,
            state=snapshot,
            error=error,
        )

    async def _send_state_to_player(self, player_index: int, error: str | None = None) -> None:
        await self._players[player_index].on_state(self, self._state_message_for(player_index, error=error))

    async def _push_state_to_all(self) -> None:
        """Push state to all players."""
        self._seq += 1
        for i in range(len(self._players)):
            await self._players[i].on_state(self, self._state_message_for(i, error=None))


def _is_str_dict(val: object) -> TypeGuard[dict[str, object]]:
    return isinstance(val, dict)


def _is_obj_list(val: object) -> TypeGuard[list[object]]:
    return isinstance(val, list)


def _extract_card_ids(cards_val: object) -> StateResult[list[str]]:
    """Extract card IDs after seq validation has accepted the message."""
    if not _is_obj_list(cards_val):
        return Ok(value=[])
    ids: list[str] = []
    for item in cards_val:
        if isinstance(item, str):
            ids.append(item)
        elif _is_str_dict(item):
            id_val = item.get("id")
            if isinstance(id_val, str):
                ids.append(id_val)
            else:
                return Rejected(reason=f"Invalid card format: missing 'id' in {item}")
        else:
            return Rejected(reason=f"Invalid card format: {item}")
    return Ok(value=ids)
