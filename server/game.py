"""Game aggregate root for Tractor game.

Wraps sm state machines, manages 4 Player instances, drives the dealing loop,
and provides act(), run(), snapshot(), is_over(), get_phase(), set_on_game_over(),
get_player(), cancel(), and resolve_cards() interfaces.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable

from server.sm import game_sm, round_sm, play_rules
from server.sm.card_model import Card, Suit, Rank
from server.sm.types import BidEvent
from server.player import Player, BidAction, StirAction, SkipStirAction, DiscardAction, PlayAction, NextRoundAction

logger = logging.getLogger(__name__)


def _card_to_dict(card: Card) -> dict:
    """Convert a Card Pydantic model to a JSON-serializable dict.

    Returns {"id": card.id, "suit": card.suit.value, "rank": card.rank.value}.
    Omits internal sm fields (is_joker, is_big_joker, points, deck) per spec.
    """
    return {
        "id": card.id,
        "suit": card.suit.value,
        "rank": card.rank.value,
    }


@dataclass
class StateSnapshot:
    """A player-facing snapshot of the current game state.

    Contains all fields from spec section 3.3. The to_dict() method
    serializes to JSON format matching spec section 5.5.
    """

    phase: str
    player_hand: list
    player_hand_counts: list[int]
    bottom_cards: list
    trump_suit: Suit | None
    trump_rank: Rank
    declarer_team: int | None
    declarer_player: int | None
    current_player: int
    defender_points: int
    trick: dict | None
    trick_history: list
    legal_actions: list
    awaiting_action: str | None
    scoring: dict | None
    winning_team: int | None
    team0_level: Rank
    team1_level: Rank
    bid_events: list
    bid_winner: dict | None
    stirring_state: dict | None
    exchange_state: dict | None

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict matching spec section 5.5.

        Cards are serialized as {"id", "suit", "rank"}.
        Enums are serialized as their string values.
        legal_actions entries are serialized as lists of card-dict lists.
        """
        return {
            "phase": self.phase,
            "player_hand": [_card_to_dict(c) for c in self.player_hand],
            "player_hand_counts": self.player_hand_counts,
            "bottom_cards": [_card_to_dict(c) for c in self.bottom_cards],
            "trump_suit": self.trump_suit.value if self.trump_suit is not None else None,
            "trump_rank": self.trump_rank.value,
            "declarer_team": self.declarer_team,
            "declarer_player": self.declarer_player,
            "current_player": self.current_player,
            "defender_points": self.defender_points,
            "trick": self._serialize_trick(self.trick),
            "trick_history": [_serialize_completed_trick(t) for t in self.trick_history],
            "legal_actions": [
                [_card_to_dict(c) for c in entry]
                for entry in self.legal_actions
            ],
            "awaiting_action": self.awaiting_action,
            "scoring": self.scoring,
            "winning_team": self.winning_team,
            "team0_level": self.team0_level.value,
            "team1_level": self.team1_level.value,
            "bid_events": [_serialize_bid_event(e) for e in self.bid_events],
            "bid_winner": _serialize_bid_event(self.bid_winner) if self.bid_winner is not None else None,
            "stirring_state": self.stirring_state,
            "exchange_state": self.exchange_state,
        }

    def _serialize_trick(self, trick: dict | None) -> dict | None:
        """Serialize the trick dict, converting cards within to dict format."""
        if trick is None:
            return None
        return _serialize_dict_trick(trick)


def _serialize_bid_event(event: BidEvent) -> dict:
    """Serialize a BidEvent to a JSON-serializable dict."""
    return {
        "player": event.player,
        "cards": [_card_to_dict(c) for c in event.cards],
        "kind": event.kind,
        "suit": event.suit.value if event.suit is not None else None,
        "joker_type": event.joker_type,
        "count": event.count,
    }


def _serialize_dict_trick(trick: dict) -> dict:
    """Serialize a dict-formatted trick/CompletedTrick, converting cards to dict format.

    Shared helper for _serialize_trick and _serialize_completed_trick
    to avoid duplicating the dict-format handling logic.
    """
    result = dict(trick)
    # Remove stale lead_type key if present (no longer in CompletedTrick)
    result.pop("lead_type", None)
    if "slots" in result:
        result["slots"] = [
            {
                "player": slot.get("player") if isinstance(slot, dict) else getattr(slot, "player", None),
                "cards": [_card_to_dict(c) for c in (slot.get("cards", []) if isinstance(slot, dict) else getattr(slot, "cards", []))],
            }
            for slot in result["slots"]
        ]
    return result


def _serialize_completed_trick(trick) -> dict:
    """Serialize a CompletedTrick to a JSON-serializable dict."""
    if isinstance(trick, dict):
        return _serialize_dict_trick(trick)
    return {
        "lead_player": trick.lead_player,
        "slots": [
            {
                "player": slot.player,
                "cards": [_card_to_dict(c) for c in slot.cards],
            }
            for slot in trick.slots
        ],
        "winner": trick.winner,
        "points": trick.points,
    }


class Game:
    """Aggregate root that orchestrates game lifecycle using sm state machines.

    Manages 4 Player instances, drives the dealing loop, and provides
    the public API for the server layer.
    """

    def __init__(self, players: list[Player]) -> None:
        self._game_state = game_sm.create_game()
        self._round_state: round_sm.RoundState | None = None
        self._players = players
        self._dealing_task: asyncio.Task | None = None
        self._on_game_over: Callable[['Game'], None] | None = None
        self._cancelled: bool = False

    async def run(self) -> None:
        """Start the game: transition to IN_ROUND, create round, start dealing loop.

        Raises RuntimeError if called more than once.
        """
        if self._round_state is not None:
            raise RuntimeError("Game already started; run() can only be called once")
        self._game_state = game_sm.start_game(self._game_state)
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

        After applying the action, pushes state to the appropriate player(s).
        If the action causes GAME_OVER, pushes final state to all players
        and invokes the on_game_over callback.

        Raises ValueError if player_index is out of range.
        """
        if player_index < 0 or player_index >= len(self._players):
            raise ValueError(f"Player index {player_index} out of range (0-{len(self._players) - 1})")
        phase = self.get_phase()
        logger.debug("Game.act: player=%d action=%s phase=%s", player_index, type(action).__name__, phase)

        if phase == "DEAL_BID" and isinstance(action, BidAction):
            bid_event = self._convert_bid_action(player_index, action)
            self._round_state = round_sm.reveal(self._round_state, bid_event)
            # No state push here: the dealing loop pushes to all players every
            # 0.75s, so the next tick will carry the updated bid_winner.
            # Pushing here would trigger AutoPlayer.on_state → create_task(bid)
            # → act → _push_state_to_all → on_state → … exponential cascade.

        elif phase == "STIRRING" and isinstance(action, SkipStirAction):
            self._round_state = round_sm.pass_stir(self._round_state)
            await self._push_state_to_all()

        elif phase == "STIRRING" and isinstance(action, StirAction):
            self._round_state = round_sm.stir(self._round_state, action.cards)
            await self._push_state_to_all()

        elif phase == "EXCHANGE" and isinstance(action, DiscardAction):
            self._round_state = round_sm.discard(self._round_state, action.cards)
            await self._push_state_to_all()

        elif phase == "PLAYING" and isinstance(action, PlayAction):
            self._round_state = round_sm.play(self._round_state, action.cards)
            await self._push_state_to_all()

        elif phase == "COMPLETE" and isinstance(action, NextRoundAction):
            round_result = round_sm.get_round_result(self._round_state)
            if round_result is None:
                raise ValueError(
                    "Round result is None in COMPLETE phase; this indicates an sm layer bug"
                )
            self._game_state = game_sm.process_round_result(self._game_state, round_result)

            if self._game_state.phase == "GAME_OVER":
                await self._push_state_to_all()
                if self._on_game_over is not None:
                    self._on_game_over(self)
            else:
                self._cancelled = False
                self._round_state = round_sm.create_round(round_sm.RoundInput(
                    declarer_team=self._game_state.declarer_team,
                    trump_rank=self._game_state.team0_level,
                    last_declarer_player=self._game_state.last_declarer_player,
                    team0_level=self._game_state.team0_level,
                    team1_level=self._game_state.team1_level,
                ))
                # Start a new dealing loop for the next round
                self._dealing_task = asyncio.create_task(self._dealing_loop())
                await self._push_state_to_all()
        else:
            raise ValueError(f"Invalid action {type(action).__name__} in phase {phase}")

    def snapshot(self, for_player: int) -> StateSnapshot:
        """Build a StateSnapshot for the given player.

        Raises RuntimeError if called before run().
        Raises ValueError if for_player is out of range.
        """
        if for_player < 0 or for_player >= len(self._players):
            raise ValueError(f"Player index {for_player} out of range (0-{len(self._players) - 1})")
        if self._round_state is None:
            raise RuntimeError("Game not started")

        rs = self._round_state
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
        legal_actions: list = []
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
                other_hands: list = []
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
        trick = None
        if rs.phase == "PLAYING" and rs.trick_state is not None:
            ts = rs.trick_state
            trick = {
                "lead_player": ts.lead_player,
                "slots": [
                    {"player": slot.player, "cards": slot.cards}
                    for slot in ts.slots
                ],
                "current_player": ts.cur,
            }

        # bid_events and bid_winner
        bid_events = []
        bid_winner = None
        if rs.deal_bid_state is not None:
            bid_events = list(rs.deal_bid_state.bid_events)
            bid_winner = rs.deal_bid_state.bid_winner

        # stirring_state
        stirring_state_dict = None
        if rs.stirring_state is not None and rs.phase == "STIRRING":
            stirring_state_dict = {
                "phase": rs.stirring_state.phase,
                "trump_suit": rs.stirring_state.trump_suit.value if rs.stirring_state.trump_suit is not None else None,
                "current_player": rs.stirring_state.current_player,
            }

        # exchange_state
        exchange_state_dict = None
        if rs.exchange_state is not None and rs.phase == "EXCHANGE":
            exchange_state_dict = {
                "phase": rs.exchange_state.phase,
                "declarer_player": rs.exchange_state.declarer_player,
                "count": rs.exchange_state.count,
            }

        # scoring
        scoring_dict = None
        if rs.result is not None:
            scoring_dict = {
                "declarer_team": rs.declarer_team,
                "defender_points": rs.defender_points,
                "bottom_cards": [_card_to_dict(c) for c in rs.bottom_cards],
            }

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
            scoring=scoring_dict,
            winning_team=gs.winning_team,
            team0_level=rs.team0_level,
            team1_level=rs.team1_level,
            bid_events=bid_events,
            bid_winner=bid_winner,
            stirring_state=stirring_state_dict,
            exchange_state=exchange_state_dict,
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

        Raises ValueError if the index is out of range.
        """
        if index < 0 or index >= len(self._players):
            raise ValueError(f"Player index {index} out of range (0-{len(self._players) - 1})")
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

    def resolve_cards(self, player_index: int, card_ids: list[str]) -> list[Card]:
        """Resolve card ID strings to Card objects from the player's hand.

        Raises ValueError if any card_id is not found in the player's hand.
        Raises ValueError if player_index is out of range.
        """
        if player_index < 0 or player_index >= len(self._players):
            raise ValueError(f"Player index {player_index} out of range (0-{len(self._players) - 1})")
        if self._round_state is None:
            raise RuntimeError("Game not started")

        hand = self._round_state.players_hand[player_index]
        card_map = {c.id: c for c in hand}

        result = []
        for card_id in card_ids:
            if card_id not in card_map:
                raise ValueError(f"Card {card_id} not in hand of player {player_index}")
            result.append(card_map[card_id])
        return result

    async def _dealing_loop(self) -> None:
        """Background coroutine that deals cards one at a time.

        Checks _cancelled at the start of each iteration.
        Sleeps 0.75s between deals. Pushes state to all players after each deal.
        When dealing completes, transitions to STIRRING automatically via sm.
        """
        try:
            while not self._cancelled:
                if self._round_state is None:
                    break
                if self._round_state.phase != "DEAL_BID":
                    break
                if self._round_state.deal_bid_state is None:
                    break

                self._round_state = round_sm.deal_next_card(self._round_state)
                await self._push_state_to_all()

                if self._round_state.phase != "DEAL_BID":
                    break

                await asyncio.sleep(0.5)
        except Exception:
            logger.exception("Dealing loop failed with unexpected exception")

    def _convert_bid_action(self, player_index: int, action: BidAction) -> BidEvent:
        """Convert a player BidAction to an sm BidEvent."""
        cards = action.cards
        if not cards:
            raise ValueError("BidAction requires at least one card")
        # Determine kind, suit, joker_type from the cards
        if cards and cards[0].is_joker:
            kind = "joker"
            joker_type = "big" if cards[0].is_big_joker else "small"
            suit = None
        else:
            kind = "trump_rank"
            suit = cards[0].suit
            joker_type = None

        return BidEvent(
            player=player_index,
            cards=cards,
            kind=kind,
            suit=suit,
            joker_type=joker_type,
            count=action.count,
        )

    async def _push_state_to_all(self) -> None:
        """Push state to all players."""
        for i in range(len(self._players)):
            await self._players[i].on_state(self)
