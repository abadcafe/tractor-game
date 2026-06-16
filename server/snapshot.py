"""Player-facing snapshot types for the Tractor game.

Contains StateSnapshot and its sub-snapshots, plus serialization
helpers and TypedDict types for the JSON-serialized output.
Depends only on card_model and sm types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from server.sm.card_model import Card, Rank, Suit
from server.sm.types import BidEvent, CompletedTrick


def _card_to_dict(card: Card) -> CardDict:
    """Convert a Card Pydantic model to a JSON-serializable dict.

    Returns {"id": card.id, "suit": card.suit.value, "rank": card.rank.value}.
    Omits internal sm fields (is_joker, is_big_joker, points, deck) per spec.
    """
    return {
        "id": card.id,
        "suit": card.suit.value,
        "rank": card.rank.value,
    }


# ---- Snapshot dataclasses ----


@dataclass
class TrickSlotSnapshot:
    """One player's contribution in the current trick snapshot."""

    player: int
    cards: list[Card]


@dataclass
class TrickSnapshot:
    """Snapshot of the current in-progress trick."""

    lead_player: int
    slots: list[TrickSlotSnapshot]
    current_player: int


@dataclass
class ScoringSnapshot:
    """Snapshot of round scoring information."""

    declarer_team: int | None
    defender_points: int
    total_defender_points: int
    bottom_card_bonus: int
    bottom_cards: list[Card]


@dataclass
class StirringStateSnapshot:
    """Snapshot of the stirring (炒地皮) phase state."""

    phase: str
    trump_suit: Suit | None
    current_player: int
    declarer_player: int
    legal_actions: list[list[Card]]


@dataclass
class ExchangeStateSnapshot:
    """Snapshot of the exchange (换底牌) phase state."""

    phase: str
    declarer_player: int
    count: int


@dataclass
class StateSnapshot:
    """A player-facing snapshot of the current game state.

    Contains all fields from spec section 3.3. The to_dict() method
    serializes to JSON format matching spec section 5.5.
    """

    phase: str
    player_hand: list[Card]
    player_hand_counts: list[int]
    bottom_cards: list[Card]
    trump_suit: Suit | None
    trump_rank: Rank
    declarer_team: int | None
    declarer_player: int | None
    defender_points: int
    trick: TrickSnapshot | None
    trick_history: list[CompletedTrick]
    legal_actions: list[list[Card]]
    awaiting_action: str | None
    bid_legal_actions: list[list[Card]] | None
    scoring: ScoringSnapshot | None
    winning_team: int | None
    team0_level: Rank
    team1_level: Rank
    bid_events: list[BidEvent]
    bid_winner: BidEvent | None
    stirring_state: StirringStateSnapshot | None
    exchange_state: ExchangeStateSnapshot | None
    next_round_confirmed: list[int]

    def to_dict(self) -> SnapshotDict:
        """Convert to a JSON-serializable dict matching spec section 5.5.

        Cards are serialized as {"id", "suit", "rank"}.
        Enums are serialized as their string values.
        legal_actions entries are serialized as lists of card-dict lists.
        """
        trick_dict: TrickDict | None = None
        if self.trick is not None:
            trick_dict = {
                "lead_player": self.trick.lead_player,
                "slots": [
                    {
                        "player": slot.player,
                        "cards": [_card_to_dict(c) for c in slot.cards],
                    }
                    for slot in self.trick.slots
                ],
                "current_player": self.trick.current_player,
            }

        scoring_dict: ScoringDict | None = None
        if self.scoring is not None:
            scoring_dict = {
                "declarer_team": self.scoring.declarer_team,
                "defender_points": self.scoring.defender_points,
                "total_defender_points": self.scoring.total_defender_points,
                "bottom_card_bonus": self.scoring.bottom_card_bonus,
                "bottom_cards": [_card_to_dict(c) for c in self.scoring.bottom_cards],
            }

        stirring_dict: StirringStateDict | None = None
        if self.stirring_state is not None:
            stirring_dict = {
                "phase": self.stirring_state.phase,
                "trump_suit": self.stirring_state.trump_suit.value if self.stirring_state.trump_suit is not None else None,
                "current_player": self.stirring_state.current_player,
                "declarer_player": self.stirring_state.declarer_player,
                "legal_actions": [
                    [_card_to_dict(c) for c in entry]
                    for entry in self.stirring_state.legal_actions
                ],
            }

        exchange_dict: ExchangeStateDict | None = None
        if self.exchange_state is not None:
            exchange_dict = {
                "phase": self.exchange_state.phase,
                "declarer_player": self.exchange_state.declarer_player,
                "count": self.exchange_state.count,
            }

        return {
            "phase": self.phase,
            "player_hand": [_card_to_dict(c) for c in self.player_hand],
            "player_hand_counts": self.player_hand_counts,
            "bottom_cards": [_card_to_dict(c) for c in self.bottom_cards],
            "trump_suit": self.trump_suit.value if self.trump_suit is not None else None,
            "trump_rank": self.trump_rank.value,
            "declarer_team": self.declarer_team,
            "declarer_player": self.declarer_player,
            "defender_points": self.defender_points,
            "trick": trick_dict,
            "trick_history": [
                {
                    "lead_player": t.lead_player,
                    "slots": [
                        {
                            "player": slot.player,
                            "cards": [_card_to_dict(c) for c in slot.cards],
                        }
                        for slot in t.slots
                    ],
                    "winner": t.winner,
                    "points": t.points,
                }
                for t in self.trick_history
            ],
            "legal_actions": [
                [_card_to_dict(c) for c in entry]
                for entry in self.legal_actions
            ],
            "awaiting_action": self.awaiting_action,
            "bid_legal_actions": (
                [[_card_to_dict(c) for c in entry] for entry in self.bid_legal_actions]
                if self.bid_legal_actions is not None
                else None
            ),
            "scoring": scoring_dict,
            "winning_team": self.winning_team,
            "team0_level": self.team0_level.value,
            "team1_level": self.team1_level.value,
            "bid_events": [_serialize_bid_event(e) for e in self.bid_events],
            "bid_winner": _serialize_bid_event(self.bid_winner) if self.bid_winner is not None else None,
            "stirring_state": stirring_dict,
            "exchange_state": exchange_dict,
            "next_round_confirmed": self.next_round_confirmed,
        }


def _serialize_bid_event(event: BidEvent) -> BidEventDict:
    """Serialize a BidEvent to a JSON-serializable dict."""
    return {
        "player": event.player,
        "cards": [_card_to_dict(c) for c in event.cards],
        "kind": event.kind,
        "suit": event.suit.value if event.suit is not None else None,
        "joker_type": event.joker_type,
        "count": event.count,
    }


# ---- TypedDict types for JSON-serialized snapshots ----


class CardDict(TypedDict):
    """Serialized Card: {"id", "suit", "rank"}."""

    id: str
    suit: str
    rank: str


class TrickSlotDict(TypedDict):
    """One player's contribution in a serialized trick."""

    player: int
    cards: list[CardDict]


class TrickDict(TypedDict):
    """Serialized in-progress trick."""

    lead_player: int
    slots: list[TrickSlotDict]
    current_player: int


class CompletedTrickSlotDict(TypedDict):
    """One player's contribution in a completed trick."""

    player: int
    cards: list[CardDict]


class CompletedTrickDict(TypedDict):
    """Serialized completed trick."""

    lead_player: int
    slots: list[CompletedTrickSlotDict]
    winner: int
    points: int


class ScoringDict(TypedDict):
    """Serialized scoring info."""

    declarer_team: int | None
    defender_points: int
    total_defender_points: int
    bottom_card_bonus: int
    bottom_cards: list[CardDict]


class StirringStateDict(TypedDict):
    """Serialized stirring phase state."""

    phase: str
    trump_suit: str | None
    current_player: int
    declarer_player: int
    legal_actions: list[list[CardDict]]


class ExchangeStateDict(TypedDict):
    """Serialized exchange phase state."""

    phase: str
    declarer_player: int
    count: int


class BidEventDict(TypedDict):
    """Serialized bid event."""

    player: int
    cards: list[CardDict]
    kind: str
    suit: str | None
    joker_type: str | None
    count: int


class SnapshotDict(TypedDict):
    """JSON-serialized snapshot matching spec section 5.5."""

    phase: str
    player_hand: list[CardDict]
    player_hand_counts: list[int]
    bottom_cards: list[CardDict]
    trump_suit: str | None
    trump_rank: str
    declarer_team: int | None
    declarer_player: int | None
    defender_points: int
    trick: TrickDict | None
    trick_history: list[CompletedTrickDict]
    legal_actions: list[list[CardDict]]
    awaiting_action: str | None
    bid_legal_actions: list[list[CardDict]] | None
    scoring: ScoringDict | None
    winning_team: int | None
    team0_level: str
    team1_level: str
    bid_events: list[BidEventDict]
    bid_winner: BidEventDict | None
    stirring_state: StirringStateDict | None
    exchange_state: ExchangeStateDict | None
    next_round_confirmed: list[int]
