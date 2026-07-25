"""Immutable values shared by mutually exclusive game phases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from server.game.rules.cards import Card, Rank, Suit
from server.game.seating import (
    Partnership,
    PartnershipMap,
    Seat,
    SeatMap,
)

type Hands = SeatMap[tuple[Card, ...]]
type TeamLevels = PartnershipMap[Rank]


class BottomExchangeCause(str, Enum):
    """Why a player currently owns the bottom exchange."""

    INITIAL = "INITIAL"
    STIR = "STIR"


@dataclass(frozen=True, slots=True)
class Declaration:
    """A successful declaration at a particular deal ordinal."""

    actor: Seat
    cards: tuple[Card, ...]
    deal_ordinal: int


@dataclass(frozen=True, slots=True)
class Contract:
    """The established round contract."""

    declarer: Seat
    trump_rank: Rank
    trump_suit: Suit | None


@dataclass(frozen=True, slots=True)
class StirEvent:
    """A public stir decision."""

    actor: Seat
    cards: tuple[Card, ...]
    priority: int | None


@dataclass(frozen=True, slots=True)
class BottomExchangeEvent:
    """Private cards observed during one bottom exchange."""

    actor: Seat
    cause: BottomExchangeCause
    stir_event_index: int | None
    picked_up: tuple[Card, ...]
    discarded: tuple[Card, ...]
    resulting_bottom: tuple[Card, ...]


@dataclass(frozen=True, slots=True)
class InternalFailedThrow:
    """A failed throw and the forced cards that actually led."""

    actor: Seat
    attempted_cards: tuple[Card, ...]
    forced_cards: tuple[Card, ...]


@dataclass(frozen=True, slots=True)
class InternalTrickSlot:
    """One contribution to a trick."""

    actor: Seat
    cards: tuple[Card, ...]


@dataclass(frozen=True, slots=True)
class InternalCompletedTrick:
    """A resolved trick."""

    lead_actor: Seat
    slots: tuple[InternalTrickSlot, ...]
    winner: Seat
    points: int
    failed_throw: InternalFailedThrow | None


@dataclass(frozen=True, slots=True)
class OpenTrick:
    """The current, unresolved trick."""

    lead_actor: Seat
    current_actor: Seat
    slots: tuple[InternalTrickSlot, ...]
    failed_throw: InternalFailedThrow | None


@dataclass(frozen=True, slots=True)
class CompletedRound:
    """All durable facts produced by a completed round."""

    round_number: int
    contract: Contract
    winning_partnership: Partnership
    next_declarer: Seat
    defender_points: int
    total_defender_points: int
    bottom_card_bonus: int
    bottom_cards: tuple[Card, ...]
    levels_before: TeamLevels
    levels_after: TeamLevels
    last_trick: InternalCompletedTrick
    declaration: Declaration | None
    bid_reveals: tuple[Declaration, ...]
    stir_events: tuple[StirEvent, ...]
    exchange_events: tuple[BottomExchangeEvent, ...]
    defender_point_cards: tuple[Card, ...]
