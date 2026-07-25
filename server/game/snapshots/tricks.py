"""Current and completed trick snapshots."""

from __future__ import annotations

from server.game.rules.cards import Card
from server.game.seating import Seat

from ._base import SnapshotModel

__all__ = [
    "CompletedTrickSnapshot",
    "FailedThrowSnapshot",
    "TrickSlotSnapshot",
    "TrickSnapshot",
]


class TrickSlotSnapshot(SnapshotModel):
    """One player's cards in a trick."""

    actor: Seat
    cards: tuple[Card, ...]


class FailedThrowSnapshot(SnapshotModel):
    """A failed throw and its forced lead."""

    actor: Seat
    attempted_cards: tuple[Card, ...]
    forced_cards: tuple[Card, ...]


class TrickSnapshot(SnapshotModel):
    """An unresolved trick."""

    lead_actor: Seat
    slots: tuple[TrickSlotSnapshot, ...]
    current_actor: Seat
    failed_throw: FailedThrowSnapshot | None = None


class CompletedTrickSnapshot(SnapshotModel):
    """The last resolved trick."""

    lead_actor: Seat
    slots: tuple[TrickSlotSnapshot, ...]
    winner: Seat
    points: int
    failed_throw: FailedThrowSnapshot | None = None
