"""Current and completed trick snapshots."""

from __future__ import annotations

from server.game.config import Seat
from server.game.rules.cards import Card

from ._base import SnapshotModel

__all__ = [
    "CompletedTrickSnapshot",
    "FailedThrowSnapshot",
    "TrickSlotSnapshot",
    "TrickSnapshot",
]


class TrickSlotSnapshot(SnapshotModel):
    """One player's cards in a trick."""

    player: Seat
    cards: tuple[Card, ...]


class FailedThrowSnapshot(SnapshotModel):
    """A failed throw and its forced lead."""

    player: Seat
    attempted_cards: tuple[Card, ...]
    forced_cards: tuple[Card, ...]


class TrickSnapshot(SnapshotModel):
    """An unresolved trick."""

    lead_player: Seat
    slots: tuple[TrickSlotSnapshot, ...]
    current_player: Seat
    failed_throw: FailedThrowSnapshot | None = None


class CompletedTrickSnapshot(SnapshotModel):
    """The last resolved trick."""

    lead_player: Seat
    slots: tuple[TrickSlotSnapshot, ...]
    winner: Seat
    points: int
    failed_throw: FailedThrowSnapshot | None = None
