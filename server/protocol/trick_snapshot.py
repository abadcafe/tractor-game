"""Trick-related player-facing snapshot models."""

from __future__ import annotations

from server.protocol.snapshot_common import SnapshotModel
from server.rules.cards import Card


class TrickSlotSnapshot(SnapshotModel):
    """One player's contribution in a trick."""

    player: int
    cards: list[Card]


class TrickSnapshot(SnapshotModel):
    """Current in-progress trick."""

    lead_player: int
    slots: list[TrickSlotSnapshot]
    current_player: int
    failed_throw: FailedThrowSnapshot | None = None


class CompletedTrickSnapshot(SnapshotModel):
    """Completed trick visible to players."""

    lead_player: int
    slots: list[TrickSlotSnapshot]
    winner: int
    points: int
    failed_throw: FailedThrowSnapshot | None = None


class FailedThrowSnapshot(SnapshotModel):
    """Public event emitted when a throw attempt is forced smaller."""

    player: int
    attempted_cards: list[Card]
    forced_cards: list[Card]
