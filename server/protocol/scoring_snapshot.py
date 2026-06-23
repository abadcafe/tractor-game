"""Scoring snapshot model."""

from __future__ import annotations

from server.protocol.snapshot_common import SnapshotModel
from server.rules.cards import Card


class ScoringSnapshot(SnapshotModel):
    """Round scoring information."""

    declarer_team: int | None
    defender_points: int
    total_defender_points: int
    bottom_card_bonus: int
    bottom_cards: list[Card]
