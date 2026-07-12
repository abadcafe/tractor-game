"""Scoring snapshot model."""

from __future__ import annotations

from server.game.protocol.snapshot_common import SnapshotModel
from server.game.rules.cards import Card


class ScoringSnapshot(SnapshotModel):
    """Round scoring information."""

    round_winning_team: int
    defender_points: int
    total_defender_points: int
    bottom_card_bonus: int
    bottom_cards: list[Card]
