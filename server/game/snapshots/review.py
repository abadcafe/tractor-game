"""Round scoring snapshots."""

from __future__ import annotations

from server.game.rules.cards import Card
from server.game.seating import Partnership

from ._base import SnapshotModel

__all__ = ["ScoringSnapshot"]


class ScoringSnapshot(SnapshotModel):
    """All public scoring facts for a completed round."""

    winning_partnership: Partnership
    defender_points: int
    total_defender_points: int
    bottom_base_points: int
    bottom_multiplier: int | None
    bottom_points: int
    bottom_cards: tuple[Card, ...]
