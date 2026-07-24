"""Round scoring snapshots."""

from __future__ import annotations

from server.game.rules.cards import Card

from ._base import SnapshotModel

__all__ = ["ScoringSnapshot"]


class ScoringSnapshot(SnapshotModel):
    """All public scoring facts for a completed round."""

    round_winning_team: int
    defender_points: int
    total_defender_points: int
    bottom_card_bonus: int
    bottom_cards: tuple[Card, ...]
