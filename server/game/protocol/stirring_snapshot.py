"""Stirring phase snapshot model."""

from __future__ import annotations

from server.game.protocol.snapshot_common import (
    SnapshotModel,
    StirringPhase,
)
from server.game.rules.cards import Suit


class StirringStateSnapshot(SnapshotModel):
    """Public stirring phase state."""

    phase: StirringPhase
    trump_suit: Suit | None
    current_player: int
    declarer_player: int
    exchanging_player: int | None
    exchange_count: int | None
