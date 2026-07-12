"""Bid and stir declaration snapshot model."""

from __future__ import annotations

from server.game.protocol.snapshot_common import (
    BidEventKind,
    JokerType,
    SnapshotModel,
)
from server.game.rules.cards import Card, Suit


class BidEventSnapshot(SnapshotModel):
    """Public bid/stir trump declaration event."""

    player: int
    cards: list[Card]
    kind: BidEventKind
    suit: Suit | None
    joker_type: JokerType | None
    count: int
