"""Player-facing stir and bottom-exchange history snapshots."""

from __future__ import annotations

from typing import Literal

from server.game.protocol.snapshot_common import SnapshotModel
from server.game.rules.cards import Card, Suit

type StirEventKind = Literal["stir", "pass"]


class BottomExchangeSnapshot(SnapshotModel):
    """Viewer-private bottom exchange memory."""

    picked_up_bottom_cards: list[Card]
    discarded_bottom_cards: list[Card]


class StirDeclarationEventSnapshot(SnapshotModel):
    """Public stir event with viewer-private exchange if owned."""

    player: int
    kind: StirEventKind
    cards: list[Card]
    new_suit: Suit | None
    priority: int | None
    own_bottom_exchange: BottomExchangeSnapshot | None
