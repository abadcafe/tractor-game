"""Player-facing stir and bottom-exchange history snapshots."""

from __future__ import annotations

from typing import Literal

from server.protocol.snapshot_common import SnapshotModel
from server.rules.cards import Card, Suit

type StirEventKind = Literal["stir", "pass"]
type BottomExchangeTrigger = Literal["initial", "stir"]


class StirDeclarationEventSnapshot(SnapshotModel):
    """Public stir declaration or pass event visible to every player."""

    player: int
    kind: StirEventKind
    cards: list[Card]
    new_suit: Suit | None
    priority: int | None


class BottomExchangeEventSnapshot(SnapshotModel):
    """Viewer-private bottom exchange memory."""

    player: int
    trigger: BottomExchangeTrigger
    stir_event_index: int | None
    picked_up_bottom_cards: list[Card]
    discarded_bottom_cards: list[Card]
    resulting_bottom_cards: list[Card]
