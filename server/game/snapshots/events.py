"""Bidding, stirring, and private exchange snapshot events."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from server.game.config import Seat
from server.game.rules.cards import Card, Suit

from ._base import SnapshotModel

__all__ = [
    "BidEventSnapshot",
    "BottomExchangeSnapshot",
    "StirDeclarationEventSnapshot",
    "StirringStateSnapshot",
]


class BidEventSnapshot(SnapshotModel):
    """One successful deal-time reveal."""

    player: Seat
    cards: tuple[Card, ...]
    kind: Literal["trump_rank", "joker"]
    suit: Suit | None
    joker_type: Literal["big", "small"] | None
    count: int
    deal_ordinal: int

    @model_validator(mode="after")
    def _validate_kind(self) -> Self:
        assert self.cards
        assert self.count == len(self.cards)
        assert (self.kind == "joker") == (self.suit is None)
        return self


class BottomExchangeSnapshot(SnapshotModel):
    """One private bottom exchange visible only to its actor."""

    player: Seat
    trigger: Literal["initial", "stir"]
    stir_event_index: int | None
    picked_up_bottom_cards: tuple[Card, ...]
    discarded_bottom_cards: tuple[Card, ...]
    resulting_bottom_cards: tuple[Card, ...]


class StirDeclarationEventSnapshot(SnapshotModel):
    """One public stir or pass decision."""

    player: Seat
    kind: Literal["stir", "pass"]
    cards: tuple[Card, ...]
    new_suit: Suit | None
    priority: int | None
    own_bottom_exchange: BottomExchangeSnapshot | None


class StirringStateSnapshot(SnapshotModel):
    """Current stirring decision or bottom exchange."""

    phase: Literal["WAITING", "EXCHANGING"]
    trump_suit: Suit | None
    current_player: Seat
    declarer_player: Seat
    exchanging_player: Seat | None
    exchange_count: int | None
