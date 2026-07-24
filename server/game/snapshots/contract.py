"""Pending and established round-contract snapshots."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from server.game.rules.cards import Rank, Suit

from ._base import SnapshotModel

__all__ = [
    "NoTrump",
    "PendingTrump",
    "SuitedTrump",
    "TrumpSnapshot",
    "trump_rank",
    "trump_suit",
]


class PendingTrump(SnapshotModel):
    """Cards are still being dealt; trump is not established."""

    kind: Literal["pending"] = "pending"
    rank: Rank


class NoTrump(SnapshotModel):
    """The contract has explicitly established no-trump."""

    kind: Literal["no_trump"] = "no_trump"
    rank: Rank


class SuitedTrump(SnapshotModel):
    """The contract has established a trump suit."""

    kind: Literal["suited"] = "suited"
    rank: Rank
    suit: Suit


type TrumpSnapshot = Annotated[
    PendingTrump | NoTrump | SuitedTrump,
    Field(discriminator="kind"),
]


def trump_rank(trump: TrumpSnapshot) -> Rank:
    """Return the rank shared by every contract state."""
    return trump.rank


def trump_suit(trump: TrumpSnapshot) -> Suit | None:
    """Return the established suit, including explicit no-trump."""
    if isinstance(trump, SuitedTrump):
        return trump.suit
    return None
