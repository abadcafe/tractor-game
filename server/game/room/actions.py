"""Player action types for the Tractor game.

Pure data containers used by both Player (to submit actions)
and Game (to dispatch actions). Depends only on Card.
"""

from dataclasses import dataclass
from typing import Literal

from server.game.rules.cards import Card

type GameActionKind = Literal[
    "bid",
    "skip_bid",
    "stir",
    "skip_stir",
    "discard",
    "play",
    "next_round",
]

type CardActionKind = Literal["bid", "stir", "discard", "play"]


@dataclass
class BidAction:
    """Cards the player reveals during bidding."""

    cards: list[Card]
    count: int


@dataclass
class StirAction:
    """Pair of cards to stir with (change trump suit)."""

    cards: list[Card]


@dataclass
class SkipStirAction:
    """Pass during stirring."""

    pass


@dataclass
class SkipBidAction:
    """Action to skip/pass during DEAL_BID phase."""

    pass


@dataclass
class DiscardAction:
    """Cards to discard for the bottom pile."""

    cards: list[Card]


@dataclass
class PlayAction:
    """Cards to play in the current trick."""

    cards: list[Card]


@dataclass
class NextRoundAction:
    """Signal to proceed to the next round."""

    pass
