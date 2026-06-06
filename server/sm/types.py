"""Shared type definitions used across Shengji/Tractor state machines.

Defines enums and Pydantic models for play types, bid events, stir actions,
player representation, and completed trick data.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from server.sm.card_model import Card, Suit


# ---- Enums ----


class PlayType(str, Enum):
    SINGLE = "single"
    PAIR = "pair"
    TRACTOR = "tractor"
    THROW = "throw"


# ---- Action / Event Models ----


class PlayAction(BaseModel):
    """A player's play: the type of play and the cards involved."""

    model_config = ConfigDict(frozen=True)

    type: PlayType
    cards: list[Card]


class BidEvent(BaseModel):
    """Records a bid event: revealing cards from hand during bidding."""

    model_config = ConfigDict(frozen=True)

    player: int
    cards: list[Card]
    kind: Literal["trump_rank", "joker"]
    suit: Suit | None
    joker_type: Literal["big", "small"] | None
    count: int


class StirAction(BaseModel):
    """Records a player's stir (change suit) or pass during the stir phase."""

    model_config = ConfigDict(frozen=True)

    player: int
    kind: Literal["stir", "pass"]
    new_suit: Suit | None


# ---- Player Model ----


class Player(BaseModel):
    """Represents a player in the game with index, team, and hand."""

    index: int
    team: Literal[0, 1]
    hand: list[Card]
    is_declarer: bool = False


# ---- Completed Trick Models ----


class CompletedTrickSlot(BaseModel):
    """One player's contribution to a completed trick."""

    model_config = ConfigDict(frozen=True)

    player: int
    cards: list[Card]


class CompletedTrick(BaseModel):
    """A fully completed trick with all slots, winner, and points."""

    model_config = ConfigDict(frozen=True)

    lead_player: int
    lead_type: PlayType
    slots: list[CompletedTrickSlot]
    winner: int
    points: int
