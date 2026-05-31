"""Game phase, play type, and action type definitions for 升级 (Shengji/Tractor).

Defines Phase, PlayType enums and action models (PlayAction, BidAction, StirAction).
Game state models are in game_state module. API response types are in api_types.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from server.engine.card import Card, Rank, Suit


# ---- Enums ----


class Phase(str, Enum):
    """Game phase enumeration."""
    DEALING = "dealing"
    BIDDING = "bidding"
    STIRRING = "stirring"
    EXCHANGE = "exchange"
    PLAYING = "playing"
    SCORING = "scoring"
    GAME_OVER = "game_over"


class PlayType(str, Enum):
    """Play type enumeration for card plays."""
    SINGLE = "single"
    PAIR = "pair"
    TRACTOR = "tractor"
    THROW = "throw"


# ---- Action Models ----


class PlayAction(BaseModel):
    """A play action consisting of a type and cards to play."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    type: PlayType
    cards: list[Card]


class BidAction(BaseModel):
    """A bid action by a player."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    player_index: int
    level: Rank | None = None
    pass_: bool = Field(alias="pass")


class StirAction(BaseModel):
    """A stir action by a player to change trump suit."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    player_index: int
    new_trump_suit: Suit | None = None
    level: Rank | None = None
