"""Shared type definitions used across Shengji/Tractor state machines.

Defines Pydantic models for bid events, stir actions, sub-plays,
player representation, and completed trick data.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from server.rules.cards import Card, Suit


# ---- State / domain aliases ----

type RoundPhase = Literal["DEAL_BID", "STIRRING", "PLAYING", "SCORING", "WAITING"]
type DealBidPhase = Literal["DEALING", "COMPLETE", "NO_BID"]
type StirringPhase = Literal["WAITING", "EXCHANGING", "COMPLETE"]
type ExchangePhase = Literal["PICKED_UP", "COMPLETE"]
type TrickPhase = Literal["LEADING", "FOLLOWING", "RESOLVED"]


# ---- Action / Event Models ----


class BidEvent(BaseModel):
    """Records a bid event: revealing cards from hand during bidding."""

    model_config = ConfigDict(frozen=True)

    player: int
    cards: list[Card]
    kind: Literal["trump_rank", "joker"]
    suit: Suit | None
    joker_type: Literal["big", "small"] | None
    count: int

    @model_validator(mode="after")
    def _validate_suit_kind_consistency(self) -> Self:
        if self.kind == "trump_rank" and self.suit is None:
            raise ValueError("suit must be set when kind='trump_rank'")
        if self.kind == "joker" and self.suit is not None:
            raise ValueError("suit must be None when kind='joker'")
        return self


class StirAction(BaseModel):
    """Records a player's stir (change suit) or pass during the stir phase."""

    model_config = ConfigDict(frozen=True)

    player: int
    kind: Literal["stir", "pass"]
    new_suit: Suit | None

    @model_validator(mode="after")
    def _validate_suit_kind_consistency(self) -> Self:
        if self.kind == "pass" and self.new_suit is not None:
            raise ValueError("new_suit must be None when kind='pass'")
        return self


class FailedThrow(BaseModel):
    """Public event for a failed throw that forced a smaller sub-play."""

    model_config = ConfigDict(frozen=True)

    player: int
    attempted_cards: list[Card]
    forced_cards: list[Card]


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
    slots: list[CompletedTrickSlot]
    winner: int
    points: int
