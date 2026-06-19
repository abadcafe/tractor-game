"""Shared type definitions used across Shengji/Tractor state machines.

Defines Pydantic models for bid events, stir actions, sub-plays,
player representation, and completed trick data.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from server.sm.card_model import Card, Suit


# ---- Enums ----


# ---- Action / Event Models ----


class SubPlay(BaseModel):
    """A sub-pattern within a play: single, pair, or tractor.

    pair_count: 0=single, 1=pair, >=2=tractor
    suit: effective suit of this sub-play ("trump" or a Suit enum)
    """
    model_config = ConfigDict(frozen=True)

    pair_count: int
    cards: list[Card]
    suit: Suit | str

    @model_validator(mode="after")
    def _validate_pair_count_and_cards(self) -> "SubPlay":
        if self.pair_count < 0:
            raise ValueError("pair_count must be >= 0")
        if len(self.cards) > 0:
            expected = 1 if self.pair_count == 0 else self.pair_count * 2
            if len(self.cards) != expected:
                raise ValueError(
                    f"cards count ({len(self.cards)}) must equal {expected} "
                    f"for pair_count={self.pair_count}"
                )
        return self

    @property
    def sub_level(self) -> int:
        """Sub-play level: pair_count + 1.

        single=1, pair=2, 2-pair tractor=3, 3-pair tractor=4, ...
        """
        return self.pair_count + 1


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
    def _validate_suit_kind_consistency(self) -> "BidEvent":
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
    def _validate_suit_kind_consistency(self) -> "StirAction":
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
