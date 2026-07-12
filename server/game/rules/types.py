"""Rule-level data types for Shengji/Tractor card analysis."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from .cards import Card, Suit

type EffectiveSuit = Suit | Literal["trump"]
type PlayShapeKind = Literal[
    "empty", "single", "pair", "tractor", "cards"
]


class SubPlay(BaseModel):
    """A sub-pattern within a play: single, pair, or tractor.

    pair_count: 0=single, 1=pair, >=2=tractor
    suit: effective suit of this sub-play ("trump" or a Suit enum)
    """

    model_config = ConfigDict(frozen=True)

    pair_count: int
    cards: list[Card]
    suit: EffectiveSuit

    @model_validator(mode="after")
    def _validate_pair_count_and_cards(self) -> Self:
        if self.pair_count < 0:
            raise ValueError("pair_count must be >= 0")
        if len(self.cards) > 0:
            expected = (
                1 if self.pair_count == 0 else self.pair_count * 2
            )
            if len(self.cards) != expected:
                raise ValueError(
                    f"cards count ({len(self.cards)}) must equal"
                    f"{expected}"
                    f"for pair_count={self.pair_count}"
                )
        return self

    @property
    def sub_level(self) -> int:
        """Sub-play level: pair_count + 1.

        single=1, pair=2, 2-pair tractor=3, 3-pair tractor=4, ...
        """
        return self.pair_count + 1


class PlayShapeInfo(BaseModel):
    """Structured description of a played shape for rejection text."""

    model_config = ConfigDict(frozen=True)

    kind: PlayShapeKind
    suit: EffectiveSuit | None
    card_count: int
    pair_count: int | None = None
