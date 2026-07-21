"""Typed semantic payloads accepted by the observation model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from server.game.rules.card_faces import MAX_FACE_COUNT, CardFace
from server.game.rules.cards import Rank
from server.game.rules.required_progress import (
    ProgressTarget,
    TerminalProgress,
)
from server.training.relative_state import (
    RelativeActor,
    TrickPosition,
    TrumpState,
)

type ActionOccurrence = Literal["fact", "query"]
type TokenActionKind = Literal["bid", "stir", "bottom_exchange", "play"]
type TokenDisposition = Literal["pass", "reveal"]


class RoundField(str, Enum):
    """Closed set of round facts visible to the policy."""

    DECLARER_ACTOR = "declarer_actor"
    OWN_LEVEL = "own_level"
    OPPONENT_LEVEL = "opponent_level"
    OWN_TARGET = "own_target"
    OPPONENT_TARGET = "opponent_target"
    OWN_DISTANCE = "own_distance"
    OPPONENT_DISTANCE = "opponent_distance"
    TRUMP_STATE = "trump_state"
    LEVEL_RANK = "level_rank"
    DEFENDER_POINTS = "defender_points"
    REMAINING_CARDS = "remaining_cards"


@dataclass(frozen=True, slots=True)
class GlobalToken:
    """One mandatory progression level."""

    rank: Rank
    progress_position: int

    def __post_init__(self) -> None:
        assert self.progress_position >= 0


@dataclass(frozen=True, slots=True)
class RoundToken:
    """One typed round fact with a closed value domain."""

    field: RoundField
    value: (
        RelativeActor | Rank | ProgressTarget | TrumpState | int | None
    )
    actor: RelativeActor | None = None

    def __post_init__(self) -> None:
        if self.field == RoundField.DECLARER_ACTOR:
            assert self.value is None or isinstance(
                self.value, RelativeActor
            )
        elif self.field in (
            RoundField.OWN_LEVEL,
            RoundField.OPPONENT_LEVEL,
            RoundField.LEVEL_RANK,
        ):
            assert isinstance(self.value, Rank)
        elif self.field in (
            RoundField.OWN_TARGET,
            RoundField.OPPONENT_TARGET,
        ):
            assert isinstance(self.value, (Rank, TerminalProgress))
        elif self.field == RoundField.TRUMP_STATE:
            assert isinstance(self.value, TrumpState)
        else:
            assert isinstance(self.value, int)
            assert not isinstance(self.value, bool)
            assert self.value >= 0
        if self.field == RoundField.REMAINING_CARDS:
            assert self.actor in (
                RelativeActor.PARTNER,
                RelativeActor.LEFT_ENEMY,
                RelativeActor.RIGHT_ENEMY,
            )
        else:
            assert self.actor is None


@dataclass(frozen=True, slots=True)
class TrickToken:
    """One trick header and its adjudicated result."""

    status: Literal["open", "completed"]
    winner: RelativeActor | None
    points: int | None

    def __post_init__(self) -> None:
        if self.status == "open":
            assert self.winner is None
            assert self.points is None
            return
        assert self.winner is not None
        assert self.points is not None
        assert self.points >= 0


@dataclass(frozen=True, slots=True)
class ActionToken:
    """One historical action head or current decision query."""

    occurrence: ActionOccurrence
    kind: TokenActionKind
    actor: RelativeActor | None
    disposition: TokenDisposition | None
    trick_position: TrickPosition | None

    def __post_init__(self) -> None:
        if self.occurrence == "query":
            assert self.actor is None
            assert self.disposition is None
        else:
            assert self.actor is not None
            if self.kind in ("bid", "stir"):
                assert self.disposition is not None
            else:
                assert self.disposition is None
        if self.kind == "play":
            assert self.trick_position is not None
        else:
            assert self.trick_position is None


@dataclass(frozen=True, slots=True)
class CardToken:
    """One canonical card face with universal multiplicity."""

    face: CardFace
    count: int

    def __post_init__(self) -> None:
        assert 1 <= self.count <= MAX_FACE_COUNT


type TokenPayload = (
    GlobalToken | RoundToken | TrickToken | ActionToken | CardToken
)


__all__ = (
    "ActionToken",
    "CardToken",
    "GlobalToken",
    "RoundField",
    "RoundToken",
    "TokenPayload",
    "TrickToken",
)
