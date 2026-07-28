"""Complete viewer-relative observation entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from server.game.rules.cards import Rank
from server.game.rules.cards.faces import FaceCount
from server.game.rules.progression import ProgressTarget
from server.policy_model._schema.positions import (
    RelativeActor,
    TrickPosition,
    TrumpState,
)
from server.policy_model.observation.relative_state.actions import (
    RelativePlayAction,
    RelativeRoundAction,
)
from server.policy_model.observation.structure import (
    RoundEventOrdinal,
    TrickRecency,
)

type DecisionKind = Literal["bid", "stir", "bottom_exchange", "play"]
type TrickStatus = Literal["open", "completed"]


@dataclass(frozen=True, slots=True)
class GlobalContext:
    """Rules that may vary across games sharing one model schema."""

    mandatory_levels: tuple[Rank, ...]


@dataclass(frozen=True, slots=True)
class RoundContext:
    """Current round facts expressed only in viewer-relative terms."""

    declarer_actor: RelativeActor | None
    own_level: Rank
    opponent_level: Rank
    own_target: ProgressTarget
    opponent_target: ProgressTarget
    own_distance_to_target: int
    opponent_distance_to_target: int
    trump: TrumpState
    level_rank: Rank
    defender_points: int
    partner_remaining: int
    next_opponent_remaining: int
    previous_opponent_remaining: int


@dataclass(frozen=True, slots=True)
class RelativeTrick:
    """One completed or open trick in chronological memory."""

    status: TrickStatus
    recency: TrickRecency
    actions: tuple[RelativePlayAction, ...]
    winner: RelativeActor | None
    points: int | None


@dataclass(frozen=True, slots=True)
class DecisionQuery:
    """The semantic decision requested from the policy."""

    kind: DecisionKind
    round_event: RoundEventOrdinal | None
    trick_position: TrickPosition | None


@dataclass(frozen=True, slots=True)
class RelativeObservation:
    """Lossless model state with no absolute player or team identity."""

    global_context: GlobalContext
    round_context: RoundContext
    round_actions: tuple[RelativeRoundAction, ...]
    tricks: tuple[RelativeTrick, ...]
    hand: tuple[FaceCount, ...]
    visible_bottom: tuple[FaceCount, ...]
    query: DecisionQuery | None


__all__ = (
    "DecisionQuery",
    "GlobalContext",
    "RelativeObservation",
    "RelativeTrick",
    "RoundContext",
)
