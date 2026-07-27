"""Model evaluation contract used by belief updates and search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from server.foundation.result import Ok, Rejected
from server.game import Seat, commands
from server.game.snapshots import PlayerSnapshot
from server.training.legal_actions import LegalActionIndex
from server.training.observation import Observation
from server.training.semantic_actions import GeneratedAction


@dataclass(frozen=True, slots=True)
class ActionEvaluation:
    """One legal semantic action with policy and value estimates."""

    action: GeneratedAction
    log_probability: float
    value: float


@dataclass(frozen=True, slots=True)
class ModelQuery:
    """One observation paired with its exact legal action space."""

    observation: Observation
    legal_actions: LegalActionIndex
    seat: Seat


@dataclass(frozen=True, slots=True)
class ActionScoreRequest:
    """One supplied action to score in its originating model query."""

    query: ModelQuery
    action: GeneratedAction


class ModelEvaluator(Protocol):
    """Read-only current-checkpoint model operations required by AI."""

    def sample(
        self,
        *,
        queries: tuple[ModelQuery, ...],
        decision_seed: int,
    ) -> Ok[tuple[ActionEvaluation, ...]] | Rejected:
        """Sample one legal action for every supplied query."""
        ...

    def score(
        self,
        *,
        requests: tuple[ActionScoreRequest, ...],
    ) -> Ok[tuple[ActionEvaluation, ...]] | Rejected:
        """Score supplied actions under exact masked policy logits."""
        ...

    def values(
        self,
        *,
        observations: tuple[Observation, ...],
    ) -> Ok[tuple[float, ...]] | Rejected:
        """Estimate each observer's zero-sum round value."""
        ...


class AIControllerPort(Protocol):
    """Sequenced observation and decision boundary for players."""

    def observe(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
        error: str | None,
    ) -> Ok[None] | Rejected:
        """Record one complete player observation."""
        ...

    def decide(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
    ) -> Ok[commands.Command] | Rejected:
        """Return one strategic command for the current view."""
        ...


__all__ = (
    "AIControllerPort",
    "ActionEvaluation",
    "ActionScoreRequest",
    "ModelEvaluator",
    "ModelQuery",
)
