"""Model evaluation contracts consumed by belief updates and search."""

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
class ModelQuery:
    """One observation paired with its exact legal action space."""

    observation: Observation
    legal_actions: LegalActionIndex
    seat: Seat


@dataclass(frozen=True, slots=True)
class ActionDrawKey:
    """Stable random stream identity independent of runtime batching."""

    seed: int
    ordinal: int

    def __post_init__(self) -> None:
        assert self.seed >= 0
        assert self.ordinal >= 0


@dataclass(frozen=True, slots=True)
class SampledAction:
    """One sampled legal semantic action and its policy probability."""

    action: GeneratedAction
    log_probability: float


@dataclass(frozen=True, slots=True)
class ActionSamples:
    """Samples returned for one model query in draw-key order."""

    samples: tuple[SampledAction, ...]

    def __post_init__(self) -> None:
        assert self.samples


@dataclass(frozen=True, slots=True)
class ActionSampleRequest:
    """One model query paired with explicit independent random draws."""

    query: ModelQuery
    draws: tuple[ActionDrawKey, ...]

    def __post_init__(self) -> None:
        assert self.draws
        assert len(set(self.draws)) == len(self.draws)


@dataclass(frozen=True, slots=True)
class ActionScoreRequest:
    """One supplied action to score in its originating model query."""

    query: ModelQuery
    action: GeneratedAction


class ModelEvaluator(Protocol):
    """Read-only current-checkpoint model operations required by AI."""

    async def sample(
        self,
        *,
        requests: tuple[ActionSampleRequest, ...],
    ) -> Ok[tuple[ActionSamples, ...]] | Rejected:
        """Sample every explicit draw without batching-dependent RNG."""
        ...

    async def score(
        self,
        *,
        requests: tuple[ActionScoreRequest, ...],
    ) -> Ok[tuple[float, ...]] | Rejected:
        """Return masked log probabilities for supplied actions."""
        ...

    async def values(
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

    async def decide(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
    ) -> Ok[commands.Command] | Rejected:
        """Return one strategic command for the current view."""
        ...


__all__ = (
    "AIControllerPort",
    "ActionDrawKey",
    "ActionSampleRequest",
    "ActionSamples",
    "ActionScoreRequest",
    "ModelEvaluator",
    "ModelQuery",
    "SampledAction",
)
