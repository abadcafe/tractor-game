"""Trajectory records collected by training players."""

from __future__ import annotations

from dataclasses import dataclass, field

from server.training.action_tokens import ActionQuery, GeneratedAction
from server.training.observation import Observation


@dataclass(frozen=True, slots=True)
class DecisionStep:
    """One player decision needed for policy-gradient training."""

    player_index: int
    seq: int
    observation: Observation
    action_query: ActionQuery
    action: GeneratedAction
    log_probability: float
    value_estimate: float
    entropy: float
    token_count: int


@dataclass(frozen=True, slots=True)
class RewardedDecisionStep:
    """A decision step annotated with terminal team reward."""

    step: DecisionStep
    reward: float


def _step_list() -> list[DecisionStep]:
    return []


@dataclass(slots=True)
class TrajectoryRecorder:
    """Append-only in-memory decision recorder."""

    _steps: list[DecisionStep] = field(default_factory=_step_list)

    def append(self, step: DecisionStep) -> None:
        self._steps.append(step)

    def steps(self) -> tuple[DecisionStep, ...]:
        return tuple(self._steps)

    def clear(self) -> None:
        self._steps.clear()
