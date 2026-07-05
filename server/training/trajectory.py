"""Trajectory records collected by training players."""

from __future__ import annotations

from dataclasses import dataclass, field

from server.training.policy_sampling.records import DecisionHandle
from server.training.semantic_actions.values import GeneratedAction


@dataclass(frozen=True, slots=True)
class DecisionStep:
    """One accepted player decision plus its replay handle."""

    player_index: int
    seq: int
    action: GeneratedAction
    decision_handle: DecisionHandle
    choice_count: int

    def __post_init__(self) -> None:
        assert self.player_index in (0, 1, 2, 3)
        assert self.seq >= 0
        assert self.choice_count > 0


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
