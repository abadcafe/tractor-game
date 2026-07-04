"""Trajectory records collected by training players."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from server.sm.constants import get_team_index
from server.training.choice_trace import SemanticChoiceTrace
from server.training.semantic_actions.codec import semantic_argument_id
from server.training.semantic_actions.values import GeneratedAction
from server.training.tensorize import ObservationTensorBatch


@dataclass(frozen=True, slots=True)
class DecisionStep:
    """One player decision needed for policy-gradient training."""

    player_index: int
    seq: int
    observation_batch: ObservationTensorBatch
    choice_trace: SemanticChoiceTrace
    action: GeneratedAction
    log_probability: float
    value_estimate: float
    entropy: float
    choice_count: int

    def __post_init__(self) -> None:
        arguments = self.action.semantic_trace.arguments
        assert len(self.choice_trace.steps) == len(arguments)
        assert all(
            step.selected_argument_id == semantic_argument_id(argument)
            for step, argument in zip(
                self.choice_trace.steps, arguments, strict=True
            )
        )


@dataclass(frozen=True, slots=True)
class DecisionTransition:
    """One accepted decision with the reward observed after it."""

    decision: DecisionStep
    reward_after_step: float


@dataclass(frozen=True, slots=True)
class TeamTrajectory:
    """One completed team reward stream for PPO target calculation."""

    team_index: int
    transitions: tuple[DecisionTransition, ...]
    terminal_reward: float

    def __post_init__(self) -> None:
        assert self.team_index in (0, 1)
        assert self.transitions
        assert math.isfinite(self.terminal_reward)
        assert all(
            get_team_index(transition.decision.player_index)
            == self.team_index
            for transition in self.transitions
        )
        assert all(
            math.isfinite(transition.reward_after_step)
            for transition in self.transitions
        )


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """Completed trajectories collected under one policy version."""

    trajectories: tuple[TeamTrajectory, ...]

    def transition_count(self) -> int:
        """Return the number of decision transitions in the batch."""
        return sum(
            len(trajectory.transitions)
            for trajectory in self.trajectories
        )

    def is_empty(self) -> bool:
        """Return whether this batch has no trainable transitions."""
        return self.transition_count() == 0


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
