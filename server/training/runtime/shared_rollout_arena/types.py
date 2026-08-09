"""Public value types for shared rollout arenas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class RolloutArenaLock(Protocol):
    def acquire(self) -> bool: ...

    def release(self) -> None: ...


class RolloutProgressCondition(RolloutArenaLock, Protocol):
    def wait(self, timeout: float | None = None) -> bool: ...

    def notify_all(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RolloutArenaHandle:
    """Picklable handle for one worker-owned arena."""

    worker_index: int
    shared_memory_name: str
    round_capacity: int
    decision_capacity: int
    trajectory_capacity: int
    lock: RolloutArenaLock
    progress_condition: RolloutProgressCondition

    def __post_init__(self) -> None:
        assert self.worker_index >= 0
        assert self.shared_memory_name
        assert self.round_capacity > 0
        assert self.decision_capacity > 0
        assert self.trajectory_capacity > 0


@dataclass(frozen=True, slots=True)
class RolloutRoundMetrics:
    """Metrics belonging to one completed model-vs-Auto round."""

    model_reward: float
    auto_reward: float
    model_action_count: int
    auto_action_count: int
    forced_action_count: int
    trainable_decision_count: int
    scored_choice_step_count: int
    elapsed_seconds: float
    game_over: bool
    model_declarer: bool

    def __post_init__(self) -> None:
        assert self.model_action_count >= 0
        assert self.auto_action_count >= 0
        assert self.forced_action_count >= 0
        assert self.trainable_decision_count > 0
        assert self.scored_choice_step_count > 0
        assert self.elapsed_seconds >= 0.0
        assert self.model_reward + self.auto_reward == 0.0


@dataclass(frozen=True, slots=True)
class RolloutArenaAppendResult:
    """Atomic result of appending one complete round."""

    accepted: bool
    capacity_reached: bool


@dataclass(frozen=True, slots=True)
class RolloutArenaSnapshot:
    """Coordinator-visible aggregate of rollout arenas."""

    policy_version: int
    round_capacity: int
    round_count: int
    decision_count: int
    trajectory_count: int
    model_action_count: int
    auto_action_count: int
    forced_action_count: int
    trainable_decision_count: int
    scored_choice_step_count: int
    game_over_count: int
    rejected_round_count: int
    cancelled_env_count: int
    total_trace_step_count: int
    max_trace_step_count: int
    model_win_count: int
    auto_win_count: int
    model_declarer_round_count: int
    model_reward_sum: float
    auto_reward_sum: float
    elapsed_seconds_max: float

    def average_model_reward(self) -> float:
        if self.round_count == 0:
            return 0.0
        return self.model_reward_sum / self.round_count

    def average_auto_reward(self) -> float:
        if self.round_count == 0:
            return 0.0
        return self.auto_reward_sum / self.round_count


__all__ = (
    "RolloutArenaAppendResult",
    "RolloutArenaHandle",
    "RolloutArenaSnapshot",
    "RolloutRoundMetrics",
)
