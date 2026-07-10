"""Public value types for shared rollout arenas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class RolloutArenaLock(Protocol):
    """Lock interface used to protect one arena header and columns."""

    def acquire(self) -> bool: ...

    def release(self) -> None: ...


class RolloutProgressCondition(RolloutArenaLock, Protocol):
    """Condition interface used to wait for aggregate arena progress."""

    def wait(self, timeout: float | None = None) -> bool: ...

    def notify_all(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RolloutArenaHandle:
    """Picklable handle for one worker-owned arena."""

    worker_index: int
    shared_memory_name: str
    capacity: int
    lock: RolloutArenaLock
    progress_condition: RolloutProgressCondition

    def __post_init__(self) -> None:
        assert self.worker_index >= 0
        assert self.shared_memory_name
        assert self.capacity > 0


@dataclass(frozen=True, slots=True)
class RolloutRoundMetrics:
    """Metrics for one completed self-play round."""

    team0_reward: float
    team1_reward: float
    generated_action_count: int
    accepted_action_count: int
    action_choice_count: int
    decision_count: int
    elapsed_seconds: float
    game_over: bool

    def __post_init__(self) -> None:
        assert self.generated_action_count >= 0
        assert self.accepted_action_count >= 0
        assert self.action_choice_count >= 0
        assert self.decision_count >= 0
        assert self.elapsed_seconds >= 0.0


@dataclass(frozen=True, slots=True)
class RolloutArenaAppendResult:
    """Result of appending one completed round to an arena."""

    accepted_sample_count: int
    dropped_sample_count: int
    capacity_reached: bool

    def __post_init__(self) -> None:
        assert self.accepted_sample_count >= 0
        assert self.dropped_sample_count >= 0


@dataclass(frozen=True, slots=True)
class RolloutArenaSnapshot:
    """Coordinator-visible aggregate of filled rollout arenas."""

    policy_version: int
    capacity: int
    round_count: int
    sample_count: int
    generated_action_count: int
    accepted_action_count: int
    action_choice_count: int
    game_over_count: int
    dropped_sample_count: int
    cancelled_env_count: int
    total_step_count: int
    max_step_count: int
    team0_reward_sum: float
    team1_reward_sum: float
    elapsed_seconds_max: float

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert self.capacity >= 0
        assert self.round_count >= 0
        assert self.sample_count >= 0
        assert self.generated_action_count >= 0
        assert self.accepted_action_count >= 0
        assert self.action_choice_count >= 0
        assert self.game_over_count >= 0
        assert self.dropped_sample_count >= 0
        assert self.cancelled_env_count >= 0
        assert self.total_step_count >= 0
        assert self.max_step_count >= 0
        assert self.elapsed_seconds_max >= 0.0

    def average_team0_reward(self) -> float:
        """Return mean team0 reward for accepted rounds."""
        if self.round_count == 0:
            return 0.0
        return self.team0_reward_sum / self.round_count

    def average_team1_reward(self) -> float:
        """Return mean team1 reward for accepted rounds."""
        if self.round_count == 0:
            return 0.0
        return self.team1_reward_sum / self.round_count
