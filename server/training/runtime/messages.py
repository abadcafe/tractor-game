"""Worker process message protocol."""

from __future__ import annotations

from dataclasses import dataclass

from server.training.ppo import PPOUpdateStats
from server.training.runtime.state import RuntimeTrainingState


@dataclass(frozen=True, slots=True)
class WorkerStartSamplingCommand:
    """Run worker game envs until the worker arena is full."""

    policy_version: int
    game_env_count: int

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert self.game_env_count > 0


@dataclass(frozen=True, slots=True)
class WorkerLoadStateCommand:
    """Load canonical state into a worker-local model rank."""

    state: RuntimeTrainingState
    policy_version: int

    def __post_init__(self) -> None:
        assert self.policy_version >= 0


@dataclass(frozen=True, slots=True)
class WorkerUpdateCommand:
    """Apply one synchronized PPO update rank."""

    policy_version: int

    def __post_init__(self) -> None:
        assert self.policy_version >= 0


@dataclass(frozen=True, slots=True)
class WorkerSnapshotCommand:
    """Capture the worker-local model rank state for checkpointing."""

    policy_version: int

    def __post_init__(self) -> None:
        assert self.policy_version >= 0


@dataclass(frozen=True, slots=True)
class StopWorkerCommand:
    """Command to stop a worker process."""

    reason: str


type WorkerCommand = (
    WorkerLoadStateCommand
    | WorkerStartSamplingCommand
    | WorkerUpdateCommand
    | WorkerSnapshotCommand
    | StopWorkerCommand
)


def decode_worker_command(value: object) -> WorkerCommand | None:
    """Return a worker command when the payload type is valid."""
    if isinstance(
        value,
        WorkerLoadStateCommand
        | WorkerStartSamplingCommand
        | WorkerUpdateCommand
        | WorkerSnapshotCommand
        | StopWorkerCommand,
    ):
        return value
    return None


@dataclass(frozen=True, slots=True)
class WorkerStateLoaded:
    """Worker acknowledged loading a canonical state."""

    worker_index: int
    policy_version: int

    def __post_init__(self) -> None:
        assert self.worker_index >= 0
        assert self.policy_version >= 0


@dataclass(frozen=True, slots=True)
class WorkerUpdateCompleted:
    """Worker response after synchronized PPO update."""

    worker_index: int
    policy_version: int
    update_stats: PPOUpdateStats

    def __post_init__(self) -> None:
        assert self.worker_index >= 0
        assert self.policy_version >= 0


@dataclass(frozen=True, slots=True)
class WorkerSnapshotCompleted:
    """Worker response carrying a checkpoint state snapshot."""

    worker_index: int
    policy_version: int
    state: RuntimeTrainingState

    def __post_init__(self) -> None:
        assert self.worker_index >= 0
        assert self.policy_version >= 0


@dataclass(frozen=True, slots=True)
class WorkerSamplingStopped:
    """Worker response after its rollout arena becomes full."""

    worker_index: int
    policy_version: int

    def __post_init__(self) -> None:
        assert self.worker_index >= 0
        assert self.policy_version >= 0


@dataclass(frozen=True, slots=True)
class WorkerRejected:
    """Business rejection from one worker."""

    worker_index: int
    reason: str


type WorkerResponse = (
    WorkerStateLoaded
    | WorkerUpdateCompleted
    | WorkerSnapshotCompleted
    | WorkerSamplingStopped
    | WorkerRejected
)


def decode_worker_response(value: object) -> WorkerResponse | None:
    """Return a worker response when the payload type is valid."""
    if isinstance(
        value,
        WorkerStateLoaded
        | WorkerUpdateCompleted
        | WorkerSnapshotCompleted
        | WorkerSamplingStopped
        | WorkerRejected,
    ):
        return value
    return None
