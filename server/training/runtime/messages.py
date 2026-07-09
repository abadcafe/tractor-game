"""Worker process message protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from server.training.ppo import PPOUpdateStats
from server.training.runtime.state import RuntimeTrainingState

type WorkerCommandKind = Literal[
    "load_state",
    "start_sampling",
    "stop_sampling",
    "update",
    "snapshot",
    "setup",
]


@dataclass(frozen=True, slots=True)
class WorkerStartSamplingCommand:
    """Start worker game envs for one policy version."""

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
class WorkerStopSamplingCommand:
    """Stop active worker game envs for one policy version."""

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
    | WorkerStopSamplingCommand
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
        | WorkerStopSamplingCommand
        | WorkerUpdateCommand
        | WorkerSnapshotCommand
        | StopWorkerCommand,
    ):
        return value
    return None


@dataclass(frozen=True, slots=True)
class WorkerSamplingStarted:
    """Worker response after sampling tasks are running."""

    worker_index: int
    policy_version: int

    def __post_init__(self) -> None:
        assert self.worker_index >= 0
        assert self.policy_version >= 0


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
    """Worker response after active game envs have stopped."""

    worker_index: int
    policy_version: int
    cancelled_env_count: int

    def __post_init__(self) -> None:
        assert self.worker_index >= 0
        assert self.policy_version >= 0
        assert self.cancelled_env_count >= 0


@dataclass(frozen=True, slots=True)
class WorkerSamplingAlreadyStopped:
    """Worker response when stop finds no active sampling task."""

    worker_index: int
    policy_version: int

    def __post_init__(self) -> None:
        assert self.worker_index >= 0
        assert self.policy_version >= 0


@dataclass(frozen=True, slots=True)
class WorkerCommandRejected:
    """Business rejection from one worker command."""

    worker_index: int
    command: WorkerCommandKind
    policy_version: int | None
    reason: str

    def __post_init__(self) -> None:
        assert self.worker_index >= 0
        if self.policy_version is not None:
            assert self.policy_version >= 0
        assert self.reason


type WorkerResponse = (
    WorkerStateLoaded
    | WorkerSamplingStarted
    | WorkerUpdateCompleted
    | WorkerSnapshotCompleted
    | WorkerSamplingStopped
    | WorkerSamplingAlreadyStopped
    | WorkerCommandRejected
)


def decode_worker_response(value: object) -> WorkerResponse | None:
    """Return a worker response when the payload type is valid."""
    if isinstance(
        value,
        WorkerStateLoaded
        | WorkerSamplingStarted
        | WorkerUpdateCompleted
        | WorkerSnapshotCompleted
        | WorkerSamplingStopped
        | WorkerSamplingAlreadyStopped
        | WorkerCommandRejected,
    ):
        return value
    return None
