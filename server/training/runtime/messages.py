"""Worker process message protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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
class StopWorkerCommand:
    """Command to stop a worker process."""

    reason: str


type WorkerCommand = (
    WorkerLoadStateCommand
    | WorkerStartSamplingCommand
    | WorkerUpdateCommand
    | StopWorkerCommand
)


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
    update_stats: PPOUpdateStats
    state: RuntimeTrainingState


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
    | WorkerSamplingStopped
    | WorkerRejected
)


class WorkerCommandReceiver(Protocol):
    """Receive commands from the coordinator."""

    def get(self) -> WorkerCommand: ...


class WorkerCommandSender(Protocol):
    """Send commands to one worker."""

    def put(self, item: WorkerCommand) -> None: ...


class WorkerResponseReceiver(Protocol):
    """Receive worker responses in the coordinator."""

    def get(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> WorkerResponse: ...


class WorkerResponseSender(Protocol):
    """Send worker responses from a worker process."""

    def put(self, item: WorkerResponse) -> None: ...
