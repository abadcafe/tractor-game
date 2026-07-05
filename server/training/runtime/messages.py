"""Worker process message protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from server.training.ppo import PPOUpdateStats
from server.training.rollout_commit import RolloutCommit
from server.training.runtime.state import RuntimeTrainingState
from server.training.runtime.update_wave import SynchronizedUpdateShard


@dataclass(frozen=True, slots=True)
class WorkerRoundSummary:
    """Round metrics returned by one worker."""

    team0_reward: float
    team1_reward: float
    generated_action_count: int
    accepted_action_count: int
    action_choice_count: int
    decision_count: int
    elapsed_seconds: float
    game_over: bool


@dataclass(frozen=True, slots=True)
class WorkerRolloutCommand:
    """Collect one self-play rollout using the requested policy."""

    state: RuntimeTrainingState | None
    policy_version: int
    episode_id: int

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert self.episode_id >= 0


@dataclass(frozen=True, slots=True)
class WorkerUpdateCommand:
    """Apply one synchronized PPO update rank."""

    policy_version: int
    shard: SynchronizedUpdateShard

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert self.shard.policy_version == self.policy_version


@dataclass(frozen=True, slots=True)
class StopWorkerCommand:
    """Command to stop a worker process."""

    reason: str


type WorkerCommand = (
    WorkerRolloutCommand | WorkerUpdateCommand | StopWorkerCommand
)


@dataclass(frozen=True, slots=True)
class WorkerUpdateCompleted:
    """Worker response after synchronized PPO update."""

    worker_index: int
    update_stats: PPOUpdateStats
    state: RuntimeTrainingState


@dataclass(frozen=True, slots=True)
class WorkerRolloutCompleted:
    """Worker response after rollout collection."""

    worker_index: int
    episode_id: int
    summary: WorkerRoundSummary
    rollout_commit: RolloutCommit


@dataclass(frozen=True, slots=True)
class WorkerRejected:
    """Business rejection from one worker."""

    worker_index: int
    reason: str


type WorkerResponse = (
    WorkerUpdateCompleted | WorkerRolloutCompleted | WorkerRejected
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
