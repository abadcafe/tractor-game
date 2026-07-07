"""Private model-rank process message protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from server.training.ppo import PPOUpdateStats
from server.training.runtime.state import RuntimeTrainingState


@dataclass(frozen=True, slots=True)
class ModelRankLoadStateCommand:
    """Load canonical state into one model rank before rollout."""

    state: RuntimeTrainingState
    policy_version: int

    def __post_init__(self) -> None:
        assert self.policy_version >= 0


@dataclass(frozen=True, slots=True)
class ModelRankUpdateCommand:
    """Run PPO update from shared rollout arenas."""

    policy_version: int

    def __post_init__(self) -> None:
        assert self.policy_version >= 0


@dataclass(frozen=True, slots=True)
class ModelRankStopCommand:
    """Command to stop a model-rank process."""

    reason: str


type ModelRankCommand = (
    ModelRankLoadStateCommand
    | ModelRankUpdateCommand
    | ModelRankStopCommand
)


@dataclass(frozen=True, slots=True)
class ModelRankStateLoaded:
    """Model rank acknowledged loading a canonical state."""

    model_rank_index: int
    policy_version: int


@dataclass(frozen=True, slots=True)
class ModelRankUpdateCompleted:
    """Model-rank update result."""

    model_rank_index: int
    rank_index: int
    update_stats: PPOUpdateStats
    state: RuntimeTrainingState


@dataclass(frozen=True, slots=True)
class ModelRankRejected:
    """Business rejection from one model rank."""

    model_rank_index: int
    reason: str


type ModelRankResponse = (
    ModelRankStateLoaded | ModelRankUpdateCompleted | ModelRankRejected
)


class ModelRankCommandReceiver(Protocol):
    """Receive commands in a model-rank process."""

    def poll(self, timeout: float = 0.0) -> bool: ...

    def recv(self) -> ModelRankCommand: ...


class ModelRankCommandSender(Protocol):
    """Send commands to one model-rank process."""

    def send(self, item: ModelRankCommand) -> None: ...


class ModelRankResponseReceiver(Protocol):
    """Receive model-rank responses in the coordinator."""

    def get(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> ModelRankResponse: ...


class ModelRankResponseSender(Protocol):
    """Send model-rank responses from a model-rank process."""

    def put(self, item: ModelRankResponse) -> None: ...
