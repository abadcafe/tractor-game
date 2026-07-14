"""Private model-rank process message protocol."""

from __future__ import annotations

from dataclasses import dataclass

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
    rollout_id: str

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert self.rollout_id


@dataclass(frozen=True, slots=True)
class ModelRankSnapshotCommand:
    """Capture this model rank state for checkpointing."""

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
    | ModelRankSnapshotCommand
    | ModelRankStopCommand
)


def decode_model_rank_command(
    value: object,
) -> ModelRankCommand | None:
    """Return a model-rank command when the payload type is valid."""
    if isinstance(
        value,
        ModelRankLoadStateCommand
        | ModelRankUpdateCommand
        | ModelRankSnapshotCommand
        | ModelRankStopCommand,
    ):
        return value
    return None


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
    policy_version: int
    update_stats: PPOUpdateStats

    def __post_init__(self) -> None:
        assert self.model_rank_index >= 0
        assert self.rank_index >= 0
        assert self.policy_version >= 0


@dataclass(frozen=True, slots=True)
class ModelRankSnapshotCompleted:
    """Model-rank checkpoint state snapshot."""

    model_rank_index: int
    policy_version: int
    state: RuntimeTrainingState

    def __post_init__(self) -> None:
        assert self.model_rank_index >= 0
        assert self.policy_version >= 0


@dataclass(frozen=True, slots=True)
class ModelRankRejected:
    """Business rejection from one model rank."""

    model_rank_index: int
    reason: str


type ModelRankResponse = (
    ModelRankStateLoaded
    | ModelRankUpdateCompleted
    | ModelRankSnapshotCompleted
    | ModelRankRejected
)


def decode_model_rank_response(
    value: object,
) -> ModelRankResponse | None:
    """Return a model-rank response when the payload type is valid."""
    if isinstance(
        value,
        ModelRankStateLoaded
        | ModelRankUpdateCompleted
        | ModelRankSnapshotCompleted
        | ModelRankRejected,
    ):
        return value
    return None
