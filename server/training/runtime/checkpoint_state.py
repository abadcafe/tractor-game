"""Checkpoint bridge for portable runtime training state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from server.foundation import result as _result
from server.foundation.result import Ok, Rejected
from server.training.config import ModelConfig, TrainConfig
from server.training.runtime.config import ExecutionConfig
from server.training.runtime.state import (
    RuntimeTrainingState,
    capture_runtime_training_state,
    load_runtime_training_state,
)
from server.training.torch_checkpoints.load import load_torch_checkpoint
from server.training.torch_checkpoints.save import (
    TorchCheckpointSaveResult,
    save_torch_checkpoint,
)
from server.training.training_state import (
    LoadedTrainingState,
    create_training_state,
)


@dataclass(frozen=True, slots=True)
class RuntimeCheckpointState:
    """Portable model/optimizer state plus training progress."""

    state: RuntimeTrainingState
    total_rounds: int
    total_samples: int
    total_updates: int

    def __post_init__(self) -> None:
        assert self.total_rounds >= 0
        assert self.total_samples >= 0
        assert self.total_updates >= 0


def create_initial_runtime_checkpoint_state(
    *,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
) -> RuntimeCheckpointState:
    """Create the initial portable runtime checkpoint state."""
    loaded = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
        device=torch.device("cpu"),
    )
    return _runtime_checkpoint_state_from_loaded(loaded)


def load_runtime_checkpoint_state(
    *,
    path: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
) -> _result.Ok[RuntimeCheckpointState] | _result.Rejected:
    """Load a portable runtime state from a torch checkpoint."""
    loaded_result = load_torch_checkpoint(
        path=path,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
        device=torch.device("cpu"),
    )
    if isinstance(loaded_result, Rejected):
        return loaded_result
    return Ok(
        value=_runtime_checkpoint_state_from_loaded(loaded_result.value)
    )


def save_runtime_checkpoint_state(
    *,
    manifest_paths: tuple[Path, ...],
    state: RuntimeTrainingState,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    total_rounds: int,
    total_samples: int,
    total_updates: int,
    retained_update_count: int,
) -> _result.Ok[TorchCheckpointSaveResult] | _result.Rejected:
    """Persist a portable runtime state as a torch checkpoint."""
    assert retained_update_count >= 0
    assert total_rounds >= 0
    assert total_samples >= 0
    assert total_updates >= 0
    loaded = _loaded_state_from_runtime_state(
        state=state,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
        total_rounds=total_rounds,
        total_samples=total_samples,
        total_updates=total_updates,
    )
    return save_torch_checkpoint(
        manifest_paths=manifest_paths,
        model=loaded.model,
        trainer=loaded.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=total_rounds,
        total_samples=total_samples,
        total_updates=total_updates,
        retained_update_count=retained_update_count,
    )


def _runtime_checkpoint_state_from_loaded(
    loaded: LoadedTrainingState,
) -> RuntimeCheckpointState:
    return RuntimeCheckpointState(
        state=capture_runtime_training_state(
            model=loaded.model,
            trainer=loaded.trainer,
        ),
        total_rounds=loaded.total_rounds,
        total_samples=loaded.total_samples,
        total_updates=loaded.total_updates,
    )


def _loaded_state_from_runtime_state(
    *,
    state: RuntimeTrainingState,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    total_rounds: int,
    total_samples: int,
    total_updates: int,
) -> LoadedTrainingState:
    loaded = _create_cpu_training_state_without_rng_side_effect(
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
    )
    load_runtime_training_state(state=loaded, snapshot=state)
    return LoadedTrainingState(
        model=loaded.model,
        trainer=loaded.trainer,
        total_rounds=total_rounds,
        total_samples=total_samples,
        total_updates=total_updates,
    )


def _create_cpu_training_state_without_rng_side_effect(
    *,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
) -> LoadedTrainingState:
    cpu_rng_state = torch.random.get_rng_state()
    try:
        return create_training_state(
            model_config=model_config,
            train_config=train_config,
            execution_config=execution_config,
            device=torch.device("cpu"),
        )
    finally:
        torch.random.set_rng_state(cpu_rng_state)
