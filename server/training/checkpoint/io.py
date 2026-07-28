"""Training adapter over the policy-model checkpoint contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from server.foundation import result as _result
from server.policy_model.checkpoint import (
    CheckpointMetadata,
    CheckpointSaveResult,
    read_checkpoint_metadata,
    read_checkpoint_snapshot,
    restore_policy_model,
    save_checkpoint,
)
from server.policy_model.network import ModelConfig, PolicyValueModel
from server.training.checkpoint.validation import (
    train_config_from_json,
    validate_optimizer_state_payload,
)
from server.training.config import TrainConfig
from server.training.lifecycle.state import LoadedTrainingState
from server.training.ppo import PPOTrainer
from server.training.runtime.config import ExecutionConfig


@dataclass(frozen=True, slots=True)
class TrainingCheckpointMetadata:
    """Checkpoint metadata decoded into training-owned configuration."""

    model_config: ModelConfig
    train_config: TrainConfig
    total_rounds: int
    total_samples: int
    total_updates: int


def read_training_checkpoint_metadata(
    path: Path,
) -> _result.Ok[TrainingCheckpointMetadata] | _result.Rejected:
    """Read and validate the training-owned portion of metadata."""
    metadata_result = read_checkpoint_metadata(path)
    if isinstance(metadata_result, _result.Rejected):
        return metadata_result
    return _training_metadata(
        path=path,
        metadata=metadata_result.value,
    )


def load_training_checkpoint(
    *,
    path: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    device: torch.device,
) -> _result.Ok[LoadedTrainingState] | _result.Rejected:
    """Restore model, optimizer, and progress for resumed training."""
    checkpoint_result = read_checkpoint_snapshot(path)
    if isinstance(checkpoint_result, _result.Rejected):
        return checkpoint_result
    checkpoint = checkpoint_result.value
    metadata_result = _training_metadata(
        path=path,
        metadata=checkpoint.manifest.metadata,
    )
    if isinstance(metadata_result, _result.Rejected):
        return metadata_result
    metadata = metadata_result.value
    if metadata.model_config != model_config:
        return _checkpoint_mismatch(
            path, "model_config does not match checkpoint metadata"
        )
    if metadata.train_config.seed != train_config.seed:
        return _checkpoint_mismatch(
            path, "train_config.seed does not match checkpoint metadata"
        )
    model_result = restore_policy_model(
        checkpoint=checkpoint,
        device=device,
    )
    if isinstance(model_result, _result.Rejected):
        return model_result
    model = model_result.value
    optimizer_check = validate_optimizer_state_payload(
        state=checkpoint.payload.optimizer_state,
        parameters=tuple(model.parameters()),
        path=checkpoint.state_path,
    )
    if isinstance(optimizer_check, _result.Rejected):
        return optimizer_check
    trainer = PPOTrainer(
        model=model,
        train_config=train_config,
        device=device,
        profile_mode=execution_config.ppo_profile,
    )
    trainer.load_optimizer_state(checkpoint.payload.optimizer_state)
    return _result.Ok(
        value=LoadedTrainingState(
            model=model,
            trainer=trainer,
            total_rounds=metadata.total_rounds,
            total_samples=metadata.total_samples,
            total_updates=metadata.total_updates,
        )
    )


def save_training_checkpoint(
    *,
    manifest_paths: tuple[Path, ...],
    model: PolicyValueModel,
    trainer: PPOTrainer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    total_rounds: int,
    total_samples: int,
    total_updates: int,
    retained_update_count: int,
) -> _result.Ok[CheckpointSaveResult] | _result.Rejected:
    """Persist the training-owned state through the neutral codec."""
    return save_checkpoint(
        manifest_paths=manifest_paths,
        model_state=model.state_dict(),
        optimizer_state=trainer.optimizer_state(),
        metadata=CheckpointMetadata(
            model_config=model_config,
            training_config_values=train_config.to_json(),
            total_rounds=total_rounds,
            total_samples=total_samples,
            total_updates=total_updates,
        ),
        retained_update_count=retained_update_count,
    )


def _training_metadata(
    *,
    path: Path,
    metadata: CheckpointMetadata,
) -> _result.Ok[TrainingCheckpointMetadata] | _result.Rejected:
    train_config_result = train_config_from_json(
        metadata.training_config_values,
        path,
    )
    if isinstance(train_config_result, _result.Rejected):
        return train_config_result
    return _result.Ok(
        value=TrainingCheckpointMetadata(
            model_config=metadata.model_config,
            train_config=train_config_result.value,
            total_rounds=metadata.total_rounds,
            total_samples=metadata.total_samples,
            total_updates=metadata.total_updates,
        )
    )


def _checkpoint_mismatch(path: Path, reason: str) -> _result.Rejected:
    return _result.Rejected(
        reason=f"checkpoint mismatch: {path}: {reason}"
    )
