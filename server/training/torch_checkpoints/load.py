"""Torch checkpoint load and metadata reading."""

from __future__ import annotations

from pathlib import Path

import torch

from server.foundation import result as _result
from server.training import training_state as _training_state
from server.training.config import ModelConfig, TrainConfig
from server.training.model import TractorPolicyModel
from server.training.ppo import PPOTrainer
from server.training.runtime.config import ExecutionConfig
from server.training.torch_checkpoints.filesystem import (
    validate_checkpoint_dir,
    validate_checkpoint_object_dir,
    validate_checkpoint_objects_dir,
    validate_checkpoint_state_file,
)
from server.training.torch_checkpoints.manifest import (
    manifest_state_file_path,
    read_checkpoint_manifest,
)
from server.training.torch_checkpoints.payload import (
    read_checkpoint_payload,
)
from server.training.torch_checkpoints.schema import (
    CHECKPOINT_OBJECTS_DIR,
    CheckpointManifest,
    TorchCheckpointMetadata,
    checkpoint_corruption,
    sha256_checkpoint_file,
)
from server.training.torch_checkpoints.validation import (
    validate_optimizer_state_payload,
)


def load_torch_checkpoint(
    *,
    path: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    device: torch.device,
) -> _result.Ok[_training_state.LoadedTrainingState] | _result.Rejected:
    """Load trainable state from a torch checkpoint."""
    manifest_result = read_checkpoint_manifest(path)
    if isinstance(manifest_result, _result.Rejected):
        return manifest_result
    manifest = manifest_result.value
    metadata = manifest.metadata
    config_check = _validate_requested_config(
        path=path,
        metadata=metadata,
        model_config=model_config,
        train_config=train_config,
    )
    if isinstance(config_check, _result.Rejected):
        return config_check
    state_path_result = _validated_manifest_state_path(
        manifest_path=path,
        manifest=manifest,
    )
    if isinstance(state_path_result, _result.Rejected):
        return state_path_result
    state_path = state_path_result.value
    state_sha256_result = sha256_checkpoint_file(state_path)
    if isinstance(state_sha256_result, _result.Rejected):
        return state_sha256_result
    if state_sha256_result.value != manifest.state_sha256:
        return checkpoint_corruption(
            state_path,
            f"state sha256 does not match manifest {path}",
        )
    payload_result = read_checkpoint_payload(state_path)
    if isinstance(payload_result, _result.Rejected):
        return payload_result
    payload = payload_result.value
    if payload.checkpoint_id != manifest.checkpoint_id:
        return checkpoint_corruption(
            state_path,
            f"state checkpoint id does not match manifest {path}",
        )
    model = _create_checkpoint_model_without_rng_side_effect(
        model_config=model_config,
        device=device,
    )
    try:
        model.load_state_dict(payload.model_state)
    except RuntimeError:
        return checkpoint_corruption(
            state_path, "model state does not match model config"
        )
    trainer = PPOTrainer(
        model=model,
        train_config=train_config,
        device=device,
        profile_mode=execution_config.ppo_profile,
    )
    optimizer_check = validate_optimizer_state_payload(
        state=payload.optimizer_state,
        parameters=tuple(model.parameters()),
        path=state_path,
    )
    if isinstance(optimizer_check, _result.Rejected):
        return optimizer_check
    trainer.load_optimizer_state(payload.optimizer_state)
    return _result.Ok(
        value=_training_state.LoadedTrainingState(
            model=model,
            trainer=trainer,
            total_rounds=metadata.total_rounds,
            total_samples=metadata.total_samples,
            total_updates=metadata.total_updates,
        )
    )


def read_torch_checkpoint_metadata(
    path: Path,
) -> _result.Ok[TorchCheckpointMetadata] | _result.Rejected:
    """Read checkpoint metadata without binding to a training device."""
    manifest_result = read_checkpoint_manifest(path)
    if isinstance(manifest_result, _result.Rejected):
        return manifest_result
    return _result.Ok(value=manifest_result.value.metadata)


def _validated_manifest_state_path(
    *,
    manifest_path: Path,
    manifest: CheckpointManifest,
) -> _result.Ok[Path] | _result.Rejected:
    checkpoint_dir = manifest_path.parent
    checkpoint_dir_check = validate_checkpoint_dir(checkpoint_dir)
    if isinstance(checkpoint_dir_check, _result.Rejected):
        return checkpoint_dir_check
    if not checkpoint_dir_check.value:
        return checkpoint_corruption(
            checkpoint_dir, "checkpoint directory is missing"
        )
    objects_dir = checkpoint_dir / CHECKPOINT_OBJECTS_DIR
    objects_dir_check = validate_checkpoint_objects_dir(objects_dir)
    if isinstance(objects_dir_check, _result.Rejected):
        return objects_dir_check
    if not objects_dir_check.value:
        return checkpoint_corruption(
            objects_dir, "checkpoint objects directory is missing"
        )
    state_path = manifest_state_file_path(
        manifest_path=manifest_path,
        manifest=manifest,
    )
    object_dir_check = validate_checkpoint_object_dir(state_path.parent)
    if isinstance(object_dir_check, _result.Rejected):
        return object_dir_check
    if not object_dir_check.value:
        return checkpoint_corruption(
            state_path.parent, "checkpoint object is missing"
        )
    state_file_check = validate_checkpoint_state_file(state_path)
    if isinstance(state_file_check, _result.Rejected):
        return state_file_check
    return _result.Ok(value=state_path)


def _create_checkpoint_model_without_rng_side_effect(
    *,
    model_config: ModelConfig,
    device: torch.device,
) -> TractorPolicyModel:
    cpu_rng_state = torch.random.get_rng_state()
    try:
        return _training_state.create_model(model_config, device)
    finally:
        torch.random.set_rng_state(cpu_rng_state)


def _validate_requested_config(
    *,
    path: Path,
    metadata: TorchCheckpointMetadata,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> _result.Ok[None] | _result.Rejected:
    if metadata.model_config != model_config:
        return _checkpoint_mismatch(
            path, "model_config does not match checkpoint metadata"
        )
    if metadata.train_config.seed != train_config.seed:
        return _checkpoint_mismatch(
            path, "train_config.seed does not match checkpoint metadata"
        )
    return _result.Ok(value=None)


def _checkpoint_mismatch(path: Path, reason: str) -> _result.Rejected:
    return _result.Rejected(
        reason=f"checkpoint mismatch: {path}: {reason}"
    )
