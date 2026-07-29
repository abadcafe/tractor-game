"""Verified policy checkpoint loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from server.checkpoint_contract import CHECKPOINT_OBJECTS_DIR
from server.foundation import result as _result
from server.policy_model.checkpoint.filesystem import (
    validate_checkpoint_dir,
    validate_checkpoint_object_dir,
    validate_checkpoint_objects_dir,
    validate_checkpoint_state_file,
)
from server.policy_model.checkpoint.manifest import (
    manifest_state_file_path,
    read_checkpoint_manifest,
)
from server.policy_model.checkpoint.payload import (
    CheckpointPayload,
    read_checkpoint_payload,
)
from server.policy_model.checkpoint.schema import (
    CheckpointManifest,
    CheckpointMetadata,
    checkpoint_corruption,
    sha256_checkpoint_file,
)
from server.policy_model.network import ModelConfig, PolicyActionModel


@dataclass(frozen=True, slots=True)
class LoadedPolicyCheckpoint:
    """Current model state restored for policy inference."""

    model: PolicyActionModel
    model_config: ModelConfig
    metadata: CheckpointMetadata


def load_policy_checkpoint(
    *,
    path: Path,
    device: torch.device,
) -> _result.Ok[LoadedPolicyCheckpoint] | _result.Rejected:
    """Restore the model declared by the current checkpoint contract."""
    checkpoint_result = read_checkpoint_snapshot(path)
    if isinstance(checkpoint_result, _result.Rejected):
        return checkpoint_result
    checkpoint = checkpoint_result.value
    metadata = checkpoint.manifest.metadata
    model_result = restore_policy_model(
        checkpoint=checkpoint,
        device=device,
    )
    if isinstance(model_result, _result.Rejected):
        return model_result
    model = model_result.value
    model.eval()
    return _result.Ok(
        value=LoadedPolicyCheckpoint(
            model=model,
            model_config=metadata.model_config,
            metadata=metadata,
        )
    )


def read_checkpoint_metadata(
    path: Path,
) -> _result.Ok[CheckpointMetadata] | _result.Rejected:
    """Read checkpoint metadata without creating a model."""
    manifest_result = read_checkpoint_manifest(path)
    if isinstance(manifest_result, _result.Rejected):
        return manifest_result
    return _result.Ok(value=manifest_result.value.metadata)


@dataclass(frozen=True, slots=True)
class CheckpointSnapshot:
    """One manifest and its verified immutable state payload."""

    manifest: CheckpointManifest
    state_path: Path
    payload: CheckpointPayload


def read_checkpoint_snapshot(
    path: Path,
) -> _result.Ok[CheckpointSnapshot] | _result.Rejected:
    """Read and cryptographically bind a manifest to its payload."""
    manifest_result = read_checkpoint_manifest(path)
    if isinstance(manifest_result, _result.Rejected):
        return manifest_result
    manifest = manifest_result.value
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
    return _result.Ok(
        value=CheckpointSnapshot(
            manifest=manifest,
            state_path=state_path,
            payload=payload,
        )
    )


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


def restore_policy_model(
    *,
    checkpoint: CheckpointSnapshot,
    device: torch.device,
) -> _result.Ok[PolicyActionModel] | _result.Rejected:
    """Restore a policy model without changing the caller's CPU RNG."""
    model_config = checkpoint.manifest.metadata.model_config
    cpu_rng_state = torch.random.get_rng_state()
    try:
        model = PolicyActionModel(
            d_model=model_config.d_model,
            layers=model_config.layers,
            heads=model_config.heads,
            action_value_layers=model_config.action_value_layers,
        )
        model.to(device)
    finally:
        torch.random.set_rng_state(cpu_rng_state)
    try:
        model.load_state_dict(checkpoint.payload.model_state)
    except RuntimeError:
        return checkpoint_corruption(
            checkpoint.state_path,
            "model state does not match model config",
        )
    return _result.Ok(value=model)
