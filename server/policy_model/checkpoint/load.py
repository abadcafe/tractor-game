"""Strict schema-27 checkpoint loading."""

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
    validate_checkpoint_payload_file,
)
from server.policy_model.checkpoint.manifest import (
    read_checkpoint_manifest,
)
from server.policy_model.checkpoint.payload import (
    PolicyPayload,
    TrainerPayload,
    read_policy_payload,
    read_trainer_payload,
)
from server.policy_model.checkpoint.schema import (
    CheckpointManifest,
    CheckpointMetadata,
    checkpoint_corruption,
    sha256_checkpoint_file,
)
from server.policy_model.network import ModelConfig, PolicyModel


@dataclass(frozen=True, slots=True)
class LoadedPolicyCheckpoint:
    """Inference actor restored without reading trainer.pt."""

    model: PolicyModel
    model_config: ModelConfig
    metadata: CheckpointMetadata


@dataclass(frozen=True, slots=True)
class CheckpointSnapshot:
    """Manifest plus both verified schema-27 payloads."""

    manifest: CheckpointManifest
    policy_path: Path
    trainer_path: Path
    policy: PolicyPayload
    trainer: TrainerPayload


def load_policy_checkpoint(
    *, path: Path, device: torch.device
) -> _result.Ok[LoadedPolicyCheckpoint] | _result.Rejected:
    """Load only policy.pt for inference."""
    manifest_result = read_checkpoint_manifest(path)
    if isinstance(manifest_result, _result.Rejected):
        return manifest_result
    manifest = manifest_result.value
    policy_result = _read_verified_policy(
        manifest_path=path,
        manifest=manifest,
    )
    if isinstance(policy_result, _result.Rejected):
        return policy_result
    policy_path, payload = policy_result.value
    model_result = _restore_policy_state(
        policy_state=payload.policy_state,
        metadata=manifest.metadata,
        policy_path=policy_path,
        device=device,
    )
    if isinstance(model_result, _result.Rejected):
        return model_result
    _ = model_result.value.eval()
    return _result.Ok(
        LoadedPolicyCheckpoint(
            model=model_result.value,
            model_config=manifest.metadata.model_config,
            metadata=manifest.metadata,
        )
    )


def read_checkpoint_metadata(
    path: Path,
) -> _result.Ok[CheckpointMetadata] | _result.Rejected:
    manifest = read_checkpoint_manifest(path)
    if isinstance(manifest, _result.Rejected):
        return manifest
    return _result.Ok(manifest.value.metadata)


def read_checkpoint_snapshot(
    path: Path,
) -> _result.Ok[CheckpointSnapshot] | _result.Rejected:
    """Read and verify both payloads for training resume."""
    manifest_result = read_checkpoint_manifest(path)
    if isinstance(manifest_result, _result.Rejected):
        return manifest_result
    manifest = manifest_result.value
    policy_result = _read_verified_policy(
        manifest_path=path,
        manifest=manifest,
    )
    if isinstance(policy_result, _result.Rejected):
        return policy_result
    trainer_result = _read_verified_trainer(
        manifest_path=path,
        manifest=manifest,
    )
    if isinstance(trainer_result, _result.Rejected):
        return trainer_result
    policy_path, policy = policy_result.value
    trainer_path, trainer = trainer_result.value
    if policy.checkpoint_id != trainer.checkpoint_id:
        return checkpoint_corruption(
            trainer_path, "policy and trainer checkpoint ids differ"
        )
    return _result.Ok(
        CheckpointSnapshot(
            manifest=manifest,
            policy_path=policy_path,
            trainer_path=trainer_path,
            policy=policy,
            trainer=trainer,
        )
    )


def restore_policy_model(
    *, checkpoint: CheckpointSnapshot, device: torch.device
) -> _result.Ok[PolicyModel] | _result.Rejected:
    """Restore the actor from a verified training snapshot."""
    return _restore_policy_state(
        policy_state=checkpoint.policy.policy_state,
        metadata=checkpoint.manifest.metadata,
        policy_path=checkpoint.policy_path,
        device=device,
    )


def _restore_policy_state(
    *,
    policy_state: dict[str, torch.Tensor],
    metadata: CheckpointMetadata,
    policy_path: Path,
    device: torch.device,
) -> _result.Ok[PolicyModel] | _result.Rejected:
    cpu_rng_state = torch.random.get_rng_state()
    try:
        model = PolicyModel(config=metadata.model_config).to(device)
    finally:
        torch.random.set_rng_state(cpu_rng_state)
    try:
        _ = model.load_state_dict(policy_state)
    except RuntimeError:
        return checkpoint_corruption(
            policy_path, "policy state does not match model config"
        )
    return _result.Ok(model)


def _read_verified_policy(
    *, manifest_path: Path, manifest: CheckpointManifest
) -> _result.Ok[tuple[Path, PolicyPayload]] | _result.Rejected:
    path_result = _validated_payload_path(
        manifest_path=manifest_path,
        relative_path=manifest.policy_path,
    )
    if isinstance(path_result, _result.Rejected):
        return path_result
    path = path_result.value
    digest = _verify_digest(path, manifest.policy_sha256)
    if isinstance(digest, _result.Rejected):
        return digest
    payload = read_policy_payload(path)
    if isinstance(payload, _result.Rejected):
        return payload
    if payload.value.checkpoint_id != manifest.checkpoint_id:
        return checkpoint_corruption(
            path, "policy checkpoint id does not match manifest"
        )
    return _result.Ok((path, payload.value))


def _read_verified_trainer(
    *, manifest_path: Path, manifest: CheckpointManifest
) -> _result.Ok[tuple[Path, TrainerPayload]] | _result.Rejected:
    path_result = _validated_payload_path(
        manifest_path=manifest_path,
        relative_path=manifest.trainer_path,
    )
    if isinstance(path_result, _result.Rejected):
        return path_result
    path = path_result.value
    digest = _verify_digest(path, manifest.trainer_sha256)
    if isinstance(digest, _result.Rejected):
        return digest
    payload = read_trainer_payload(path)
    if isinstance(payload, _result.Rejected):
        return payload
    if payload.value.checkpoint_id != manifest.checkpoint_id:
        return checkpoint_corruption(
            path, "trainer checkpoint id does not match manifest"
        )
    return _result.Ok((path, payload.value))


def _verify_digest(
    path: Path, expected: str
) -> _result.Ok[None] | _result.Rejected:
    digest = sha256_checkpoint_file(path)
    if isinstance(digest, _result.Rejected):
        return digest
    if digest.value != expected:
        return checkpoint_corruption(
            path, "sha256 does not match manifest"
        )
    return _result.Ok(None)


def _validated_payload_path(
    *, manifest_path: Path, relative_path: Path
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
    objects_check = validate_checkpoint_objects_dir(objects_dir)
    if isinstance(objects_check, _result.Rejected):
        return objects_check
    if not objects_check.value:
        return checkpoint_corruption(
            objects_dir, "checkpoint objects directory is missing"
        )
    payload_path = checkpoint_dir / relative_path
    object_check = validate_checkpoint_object_dir(payload_path.parent)
    if isinstance(object_check, _result.Rejected):
        return object_check
    if not object_check.value:
        return checkpoint_corruption(
            payload_path.parent, "checkpoint object is missing"
        )
    payload_check = validate_checkpoint_payload_file(payload_path)
    if isinstance(payload_check, _result.Rejected):
        return payload_check
    return _result.Ok(payload_path)


__all__ = (
    "CheckpointSnapshot",
    "LoadedPolicyCheckpoint",
    "load_policy_checkpoint",
    "read_checkpoint_metadata",
    "read_checkpoint_snapshot",
    "restore_policy_model",
)
