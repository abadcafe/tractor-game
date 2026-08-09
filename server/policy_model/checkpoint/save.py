"""Atomic schema-27 checkpoint save transaction."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from torch import Tensor

import server.policy_model.checkpoint.manifest as checkpoint_manifest
from server.checkpoint_contract import (
    CHECKPOINT_OBJECTS_DIR,
    CHECKPOINT_POLICY_FILENAME,
    CHECKPOINT_TRAINER_FILENAME,
)
from server.foundation import result as _result
from server.policy_model.checkpoint.manifest import (
    checkpoint_dir_from_manifest_paths,
    managed_update_number_from_manifest_path,
)
from server.policy_model.checkpoint.payload import (
    write_policy_payload,
    write_trainer_payload,
)
from server.policy_model.checkpoint.pruning import (
    preflight_checkpoint_pruning,
    preflight_managed_checkpoints,
    prune_checkpoints,
)
from server.policy_model.checkpoint.schema import (
    CheckpointManifest,
    CheckpointMetadata,
    checkpoint_corruption,
    sha256_checkpoint_file,
)


@dataclass(frozen=True, slots=True)
class _ManifestBackup:
    path: Path
    content: bytes | None


@dataclass(frozen=True, slots=True)
class CheckpointSaveResult:
    """Committed checkpoint save result."""

    post_commit_prune_failure: _result.Rejected | None


def save_checkpoint(
    *,
    manifest_paths: tuple[Path, ...],
    policy_state: Mapping[str, Tensor],
    value_state: Mapping[str, Tensor],
    optimizer_state: dict[str, object],
    metadata: CheckpointMetadata,
    retained_update_count: int,
) -> _result.Ok[CheckpointSaveResult] | _result.Rejected:
    """Commit one immutable policy/trainer object and its manifests."""
    assert retained_update_count >= 0
    checkpoint_dir_result = checkpoint_dir_from_manifest_paths(
        manifest_paths
    )
    if isinstance(checkpoint_dir_result, _result.Rejected):
        return checkpoint_dir_result
    update_paths_result = _validate_update_manifest_paths(
        manifest_paths=manifest_paths,
        total_updates=metadata.total_updates,
    )
    if isinstance(update_paths_result, _result.Rejected):
        return update_paths_result
    checkpoint_dir = checkpoint_dir_result.value
    preflight_result = preflight_managed_checkpoints(checkpoint_dir)
    if isinstance(preflight_result, _result.Rejected):
        return preflight_result
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return checkpoint_corruption(
            checkpoint_dir, "checkpoint directory is not writable"
        )
    backup_result = _capture_manifest_backups(manifest_paths)
    if isinstance(backup_result, _result.Rejected):
        return backup_result
    checkpoint_id = uuid.uuid4().hex
    relative_object_dir = Path(CHECKPOINT_OBJECTS_DIR) / checkpoint_id
    object_dir = checkpoint_dir / relative_object_dir
    policy_path = object_dir / CHECKPOINT_POLICY_FILENAME
    trainer_path = object_dir / CHECKPOINT_TRAINER_FILENAME
    try:
        object_dir.mkdir(parents=True, exist_ok=False)
    except OSError:
        return checkpoint_corruption(
            object_dir, "checkpoint object directory cannot be created"
        )
    policy_write = write_policy_payload(
        path=policy_path,
        checkpoint_id=checkpoint_id,
        policy_state=policy_state,
    )
    if isinstance(policy_write, _result.Rejected):
        _discard_object(object_dir)
        return policy_write
    trainer_write = write_trainer_payload(
        path=trainer_path,
        checkpoint_id=checkpoint_id,
        value_state=value_state,
        optimizer_state=optimizer_state,
    )
    if isinstance(trainer_write, _result.Rejected):
        _discard_object(object_dir)
        return trainer_write
    policy_digest = sha256_checkpoint_file(policy_path)
    if isinstance(policy_digest, _result.Rejected):
        _discard_object(object_dir)
        return policy_digest
    trainer_digest = sha256_checkpoint_file(trainer_path)
    if isinstance(trainer_digest, _result.Rejected):
        _discard_object(object_dir)
        return trainer_digest
    manifest = CheckpointManifest(
        checkpoint_id=checkpoint_id,
        policy_path=(relative_object_dir / CHECKPOINT_POLICY_FILENAME),
        policy_sha256=policy_digest.value,
        trainer_path=(
            relative_object_dir / CHECKPOINT_TRAINER_FILENAME
        ),
        trainer_sha256=trainer_digest.value,
        metadata=metadata,
    )
    pruning_preflight = preflight_checkpoint_pruning(
        checkpoint_dir=checkpoint_dir,
        retained_update_count=retained_update_count,
        pending_manifest_paths=manifest_paths,
        pending_checkpoint_id=checkpoint_id,
    )
    if isinstance(pruning_preflight, _result.Rejected):
        _discard_object(object_dir)
        return pruning_preflight
    for manifest_path in manifest_paths:
        manifest_result = checkpoint_manifest.write_checkpoint_manifest(
            path=manifest_path, manifest=manifest
        )
        if isinstance(manifest_result, _result.Rejected):
            _restore_manifest_backups(backup_result.value)
            _discard_object(object_dir)
            return manifest_result
    prune_result = prune_checkpoints(
        checkpoint_dir=checkpoint_dir,
        retained_update_count=retained_update_count,
    )
    if isinstance(prune_result, _result.Rejected):
        return _result.Ok(
            CheckpointSaveResult(post_commit_prune_failure=prune_result)
        )
    return _result.Ok(
        CheckpointSaveResult(post_commit_prune_failure=None)
    )


def _validate_update_manifest_paths(
    *, manifest_paths: tuple[Path, ...], total_updates: int
) -> _result.Ok[None] | _result.Rejected:
    for path in manifest_paths:
        update_number = managed_update_number_from_manifest_path(path)
        if update_number is None:
            continue
        if update_number != total_updates:
            return checkpoint_corruption(
                path,
                "update manifest number must equal total_updates",
            )
    return _result.Ok(None)


def _capture_manifest_backups(
    manifest_paths: tuple[Path, ...],
) -> _result.Ok[tuple[_ManifestBackup, ...]] | _result.Rejected:
    backups: list[_ManifestBackup] = []
    for path in manifest_paths:
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            content = None
        except OSError:
            return checkpoint_corruption(
                path,
                "manifest file is not readable before checkpoint write",
            )
        backups.append(_ManifestBackup(path=path, content=content))
    return _result.Ok(tuple(backups))


def _restore_manifest_backups(
    backups: tuple[_ManifestBackup, ...],
) -> None:
    for backup in backups:
        if backup.content is None:
            _discard_file(backup.path)
            continue
        temporary = backup.path.with_suffix(
            f"{backup.path.suffix}.rollback"
        )
        try:
            _ = temporary.write_bytes(backup.content)
            os.replace(temporary, backup.path)
        except OSError:
            _discard_file(temporary)


def _discard_object(object_dir: Path) -> None:
    try:
        shutil.rmtree(object_dir)
    except FileNotFoundError, OSError:
        return


def _discard_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError, OSError:
        return
