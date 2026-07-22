"""Torch checkpoint save transaction."""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from server.foundation import result as _result
from server.training.config import TrainConfig
from server.training.model import (
    ModelConfig,
)
from server.training.model import (
    TractorPolicyModel as _TractorPolicyModel,
)
from server.training.ppo import PPOTrainer
from server.training.torch_checkpoints.manifest import (
    checkpoint_dir_from_manifest_paths,
    managed_update_number_from_manifest_path,
    write_checkpoint_manifest,
)
from server.training.torch_checkpoints.payload import (
    write_checkpoint_payload,
)
from server.training.torch_checkpoints.pruning import (
    preflight_managed_checkpoints,
    preflight_torch_checkpoint_pruning,
    prune_torch_checkpoints,
)
from server.training.torch_checkpoints.schema import (
    CHECKPOINT_OBJECTS_DIR,
    CHECKPOINT_STATE_FILENAME,
    CheckpointManifest,
    TorchCheckpointMetadata,
    checkpoint_corruption,
    sha256_file,
)


@dataclass(frozen=True, slots=True)
class _ManifestBackup:
    path: Path
    content: bytes | None


@dataclass(frozen=True, slots=True)
class TorchCheckpointSaveResult:
    """Committed checkpoint save result.

    Post-commit pruning failures do not undo the saved checkpoint.
    """

    post_commit_prune_failure: _result.Rejected | None


def save_torch_checkpoint(
    *,
    manifest_paths: tuple[Path, ...],
    model: _TractorPolicyModel,
    trainer: PPOTrainer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    total_rounds: int,
    total_samples: int,
    total_updates: int,
    retained_update_count: int,
) -> _result.Ok[TorchCheckpointSaveResult] | _result.Rejected:
    """Save one state object and write each manifest atomically."""
    assert retained_update_count >= 0
    assert total_rounds >= 0
    assert total_samples >= 0
    assert total_updates >= 0
    checkpoint_dir_result = checkpoint_dir_from_manifest_paths(
        manifest_paths
    )
    if isinstance(checkpoint_dir_result, _result.Rejected):
        return checkpoint_dir_result
    update_paths_result = _validate_update_manifest_paths(
        manifest_paths=manifest_paths,
        total_updates=total_updates,
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
    manifest_backups = backup_result.value
    checkpoint_id = uuid.uuid4().hex
    relative_state_path = (
        Path(CHECKPOINT_OBJECTS_DIR)
        / checkpoint_id
        / CHECKPOINT_STATE_FILENAME
    )
    state_path = checkpoint_dir / relative_state_path
    try:
        state_path.parent.mkdir(parents=True, exist_ok=False)
    except OSError:
        return checkpoint_corruption(
            state_path.parent,
            "state object directory cannot be created",
        )
    payload_result = write_checkpoint_payload(
        path=state_path,
        checkpoint_id=checkpoint_id,
        model=model,
        trainer=trainer,
    )
    if isinstance(payload_result, _result.Rejected):
        _discard_state_object(state_path)
        return payload_result
    try:
        state_sha256 = sha256_file(state_path)
    except OSError:
        _discard_state_object(state_path)
        return checkpoint_corruption(
            state_path, "state payload is not readable after write"
        )
    manifest = CheckpointManifest(
        checkpoint_id=checkpoint_id,
        state_path=relative_state_path,
        state_sha256=state_sha256,
        metadata=TorchCheckpointMetadata(
            model_config=model_config,
            train_config=train_config,
            total_rounds=total_rounds,
            total_samples=total_samples,
            total_updates=total_updates,
        ),
    )
    pruning_preflight_result = preflight_torch_checkpoint_pruning(
        checkpoint_dir=checkpoint_dir,
        retained_update_count=retained_update_count,
        pending_manifest_paths=manifest_paths,
        pending_checkpoint_id=checkpoint_id,
    )
    if isinstance(pruning_preflight_result, _result.Rejected):
        _discard_state_object(state_path)
        return pruning_preflight_result
    for manifest_path in manifest_paths:
        manifest_result = write_checkpoint_manifest(
            path=manifest_path, manifest=manifest
        )
        if isinstance(manifest_result, _result.Rejected):
            _restore_manifest_backups(manifest_backups)
            _discard_state_object(state_path)
            return manifest_result
    prune_result = prune_torch_checkpoints(
        checkpoint_dir=checkpoint_dir,
        retained_update_count=retained_update_count,
    )
    if isinstance(prune_result, _result.Rejected):
        return _result.Ok(
            value=TorchCheckpointSaveResult(
                post_commit_prune_failure=prune_result
            )
        )
    return _result.Ok(
        value=TorchCheckpointSaveResult(post_commit_prune_failure=None)
    )


def _validate_update_manifest_paths(
    *,
    manifest_paths: tuple[Path, ...],
    total_updates: int,
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
    return _result.Ok(value=None)


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
    return _result.Ok(value=tuple(backups))


def _restore_manifest_backups(
    backups: tuple[_ManifestBackup, ...],
) -> None:
    for backup in backups:
        if backup.content is None:
            _discard_file(backup.path)
            continue
        tmp_path = backup.path.with_suffix(
            f"{backup.path.suffix}.rollback"
        )
        try:
            tmp_path.write_bytes(backup.content)
            os.replace(tmp_path, backup.path)
        except OSError:
            _discard_file(tmp_path)


def _discard_state_object(state_path: Path) -> None:
    try:
        shutil.rmtree(state_path.parent)
    except FileNotFoundError:
        return
    except OSError:
        return


def _discard_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return
