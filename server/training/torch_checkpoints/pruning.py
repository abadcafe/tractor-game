"""Retention pruning for torch training checkpoints."""

from __future__ import annotations

import shutil
from pathlib import Path

from server.foundation import result as _result
from server.training.torch_checkpoints.filesystem import (
    validate_checkpoint_dir,
    validate_checkpoint_object_dir,
    validate_checkpoint_objects,
    validate_checkpoint_objects_dir,
)
from server.training.torch_checkpoints.manifest import (
    managed_checkpoint_manifest_paths,
    managed_update_number_from_manifest_path,
    read_checkpoint_manifest,
    update_checkpoint_manifest_paths,
)
from server.training.torch_checkpoints.schema import (
    CHECKPOINT_OBJECTS_DIR,
    checkpoint_corruption,
)


def preflight_managed_checkpoints(
    checkpoint_dir: Path,
) -> _result.Ok[None] | _result.Rejected:
    """Validate managed manifests before committing a new checkpoint."""
    dir_check = validate_checkpoint_dir(checkpoint_dir)
    if isinstance(dir_check, _result.Rejected):
        return dir_check
    if not dir_check.value:
        return _result.Ok(value=None)
    object_check = validate_checkpoint_objects(checkpoint_dir)
    if isinstance(object_check, _result.Rejected):
        return object_check
    live_result = _live_checkpoint_ids(checkpoint_dir)
    if isinstance(live_result, _result.Rejected):
        return live_result
    return _result.Ok(value=None)


def preflight_torch_checkpoint_pruning(
    *,
    checkpoint_dir: Path,
    retained_update_count: int,
    pending_manifest_paths: tuple[Path, ...],
    pending_checkpoint_id: str,
) -> _result.Ok[None] | _result.Rejected:
    """Validate post-commit pruning before manifests are committed."""
    assert retained_update_count >= 0
    dir_check = validate_checkpoint_dir(checkpoint_dir)
    if isinstance(dir_check, _result.Rejected):
        return dir_check
    if not dir_check.value:
        return _result.Ok(value=None)
    expired_paths_result = _future_expired_update_manifest_paths(
        checkpoint_dir=checkpoint_dir,
        retained_update_count=retained_update_count,
        pending_manifest_paths=pending_manifest_paths,
    )
    if isinstance(expired_paths_result, _result.Rejected):
        return expired_paths_result
    expired_paths = expired_paths_result.value
    expired_path_check = _preflight_expired_update_paths(expired_paths)
    if isinstance(expired_path_check, _result.Rejected):
        return expired_path_check
    live_result = _future_live_checkpoint_ids(
        checkpoint_dir=checkpoint_dir,
        pending_manifest_paths=pending_manifest_paths,
        pending_checkpoint_id=pending_checkpoint_id,
        expired_update_paths=expired_paths,
    )
    if isinstance(live_result, _result.Rejected):
        return live_result
    return validate_checkpoint_objects(checkpoint_dir)


def prune_torch_checkpoints(
    *,
    checkpoint_dir: Path,
    retained_update_count: int,
) -> _result.Ok[None] | _result.Rejected:
    """Delete expired manifests and unreferenced state objects."""
    assert retained_update_count >= 0
    dir_check = validate_checkpoint_dir(checkpoint_dir)
    if isinstance(dir_check, _result.Rejected):
        return dir_check
    if not dir_check.value:
        return _result.Ok(value=None)
    object_check = validate_checkpoint_objects(checkpoint_dir)
    if isinstance(object_check, _result.Rejected):
        return object_check
    prune_result = _prune_update_manifests(
        checkpoint_dir=checkpoint_dir,
        retained_update_count=retained_update_count,
    )
    if isinstance(prune_result, _result.Rejected):
        return prune_result
    return _remove_unreferenced_checkpoint_objects(checkpoint_dir)


def _future_expired_update_manifest_paths(
    *,
    checkpoint_dir: Path,
    retained_update_count: int,
    pending_manifest_paths: tuple[Path, ...],
) -> _result.Ok[tuple[Path, ...]] | _result.Rejected:
    try:
        update_paths = dict(
            update_checkpoint_manifest_paths(checkpoint_dir)
        )
    except OSError:
        return checkpoint_corruption(
            checkpoint_dir, "update manifests are not readable"
        )
    for path in pending_manifest_paths:
        update_number = managed_update_number_from_manifest_path(path)
        if update_number is not None:
            update_paths[update_number] = path
    sorted_paths = tuple(
        path for _, path in sorted(update_paths.items())
    )
    if retained_update_count == 0:
        return _result.Ok(value=sorted_paths)
    return _result.Ok(value=sorted_paths[:-retained_update_count])


def _preflight_expired_update_paths(
    expired_paths: tuple[Path, ...],
) -> _result.Ok[None] | _result.Rejected:
    for path in expired_paths:
        try:
            path_exists = path.exists()
        except OSError:
            return checkpoint_corruption(
                path, "expired update manifest is not readable"
            )
        if not path_exists:
            continue
        if path.is_dir():
            return checkpoint_corruption(
                path, "expired update manifest cannot be deleted"
            )
    return _result.Ok(value=None)


def _future_live_checkpoint_ids(
    *,
    checkpoint_dir: Path,
    pending_manifest_paths: tuple[Path, ...],
    pending_checkpoint_id: str,
    expired_update_paths: tuple[Path, ...],
) -> _result.Ok[set[str]] | _result.Rejected:
    try:
        current_paths = managed_checkpoint_manifest_paths(
            checkpoint_dir
        )
    except OSError:
        return checkpoint_corruption(
            checkpoint_dir,
            "managed checkpoint manifests are not readable",
        )
    pending_path_set = set(pending_manifest_paths)
    expired_path_set = set(expired_update_paths)
    future_paths = (
        set(current_paths) | pending_path_set
    ) - expired_path_set
    checkpoint_ids: set[str] = set()
    for path in future_paths:
        if path in pending_path_set:
            checkpoint_ids.add(pending_checkpoint_id)
            continue
        manifest_result = read_checkpoint_manifest(path)
        if isinstance(manifest_result, _result.Rejected):
            return manifest_result
        checkpoint_ids.add(manifest_result.value.checkpoint_id)
    return _result.Ok(value=checkpoint_ids)


def _prune_update_manifests(
    *,
    checkpoint_dir: Path,
    retained_update_count: int,
) -> _result.Ok[None] | _result.Rejected:
    try:
        update_paths = update_checkpoint_manifest_paths(checkpoint_dir)
    except OSError:
        return checkpoint_corruption(
            checkpoint_dir, "update manifests are not readable"
        )
    if retained_update_count == 0:
        expired_paths = tuple(path for _, path in update_paths)
    else:
        expired_paths = tuple(
            path for _, path in update_paths[:-retained_update_count]
        )
    for path in expired_paths:
        try:
            path.unlink()
        except OSError:
            return checkpoint_corruption(
                path, "expired update manifest cannot be deleted"
            )
    return _result.Ok(value=None)


def _remove_unreferenced_checkpoint_objects(
    checkpoint_dir: Path,
) -> _result.Ok[None] | _result.Rejected:
    objects_dir = checkpoint_dir / CHECKPOINT_OBJECTS_DIR
    objects_dir_check = validate_checkpoint_objects_dir(objects_dir)
    if isinstance(objects_dir_check, _result.Rejected):
        return objects_dir_check
    if not objects_dir_check.value:
        return _result.Ok(value=None)
    live_result = _live_checkpoint_ids(checkpoint_dir)
    if isinstance(live_result, _result.Rejected):
        return live_result
    live_checkpoint_ids = live_result.value
    try:
        object_paths = tuple(objects_dir.iterdir())
    except OSError:
        return checkpoint_corruption(
            objects_dir, "checkpoint objects directory is not readable"
        )
    for child in object_paths:
        object_check = validate_checkpoint_object_dir(child)
        if isinstance(object_check, _result.Rejected):
            return object_check
        if child.name in live_checkpoint_ids:
            continue
        try:
            shutil.rmtree(child)
        except OSError:
            return checkpoint_corruption(
                child, "checkpoint object cannot be deleted"
            )
    return _result.Ok(value=None)


def _live_checkpoint_ids(
    checkpoint_dir: Path,
) -> _result.Ok[set[str]] | _result.Rejected:
    checkpoint_ids: set[str] = set()
    try:
        managed_paths = managed_checkpoint_manifest_paths(
            checkpoint_dir
        )
    except OSError:
        return checkpoint_corruption(
            checkpoint_dir,
            "managed checkpoint manifests are not readable",
        )
    for path in managed_paths:
        manifest_result = read_checkpoint_manifest(path)
        if isinstance(manifest_result, _result.Rejected):
            return manifest_result
        checkpoint_ids.add(manifest_result.value.checkpoint_id)
    return _result.Ok(value=checkpoint_ids)
