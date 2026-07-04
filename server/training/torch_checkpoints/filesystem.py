"""Filesystem boundary validation for torch checkpoint archives."""

from __future__ import annotations

from pathlib import Path

from server import result as _result
from server.training.torch_checkpoints.schema import (
    CHECKPOINT_OBJECTS_DIR,
    checkpoint_corruption,
)


def validate_checkpoint_dir(
    checkpoint_dir: Path,
) -> _result.Ok[bool] | _result.Rejected:
    """Validate the checkpoint directory without following symlinks."""
    try:
        checkpoint_is_symlink = checkpoint_dir.is_symlink()
    except OSError:
        return checkpoint_corruption(
            checkpoint_dir, "checkpoint directory is not readable"
        )
    if checkpoint_is_symlink:
        return checkpoint_corruption(
            checkpoint_dir, "checkpoint path is a symlink"
        )
    try:
        checkpoint_exists = checkpoint_dir.exists()
    except OSError:
        return checkpoint_corruption(
            checkpoint_dir, "checkpoint directory is not readable"
        )
    if not checkpoint_exists:
        return _result.Ok(value=False)
    try:
        checkpoint_is_dir = checkpoint_dir.is_dir()
    except OSError:
        return checkpoint_corruption(
            checkpoint_dir, "checkpoint directory is not readable"
        )
    if not checkpoint_is_dir:
        return checkpoint_corruption(
            checkpoint_dir, "checkpoint path is not a directory"
        )
    return _result.Ok(value=True)


def validate_checkpoint_manifest_file(
    manifest_path: Path,
) -> _result.Ok[None] | _result.Rejected:
    """Validate one checkpoint manifest path before reading it."""
    checkpoint_dir_check = validate_checkpoint_dir(manifest_path.parent)
    if isinstance(checkpoint_dir_check, _result.Rejected):
        return checkpoint_dir_check
    try:
        manifest_is_symlink = manifest_path.is_symlink()
    except OSError:
        return checkpoint_corruption(
            manifest_path, "manifest file is not readable"
        )
    if manifest_is_symlink:
        return checkpoint_corruption(
            manifest_path, "manifest file is a symlink"
        )
    return _result.Ok(value=None)


def validate_checkpoint_objects(
    checkpoint_dir: Path,
) -> _result.Ok[None] | _result.Rejected:
    """Validate the checkpoint objects tree."""
    objects_dir = checkpoint_dir / CHECKPOINT_OBJECTS_DIR
    objects_dir_check = validate_checkpoint_objects_dir(objects_dir)
    if isinstance(objects_dir_check, _result.Rejected):
        return objects_dir_check
    if not objects_dir_check.value:
        return _result.Ok(value=None)
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
    return _result.Ok(value=None)


def validate_checkpoint_objects_dir(
    objects_dir: Path,
) -> _result.Ok[bool] | _result.Rejected:
    """Validate the objects directory without following symlinks."""
    try:
        objects_is_symlink = objects_dir.is_symlink()
    except OSError:
        return checkpoint_corruption(
            objects_dir, "checkpoint objects directory is not readable"
        )
    if objects_is_symlink:
        return checkpoint_corruption(
            objects_dir, "checkpoint objects path is a symlink"
        )
    try:
        objects_exists = objects_dir.exists()
    except OSError:
        return checkpoint_corruption(
            objects_dir, "checkpoint objects directory is not readable"
        )
    if not objects_exists:
        return _result.Ok(value=False)
    try:
        objects_is_dir = objects_dir.is_dir()
    except OSError:
        return checkpoint_corruption(
            objects_dir, "checkpoint objects directory is not readable"
        )
    if not objects_is_dir:
        return checkpoint_corruption(
            objects_dir, "checkpoint objects path is not a directory"
        )
    return _result.Ok(value=True)


def validate_checkpoint_object_dir(
    object_dir: Path,
) -> _result.Ok[bool] | _result.Rejected:
    """Validate one checkpoint object directory."""
    try:
        object_is_symlink = object_dir.is_symlink()
    except OSError:
        return checkpoint_corruption(
            object_dir, "checkpoint object is not readable"
        )
    if object_is_symlink:
        return checkpoint_corruption(
            object_dir, "checkpoint object is a symlink"
        )
    try:
        object_exists = object_dir.exists()
    except OSError:
        return checkpoint_corruption(
            object_dir, "checkpoint object is not readable"
        )
    if not object_exists:
        return _result.Ok(value=False)
    try:
        object_is_dir = object_dir.is_dir()
    except OSError:
        return checkpoint_corruption(
            object_dir, "checkpoint object is not readable"
        )
    if not object_is_dir:
        return checkpoint_corruption(
            object_dir, "checkpoint object is not a directory"
        )
    return _result.Ok(value=True)


def validate_checkpoint_state_file(
    state_path: Path,
) -> _result.Ok[None] | _result.Rejected:
    """Validate one checkpoint state file path before reading it."""
    try:
        state_is_symlink = state_path.is_symlink()
    except OSError:
        return checkpoint_corruption(
            state_path, "state file is not readable"
        )
    if state_is_symlink:
        return checkpoint_corruption(
            state_path, "state file is a symlink"
        )
    return _result.Ok(value=None)
