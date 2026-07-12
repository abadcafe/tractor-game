"""Canonicalize a run around a selected resume checkpoint."""

from __future__ import annotations

from pathlib import Path

from server.foundation import result as _result
from server.training.torch_checkpoints.manifest import (
    read_checkpoint_manifest,
    update_checkpoint_manifest_paths,
    write_checkpoint_manifest,
)
from server.training.torch_checkpoints.pruning import (
    preflight_managed_checkpoints,
    prune_torch_checkpoints,
)


def canonicalize_resume_timeline(
    *, run_dir: Path, selected_checkpoint: Path
) -> _result.Ok[None] | _result.Rejected:
    """Make the selected checkpoint the sole canonical timeline head."""
    checkpoint_dir = run_dir / "checkpoints"
    if selected_checkpoint.parent != checkpoint_dir:
        return _result.Rejected(
            reason="selected checkpoint is outside the run directory"
        )
    preflight = preflight_managed_checkpoints(checkpoint_dir)
    if isinstance(preflight, _result.Rejected):
        return preflight
    selected_result = read_checkpoint_manifest(selected_checkpoint)
    if isinstance(selected_result, _result.Rejected):
        return selected_result
    selected = selected_result.value
    update_paths_result = _validated_update_paths(checkpoint_dir)
    if isinstance(update_paths_result, _result.Rejected):
        return update_paths_result
    latest_result = write_checkpoint_manifest(
        path=checkpoint_dir / "latest.json",
        manifest=selected,
    )
    if isinstance(latest_result, _result.Rejected):
        return latest_result
    for update_number, path in update_paths_result.value:
        if update_number <= selected.metadata.total_updates:
            continue
        try:
            path.unlink()
        except OSError:
            return _result.Rejected(
                reason=f"future checkpoint could not be deleted: {path}"
            )
    remaining_count = sum(
        1
        for update_number, _ in update_paths_result.value
        if update_number <= selected.metadata.total_updates
    )
    return prune_torch_checkpoints(
        checkpoint_dir=checkpoint_dir,
        retained_update_count=remaining_count,
    )


def _validated_update_paths(
    checkpoint_dir: Path,
) -> _result.Ok[tuple[tuple[int, Path], ...]] | _result.Rejected:
    try:
        update_paths = update_checkpoint_manifest_paths(checkpoint_dir)
    except OSError:
        return _result.Rejected(
            reason=(
                f"checkpoint manifests are unreadable: {checkpoint_dir}"
            )
        )
    for update_number, path in update_paths:
        manifest_result = read_checkpoint_manifest(path)
        if isinstance(manifest_result, _result.Rejected):
            return manifest_result
        if (
            manifest_result.value.metadata.total_updates
            != update_number
        ):
            return _result.Rejected(
                reason=(
                    "checkpoint update number does not match manifest: "
                    f"{path}"
                )
            )
    return _result.Ok(value=update_paths)
