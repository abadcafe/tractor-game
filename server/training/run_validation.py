"""Complete validation of a persisted training run."""

from __future__ import annotations

from pathlib import Path

import torch

from server.foundation import result as _result
from server.training.interface import PersistedRunSummary
from server.training.persistence.schema import (
    database_path,
    open_reader,
)
from server.training.runtime.config import ExecutionConfig
from server.training.torch_checkpoints.load import load_torch_checkpoint
from server.training.torch_checkpoints.manifest import (
    managed_checkpoint_manifest_paths,
    managed_update_number_from_manifest_path,
    manifest_state_file_path,
    read_checkpoint_manifest,
)
from server.training.torch_checkpoints.pruning import (
    preflight_managed_checkpoints,
)


def validate_training_run(
    run_dir: Path,
) -> _result.Ok[PersistedRunSummary] | _result.Rejected:
    """Prove checkpoints and the event database are readable."""
    safety_result = _validate_run_paths(run_dir)
    if isinstance(safety_result, _result.Rejected):
        return safety_result
    checkpoint_dir = run_dir / "checkpoints"
    preflight = preflight_managed_checkpoints(checkpoint_dir)
    if isinstance(preflight, _result.Rejected):
        return preflight
    try:
        manifest_paths = managed_checkpoint_manifest_paths(
            checkpoint_dir
        )
    except OSError:
        return _result.Rejected(
            reason=(
                f"checkpoint manifests are unreadable: {checkpoint_dir}"
            )
        )
    latest_path = checkpoint_dir / "latest.json"
    if latest_path not in manifest_paths:
        return _result.Rejected(
            reason=f"latest checkpoint is missing: {latest_path}"
        )
    latest_manifest = None
    for path in manifest_paths:
        manifest_result = read_checkpoint_manifest(path)
        if isinstance(manifest_result, _result.Rejected):
            return manifest_result
        manifest = manifest_result.value
        update_number = managed_update_number_from_manifest_path(path)
        if (
            update_number is not None
            and update_number != manifest.metadata.total_updates
        ):
            return _result.Rejected(
                reason=(
                    "checkpoint update number does not match manifest: "
                    f"{path}"
                )
            )
        loaded = load_torch_checkpoint(
            path=path,
            model_config=manifest.metadata.model_config,
            train_config=manifest.metadata.train_config,
            execution_config=ExecutionConfig(),
            device=torch.device("cpu"),
        )
        if isinstance(loaded, _result.Rejected):
            return loaded
        if path == latest_path:
            latest_manifest = manifest
    assert latest_manifest is not None
    database_result = open_reader(run_dir)
    if isinstance(database_result, _result.Rejected):
        return database_result
    database = database_result.value
    if database is None:
        return _result.Rejected(
            reason=f"training database is missing: {run_dir}"
        )
    database.close()
    state_path = manifest_state_file_path(
        manifest_path=latest_path,
        manifest=latest_manifest,
    )
    try:
        state_size = state_path.stat().st_size
    except OSError:
        return _result.Rejected(
            reason=f"checkpoint state is unreadable: {state_path}"
        )
    metadata = latest_manifest.metadata
    return _result.Ok(
        value=PersistedRunSummary(
            checkpoint_id=latest_manifest.checkpoint_id,
            checkpoint_path=latest_path,
            state_size_bytes=state_size,
            model_config_values=metadata.model_config.to_json(),
            train_config_values=metadata.train_config.to_json(),
            total_rounds=metadata.total_rounds,
            total_samples=metadata.total_samples,
            total_updates=metadata.total_updates,
        )
    )


def _validate_run_paths(
    run_dir: Path,
) -> _result.Ok[None] | _result.Rejected:
    database = database_path(run_dir)
    try:
        if run_dir.is_symlink() or not run_dir.is_dir():
            return _result.Rejected(
                reason=f"training run directory is unsafe: {run_dir}"
            )
        if database.is_symlink() or not database.is_file():
            return _result.Rejected(
                reason=(
                    "training database is missing or unsafe: "
                    f"{database}"
                )
            )
    except OSError:
        return _result.Rejected(
            reason=f"training run is unreadable: {run_dir}"
        )
    return _result.Ok(value=None)
