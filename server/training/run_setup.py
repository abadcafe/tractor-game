"""Run directory setup for explicit user-started training."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from server.foundation import result as _result
from server.training.config import ModelConfig, TrainConfig
from server.training.event_log import (
    ProcessIdentity,
    StructuredEventSink,
)
from server.training.persistence.schema import initialize_database
from server.training.runtime.config import ExecutionConfig
from server.training.torch_checkpoints.save import save_torch_checkpoint
from server.training.training_state import create_training_state

_CHECKPOINTS_DIR_NAME = "checkpoints"


@dataclass(frozen=True, slots=True)
class InitializedTrainingRun:
    """Files and initial checkpoint created for a new training run."""

    run_dir: Path
    checkpoint_path: Path


def initialize_training_run(
    *,
    run_dir: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    replace_existing: bool = False,
) -> _result.Ok[InitializedTrainingRun] | _result.Rejected:
    """Create portable zero-update state without runtime preflight."""
    if replace_existing:
        cleanup_result = _clear_run_directory(run_dir)
        if isinstance(cleanup_result, _result.Rejected):
            return cleanup_result
    else:
        guard = _reject_existing_training_run(run_dir)
        if isinstance(guard, _result.Rejected):
            return guard
    checkpoint_path = run_dir / _CHECKPOINTS_DIR_NAME / "latest.json"
    database_result = initialize_database(run_dir)
    if isinstance(database_result, _result.Rejected):
        return database_result
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
    )
    save_result = save_torch_checkpoint(
        manifest_paths=(checkpoint_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=0,
        total_samples=0,
        total_updates=0,
        retained_update_count=0,
    )
    if isinstance(save_result, _result.Rejected):
        return save_result
    prune_failure = save_result.value.post_commit_prune_failure
    if prune_failure is not None:
        return prune_failure
    event_sink = StructuredEventSink(
        run_dir=run_dir,
        session_id=None,
        process=ProcessIdentity(kind="initializer"),
    )
    event_sink.emit(
        "run.initialized",
        fields={
            "checkpoint_path": str(checkpoint_path),
            "model_config": model_config.to_json(),
            "train_config": train_config.to_json(),
            "total_rounds": 0,
            "total_samples": 0,
            "total_updates": 0,
        },
    )
    event_sink.close()
    return _result.Ok(
        value=InitializedTrainingRun(
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
        )
    )


def _reject_existing_training_run(
    run_dir: Path,
) -> _result.Ok[None] | _result.Rejected:
    contents_result = _run_directory_has_contents(run_dir)
    if isinstance(contents_result, _result.Rejected):
        return contents_result
    if not contents_result.value:
        return _result.Ok(value=None)
    return _result.Rejected(
        reason=(
            f"training run directory is not empty: {run_dir}; use "
            "--resume or --replace-existing"
        )
    )


def _run_directory_has_contents(
    run_dir: Path,
) -> _result.Ok[bool] | _result.Rejected:
    try:
        if run_dir.is_symlink():
            return _result.Rejected(
                reason=f"training run directory is a symlink: {run_dir}"
            )
        if not run_dir.exists():
            return _result.Ok(value=False)
        if not run_dir.is_dir():
            return _result.Rejected(
                reason=(
                    f"training run path is not a directory: {run_dir}"
                )
            )
        return _result.Ok(
            value=next(run_dir.iterdir(), None) is not None
        )
    except OSError:
        return _result.Rejected(
            reason=f"training run directory is unreadable: {run_dir}"
        )


def _clear_run_directory(
    run_dir: Path,
) -> _result.Ok[None] | _result.Rejected:
    contents_result = _run_directory_has_contents(run_dir)
    if isinstance(contents_result, _result.Rejected):
        return contents_result
    if not contents_result.value:
        return _result.Ok(value=None)
    try:
        children = tuple(run_dir.iterdir())
    except OSError:
        return _result.Rejected(
            reason=f"training run directory is unreadable: {run_dir}"
        )
    for child in children:
        try:
            if child.is_symlink() or not child.is_dir():
                child.unlink()
            else:
                shutil.rmtree(child)
        except OSError:
            return _result.Rejected(
                reason=(
                    f"training run directory cleanup failed: {child}"
                )
            )
    return _result.Ok(value=None)
