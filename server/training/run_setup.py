"""Run directory setup for explicit user-started training."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from server.foundation import result as _result
from server.training.config import ModelConfig, TrainConfig
from server.training.metrics import (
    TrainingMetric,
    append_metric,
)
from server.training.persistence.schema import database_path
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
        cleanup_result = _clear_training_artifacts(run_dir)
        if isinstance(cleanup_result, _result.Rejected):
            return cleanup_result
    else:
        guard = _reject_existing_training_run(run_dir)
        if isinstance(guard, _result.Rejected):
            return guard
    checkpoint_path = run_dir / _CHECKPOINTS_DIR_NAME / "latest.json"
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
    metric_result = append_metric(
        run_dir,
        TrainingMetric(
            total_games=0,
            total_samples=0,
            total_updates=0,
            process_games_per_second=0.0,
            process_samples_per_second=0.0,
            last_round_decisions_per_second=0.0,
            last_team0_reward=0.0,
            last_team1_reward=0.0,
            last_generated_action_count=0,
            last_accepted_action_count=0,
            last_decision_count=0,
            last_average_action_choices=0.0,
            policy_loss=None,
            value_loss=None,
            entropy=None,
            approx_kl=None,
            clip_fraction=None,
            ppo_update_seconds=None,
            ppo_minibatch_loss_seconds=None,
            ppo_observation_batch_seconds=None,
            ppo_observation_encode_seconds=None,
            ppo_value_head_seconds=None,
            ppo_argument_select_seconds=None,
            ppo_argument_decode_seconds=None,
            ppo_argument_distribution_seconds=None,
            ppo_backward_seconds=None,
            ppo_optimizer_step_seconds=None,
            ppo_argument_decode_fraction=None,
            ppo_argument_trace_batch_count=None,
            ppo_argument_trace_row_count=None,
            ppo_argument_trace_token_count=None,
            ppo_argument_trace_valid_token_count=None,
            ppo_argument_trace_padding_token_count=None,
            checkpoint_path=str(checkpoint_path),
        ),
    )
    if isinstance(metric_result, _result.Rejected):
        if prune_failure is not None:
            return _result.Rejected(
                reason=f"{metric_result.reason}; {prune_failure.reason}"
            )
        return metric_result
    if prune_failure is not None:
        return prune_failure
    return _result.Ok(
        value=InitializedTrainingRun(
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
        )
    )


def _reject_existing_training_run(
    run_dir: Path,
) -> _result.Ok[None] | _result.Rejected:
    if not _has_training_artifacts(run_dir):
        return _result.Ok(value=None)
    return _result.Rejected(
        reason=(
            f"training run already exists: {run_dir}; use --resume or "
            "--replace-existing"
        )
    )


def _has_training_artifacts(run_dir: Path) -> bool:
    return any(
        path.exists()
        for path in (
            database_path(run_dir),
            Path(f"{database_path(run_dir)}-wal"),
            Path(f"{database_path(run_dir)}-shm"),
            run_dir / _CHECKPOINTS_DIR_NAME,
        )
    )


def _clear_training_artifacts(
    run_dir: Path,
) -> _result.Ok[None] | _result.Rejected:
    for path in (
        database_path(run_dir),
        Path(f"{database_path(run_dir)}-wal"),
        Path(f"{database_path(run_dir)}-shm"),
    ):
        removal = _remove_training_file(path)
        if isinstance(removal, _result.Rejected):
            return removal
    checkpoint_dir = run_dir / _CHECKPOINTS_DIR_NAME

    try:
        checkpoint_exists = checkpoint_dir.exists()
    except OSError:
        return _result.Rejected(
            reason=(
                f"training artifact is not readable: {checkpoint_dir}"
            )
        )
    if checkpoint_exists:
        if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
            return _result.Rejected(
                reason=(
                    "training checkpoint artifact is not a directory: "
                    f"{checkpoint_dir}"
                )
            )
        try:
            shutil.rmtree(checkpoint_dir)
        except OSError:
            return _result.Rejected(
                reason=(
                    "training checkpoint cleanup failed: "
                    f"{checkpoint_dir}"
                )
            )
    return _result.Ok(value=None)


def _remove_training_file(
    path: Path,
) -> _result.Ok[None] | _result.Rejected:
    try:
        exists = path.exists()
    except OSError:
        return _result.Rejected(
            reason=f"training artifact is not readable: {path}"
        )
    if exists:
        if not path.is_file():
            return _result.Rejected(
                reason=f"training artifact is not a file: {path}"
            )
        try:
            path.unlink()
        except OSError:
            return _result.Rejected(
                reason=f"training artifact cleanup failed: {path}"
            )
    return _result.Ok(value=None)
