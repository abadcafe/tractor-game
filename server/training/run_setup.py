"""Run directory setup for explicit user-started training."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from server import result as _result
from server.training.config import ModelConfig, TrainConfig
from server.training.dashboard import write_dashboard
from server.training.metrics import (
    TrainingMetric,
    append_metric,
    metrics_path,
)
from server.training.torch_checkpoints import (
    save_torch_checkpoint,
)
from server.training.training_state import (
    create_training_state,
    resolve_training_device,
)

_CHECKPOINTS_DIR_NAME = "checkpoints"


@dataclass(frozen=True, slots=True)
class PreparedTrainingRun:
    """Files prepared for a training run."""

    run_dir: Path
    dashboard_path: Path


@dataclass(frozen=True, slots=True)
class InitializedTrainingRun:
    """Files and initial checkpoint created for a new training run."""

    run_dir: Path
    dashboard_path: Path
    checkpoint_path: Path


def prepare_training_run(
    *,
    run_dir: Path,
) -> _result.Ok[PreparedTrainingRun] | _result.Rejected:
    """Create dashboard files without changing training progress."""
    dashboard_result = write_dashboard(
        run_dir, title="Tractor Training"
    )
    if isinstance(dashboard_result, _result.Rejected):
        return dashboard_result
    return _result.Ok(
        value=PreparedTrainingRun(
            run_dir=run_dir,
            dashboard_path=dashboard_result.value,
        )
    )


def initialize_training_run(
    *,
    run_dir: Path,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    force_new_run: bool = False,
) -> _result.Ok[InitializedTrainingRun] | _result.Rejected:
    """Create dashboard, initial metric, and torch checkpoint."""
    device_result = resolve_training_device(train_config.device)
    if isinstance(device_result, _result.Rejected):
        return device_result
    if force_new_run:
        cleanup_result = _clear_training_artifacts(run_dir)
        if isinstance(cleanup_result, _result.Rejected):
            return cleanup_result
    else:
        guard = _reject_existing_training_run(run_dir)
        if isinstance(guard, _result.Rejected):
            return guard
    prepared_result = prepare_training_run(run_dir=run_dir)
    if isinstance(prepared_result, _result.Rejected):
        return prepared_result
    prepared = prepared_result.value
    checkpoint_path = run_dir / _CHECKPOINTS_DIR_NAME / "latest.json"
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        device=device_result.value,
    )
    save_result = save_torch_checkpoint(
        manifest_paths=(checkpoint_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=0,
        total_updates=0,
        retained_update_count=train_config.checkpoint_retention_updates,
    )
    if isinstance(save_result, _result.Rejected):
        return save_result
    metric_result = append_metric(
        run_dir,
        TrainingMetric(
            run_id=run_id,
            total_games=0,
            total_updates=0,
            process_games_per_second=0.0,
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
            checkpoint_path=str(checkpoint_path),
        ),
    )
    if isinstance(metric_result, _result.Rejected):
        return metric_result
    return _result.Ok(
        value=InitializedTrainingRun(
            run_dir=prepared.run_dir,
            dashboard_path=prepared.dashboard_path,
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
            "--force-new-run"
        )
    )


def _has_training_artifacts(run_dir: Path) -> bool:
    return (
        metrics_path(run_dir).exists()
        or (run_dir / _CHECKPOINTS_DIR_NAME).exists()
    )


def _clear_training_artifacts(
    run_dir: Path,
) -> _result.Ok[None] | _result.Rejected:
    path = metrics_path(run_dir)
    try:
        metrics_exists = path.exists()
    except OSError:
        return _result.Rejected(
            reason=f"training artifact is not readable: {path}"
        )
    if metrics_exists:
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
