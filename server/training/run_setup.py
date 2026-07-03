"""Run directory setup for explicit user-started training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from server.training.config import ModelConfig, TrainConfig
from server.training.dashboard import write_dashboard
from server.training.metrics import TrainingMetric, append_metric
from server.training.torch_checkpoints import (
    create_training_state,
    save_torch_checkpoint,
)


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
) -> PreparedTrainingRun:
    """Create dashboard files without changing training progress."""
    dashboard_path = write_dashboard(run_dir, title="Tractor Training")
    return PreparedTrainingRun(
        run_dir=run_dir,
        dashboard_path=dashboard_path,
    )


def initialize_training_run(
    *,
    run_dir: Path,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> InitializedTrainingRun:
    """Create dashboard, initial metric, and torch checkpoint."""
    prepared = prepare_training_run(run_dir=run_dir)
    checkpoint_path = run_dir / "checkpoints" / "latest.json"
    device = torch.device(train_config.device)
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        device=device,
    )
    save_torch_checkpoint(
        manifest_paths=(checkpoint_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=0,
        total_updates=0,
        retained_update_count=train_config.checkpoint_retention_updates,
    )
    append_metric(
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
    return InitializedTrainingRun(
        run_dir=prepared.run_dir,
        dashboard_path=prepared.dashboard_path,
        checkpoint_path=checkpoint_path,
    )
