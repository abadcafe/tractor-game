"""Run directory setup for explicit user-started training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from server.training.checkpoints import (
    TrainingCheckpoint,
    save_checkpoint,
)
from server.training.config import ModelConfig, TrainConfig
from server.training.dashboard import write_dashboard
from server.training.metrics import TrainingMetric, append_metric


@dataclass(frozen=True, slots=True)
class PreparedTrainingRun:
    """Files created for a training run."""

    run_dir: Path
    dashboard_path: Path
    checkpoint_path: Path


def prepare_training_run(
    *,
    run_dir: Path,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> PreparedTrainingRun:
    """Create dashboard, initial metrics, and an initial checkpoint."""
    dashboard_path = write_dashboard(run_dir, title="Tractor Training")
    checkpoint_path = run_dir / "checkpoints" / "initial.json"
    save_checkpoint(
        checkpoint_path,
        TrainingCheckpoint(
            run_id=run_id,
            total_games=0,
            total_updates=0,
            model_config=model_config.to_json(),
            train_config=train_config.to_json(),
            token_schema_version="structured-components-numeric-v2",
            rules_progress_version="required-level-v1",
            model_state={},
            optimizer_state={},
            rng_state={},
            best_eval_score=None,
        ),
    )
    append_metric(
        run_dir,
        TrainingMetric(
            run_id=run_id,
            total_games=0,
            total_updates=0,
            games_per_second=0.0,
            decisions_per_second=0.0,
            average_reward=0.0,
            average_level_delta=0.0,
            policy_loss=None,
            value_loss=None,
            entropy=None,
            invalid_action_count=0,
            resample_count=0,
            forced_action_count=0,
            legal_action_rate=1.0,
            average_action_tokens=0.0,
            eval_win_rate=None,
            checkpoint_path=str(checkpoint_path),
        ),
    )
    return PreparedTrainingRun(
        run_dir=run_dir,
        dashboard_path=dashboard_path,
        checkpoint_path=checkpoint_path,
    )
