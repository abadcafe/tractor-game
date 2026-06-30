"""Tests for training run initialization."""

from __future__ import annotations

from pathlib import Path

from server.training.checkpoints import load_checkpoint
from server.training.config import ModelConfig, TrainConfig
from server.training.metrics import read_metrics
from server.training.run_setup import prepare_training_run


def test_prepare_training_run_writes_dashboard_checkpoint_and_metrics(
    tmp_path: Path,
) -> None:
    prepared = prepare_training_run(
        run_dir=tmp_path,
        run_id="run-setup-test",
        model_config=ModelConfig(d_model=128),
        train_config=TrainConfig(device="cpu"),
    )

    assert prepared.dashboard_path.exists()
    assert prepared.checkpoint_path.exists()
    checkpoint = load_checkpoint(prepared.checkpoint_path)
    assert checkpoint.run_id == "run-setup-test"
    assert checkpoint.total_games == 0
    metrics = read_metrics(tmp_path)
    assert len(metrics) == 1
    assert metrics[0].checkpoint_path == str(prepared.checkpoint_path)
