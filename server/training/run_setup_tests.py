"""Tests for training run initialization."""

from __future__ import annotations

from pathlib import Path

from server.training.config import ModelConfig, TrainConfig
from server.training.metrics import read_metrics
from server.training.run_setup import (
    initialize_training_run,
    prepare_training_run,
)
from server.training.torch_checkpoints import (
    read_torch_checkpoint_metadata,
)


def test_prepare_training_run_writes_dashboard_only(
    tmp_path: Path,
) -> None:
    prepared = prepare_training_run(
        run_dir=tmp_path,
    )

    assert prepared.dashboard_path.exists()
    assert read_metrics(tmp_path) == ()


def test_initialize_training_run_writes_torch_checkpoint_and_metrics(
    tmp_path: Path,
) -> None:
    prepared = initialize_training_run(
        run_dir=tmp_path,
        run_id="run-setup-test",
        model_config=ModelConfig(d_model=128),
        train_config=TrainConfig(device="cpu"),
    )

    assert prepared.dashboard_path.exists()
    assert prepared.checkpoint_path.exists()
    assert (
        prepared.checkpoint_path
        == tmp_path / "checkpoints" / "latest.json"
    )
    metadata = read_torch_checkpoint_metadata(prepared.checkpoint_path)
    assert metadata.model_config == ModelConfig(d_model=128)
    assert metadata.train_config == TrainConfig(device="cpu")
    assert metadata.total_rounds == 0
    assert metadata.total_updates == 0
    metrics = read_metrics(tmp_path)
    assert len(metrics) == 1
    assert metrics[0].checkpoint_path == str(prepared.checkpoint_path)
