"""Tests for the training CLI entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

from server.rules.cards import Rank
from server.sm.required_progress import RequiredLevelPlan
from server.training.config import ModelConfig, TrainConfig
from server.training.metrics import read_metrics
from server.training.run_setup import initialize_training_run
from server.training.torch_checkpoints import (
    read_torch_checkpoint_metadata,
)
from server.training.train import main


def test_init_only_prints_resumable_torch_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_path = tmp_path / "checkpoints" / "latest.json"

    main(
        (
            "--run-dir",
            str(tmp_path),
            "--init-only",
            "--d-model",
            "4",
            "--layers",
            "1",
            "--heads",
            "1",
            "--dropout",
            "0.0",
            "--max-tokens",
            "64",
            "--required-levels",
            "J,A",
        )
    )

    output = capsys.readouterr().out
    assert f"checkpoint: {checkpoint_path}" in output
    metadata = read_torch_checkpoint_metadata(checkpoint_path)
    assert metadata.model_config == ModelConfig(
        d_model=4,
        layers=1,
        heads=1,
        dropout=0.0,
        max_tokens=64,
    )
    assert metadata.train_config.required_level_plan == (
        RequiredLevelPlan(required_levels=(Rank.JACK, Rank.ACE))
    )
    assert metadata.total_rounds == 0
    assert metadata.total_updates == 0


def test_resume_zero_rounds_does_not_append_initial_metric(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    initialized = initialize_training_run(
        run_dir=tmp_path,
        run_id=tmp_path.name,
        model_config=ModelConfig(
            d_model=4,
            layers=1,
            heads=1,
            dropout=0.0,
            max_tokens=64,
        ),
        train_config=TrainConfig(device="cpu"),
    )
    metrics_before = read_metrics(tmp_path)

    main(
        (
            "--run-dir",
            str(tmp_path),
            "--resume",
            str(initialized.checkpoint_path),
            "--max-rounds",
            "0",
        )
    )

    capsys.readouterr()
    assert read_metrics(tmp_path) == metrics_before
