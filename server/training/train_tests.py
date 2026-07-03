"""Tests for the training CLI entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

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
            "--max-tokens",
            "64",
            "--seed",
            "123",
        )
    )

    output = capsys.readouterr().out
    assert f"checkpoint: {checkpoint_path}" in output
    metadata = read_torch_checkpoint_metadata(checkpoint_path)
    assert metadata.model_config == ModelConfig(
        d_model=4,
        layers=1,
        heads=1,
        max_tokens=64,
    )
    assert metadata.train_config == TrainConfig(seed=123)
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


def test_resume_without_run_dir_uses_checkpoint_run_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "source-run"
    initialized = initialize_training_run(
        run_dir=run_dir,
        run_id=run_dir.name,
        model_config=ModelConfig(
            d_model=4,
            layers=1,
            heads=1,
            max_tokens=64,
        ),
        train_config=TrainConfig(device="cpu"),
    )
    metrics_before = read_metrics(run_dir)

    main(
        (
            "--resume",
            str(initialized.checkpoint_path),
            "--max-rounds",
            "0",
        )
    )

    output = capsys.readouterr().out
    assert f"dashboard: {run_dir / 'index.html'}" in output
    assert f"checkpoint: {initialized.checkpoint_path}" in output
    assert read_metrics(run_dir) == metrics_before


def test_resume_rejects_mismatched_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "source-run"
    initialized = initialize_training_run(
        run_dir=run_dir,
        run_id=run_dir.name,
        model_config=ModelConfig(
            d_model=4,
            layers=1,
            heads=1,
            max_tokens=64,
        ),
        train_config=TrainConfig(device="cpu"),
    )
    other_run_dir = tmp_path / "other-run"

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from server.training.train import main\n"
                "main((\n"
                "    '--run-dir',\n"
                f"    {str(other_run_dir)!r},\n"
                "    '--resume',\n"
                f"    {str(initialized.checkpoint_path)!r},\n"
                "    '--max-rounds',\n"
                "    '0',\n"
                "))\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert (
        "--run-dir must match the run directory that owns --resume"
        in completed.stderr
    )


def test_resume_rejects_checkpoint_outside_run_dir(
    tmp_path: Path,
) -> None:
    invalid_resume = tmp_path / "latest.json"

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from server.training.train import main\n"
                "main(('--resume', "
                f"{str(invalid_resume)!r}, '--max-rounds', '0'))\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert (
        "--resume must point to "
        "<run-dir>/checkpoints/<checkpoint>.json" in completed.stderr
    )


def test_resume_seed_mismatch_reports_cli_error(tmp_path: Path) -> None:
    initialized = initialize_training_run(
        run_dir=tmp_path,
        run_id=tmp_path.name,
        model_config=ModelConfig(
            d_model=4,
            layers=1,
            heads=1,
            max_tokens=64,
        ),
        train_config=TrainConfig(device="cpu", seed=3),
    )

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from server.training.train import main\n"
                "main((\n"
                "    '--resume',\n"
                f"    {str(initialized.checkpoint_path)!r},\n"
                "    '--seed',\n"
                "    '4',\n"
                "    '--max-rounds',\n"
                "    '0',\n"
                "))\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert (
        "--seed must match the checkpoint seed when using --resume"
        in completed.stderr
    )
    assert "AssertionError" not in completed.stderr


def test_cli_rejects_removed_dropout_argument() -> None:
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from server.training.train import main\n"
                "main(('--dropout', '0.0'))\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "unrecognized arguments: --dropout" in completed.stderr
