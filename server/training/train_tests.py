"""Tests for the training CLI entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from server.result import Ok
from server.training.config import ModelConfig, TrainConfig
from server.training.metrics import read_metrics
from server.training.run_setup import (
    InitializedTrainingRun,
)
from server.training.run_setup import (
    initialize_training_run as _initialize_training_run,
)
from server.training.torch_checkpoints import (
    TorchCheckpointMetadata,
)
from server.training.torch_checkpoints import (
    read_torch_checkpoint_metadata as _read_torch_checkpoint_metadata,
)
from server.training.train import MIN_CLI_MAX_TOKENS, main


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
            "512",
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
        max_tokens=512,
    )
    assert metadata.train_config == TrainConfig(seed=123)
    assert metadata.total_rounds == 0
    assert metadata.total_updates == 0


def test_init_only_persists_ppo_profile(
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
            "512",
            "--ppo-profile",
            "detailed",
        )
    )

    capsys.readouterr()
    metadata = read_torch_checkpoint_metadata(checkpoint_path)
    assert metadata.train_config.ppo_profile == "detailed"


def test_new_run_rejects_existing_run_without_force(
    tmp_path: Path,
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
        train_config=TrainConfig(device="cpu", seed=1),
    )

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from server.training.train import main\n"
                "main((\n"
                "    '--run-dir',\n"
                f"    {str(tmp_path)!r},\n"
                "    '--init-only',\n"
                "    '--d-model',\n"
                "    '8',\n"
                "    '--layers',\n"
                "    '1',\n"
                "    '--heads',\n"
                "    '1',\n"
                "    '--max-tokens',\n"
                "    '512',\n"
                "    '--seed',\n"
                "    '2',\n"
                "))\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "training run already exists" in completed.stderr
    assert "--force-new-run" in completed.stderr
    metadata = read_torch_checkpoint_metadata(
        initialized.checkpoint_path
    )
    assert metadata.model_config == ModelConfig(
        d_model=4,
        layers=1,
        heads=1,
        max_tokens=64,
    )
    assert metadata.train_config == TrainConfig(device="cpu", seed=1)
    assert len(read_metrics(tmp_path)) == 1


def test_force_new_run_reinitializes_existing_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    initialize_training_run(
        run_dir=tmp_path,
        run_id=tmp_path.name,
        model_config=ModelConfig(
            d_model=4,
            layers=1,
            heads=1,
            max_tokens=64,
        ),
        train_config=TrainConfig(device="cpu", seed=1),
    )
    checkpoint_path = tmp_path / "checkpoints" / "latest.json"

    main(
        (
            "--run-dir",
            str(tmp_path),
            "--init-only",
            "--force-new-run",
            "--d-model",
            "8",
            "--layers",
            "1",
            "--heads",
            "1",
            "--max-tokens",
            "512",
            "--seed",
            "2",
        )
    )

    output = capsys.readouterr().out
    assert f"checkpoint: {checkpoint_path}" in output
    metadata = read_torch_checkpoint_metadata(checkpoint_path)
    assert metadata.model_config == ModelConfig(
        d_model=8,
        layers=1,
        heads=1,
        max_tokens=512,
    )
    assert metadata.train_config == TrainConfig(seed=2)
    metrics = read_metrics(tmp_path)
    assert len(metrics) == 1
    assert metrics[0].run_id == tmp_path.name


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


def test_resume_corrupt_checkpoint_reports_cli_error(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "source-run"
    checkpoint_path = run_dir / "checkpoints" / "latest.json"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text("{not checkpoint json", encoding="utf-8")

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from server.training.train import main\n"
                "main((\n"
                "    '--resume',\n"
                f"    {str(checkpoint_path)!r},\n"
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
    assert "checkpoint corruption:" in completed.stderr
    assert "latest.json" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_resume_invalid_utf8_checkpoint_reports_cli_error(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "source-run"
    checkpoint_path = run_dir / "checkpoints" / "latest.json"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"\xff")

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from server.training.train import main\n"
                "main((\n"
                "    '--resume',\n"
                f"    {str(checkpoint_path)!r},\n"
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
    assert "checkpoint corruption:" in completed.stderr
    assert "manifest is not valid UTF-8" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_resume_directory_checkpoint_reports_cli_error(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "source-run"
    checkpoint_path = run_dir / "checkpoints" / "latest.json"
    checkpoint_path.mkdir(parents=True)

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from server.training.train import main\n"
                "main((\n"
                "    '--resume',\n"
                f"    {str(checkpoint_path)!r},\n"
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
    assert "checkpoint corruption:" in completed.stderr
    assert "manifest file is not readable" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_cuda_device_unavailable_reports_cli_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    exit_code: object = None

    try:
        main(
            (
                "--run-dir",
                str(tmp_path),
                "--init-only",
                "--device",
                "cuda",
                "--d-model",
                "4",
                "--layers",
                "1",
                "--heads",
                "1",
                "--max-tokens",
                str(MIN_CLI_MAX_TOKENS),
            )
        )
    except SystemExit as error:
        exit_code = error.code

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--device cuda is unavailable" in captured.err
    assert "Traceback" not in captured.err
    assert read_metrics(tmp_path) == ()
    assert not (tmp_path / "checkpoints").exists()


def test_cli_rejects_too_small_max_tokens() -> None:
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from server.training.train import main\n"
                "main(('--max-tokens', "
                f"{str(MIN_CLI_MAX_TOKENS - 1)!r}))\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert f"must be >= {MIN_CLI_MAX_TOKENS}" in completed.stderr
    assert "Traceback" not in completed.stderr


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


def initialize_training_run(
    *,
    run_dir: Path,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> InitializedTrainingRun:
    result = _initialize_training_run(
        run_dir=run_dir,
        run_id=run_id,
        model_config=model_config,
        train_config=train_config,
    )
    assert isinstance(result, Ok)
    return result.value


def read_torch_checkpoint_metadata(
    path: Path,
) -> TorchCheckpointMetadata:
    result = _read_torch_checkpoint_metadata(path)
    assert isinstance(result, Ok)
    return result.value
