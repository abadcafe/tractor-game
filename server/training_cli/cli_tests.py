"""Process-level tests for the standalone training CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from server.foundation.result import Ok, Rejected
from server.training import TrainingResumeOptions, TrainingStopRequest
from server.training.config import ModelConfig, TrainConfig
from server.training.torch_checkpoints.load import (
    read_torch_checkpoint_metadata,
)
from server.training_cli import cli
from server.training_cli.cli import main


class _CapturingTrainingService:
    def __init__(self, captured: list[int]) -> None:
        self._captured = captured

    def resume(
        self,
        options: TrainingResumeOptions,
        stop_request: TrainingStopRequest,
    ) -> Rejected:
        assert not stop_request.is_requested()
        self._captured.append(options.checkpoint_every_updates)
        return Rejected(reason="captured CLI options")


def test_module_cli_init_creates_zero_update_run(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "server.training_cli",
            "--run-dir",
            str(run_dir),
            "init",
            "--d-model",
            "8",
            "--layers",
            "1",
            "--heads",
            "1",
            "--seed",
            "123",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    checkpoint_path = run_dir / "checkpoints" / "latest.json"
    assert f"checkpoint: {checkpoint_path}" in completed.stdout
    metadata = read_torch_checkpoint_metadata(checkpoint_path)
    assert isinstance(metadata, Ok)
    assert metadata.value.total_updates == 0
    assert metadata.value.model_config == ModelConfig(
        d_model=8, layers=1, heads=1
    )
    assert metadata.value.train_config == TrainConfig(seed=123)


def test_cli_init_rejects_invalid_model_shape_without_traceback(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "server.training_cli",
            "--run-dir",
            str(tmp_path / "invalid-run"),
            "init",
            "--d-model",
            "5",
            "--heads",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--d-model must be divisible by --heads" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_cli_init_rejects_attention_head_too_narrow(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "server.training_cli",
            "--run-dir",
            str(tmp_path / "invalid-run"),
            "init",
            "--d-model",
            "8",
            "--heads",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert (
        "--d-model divided by --heads must be at least 8"
        in completed.stderr
    )
    assert "Traceback" not in completed.stderr


def test_cli_resume_rejects_removed_ready_fd(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "server.training_cli",
            "--run-dir",
            str(tmp_path / "run"),
            "resume",
            "latest.json",
            "--ready-fd",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "unrecognized arguments: --ready-fd 1" in completed.stderr


def test_cli_owns_checkpoint_interval_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[int] = []

    def fake_service() -> _CapturingTrainingService:
        return _CapturingTrainingService(captured)

    monkeypatch.setattr(cli, "TrainingService", fake_service)
    exit_code: int | str | None = None
    try:
        main(
            (
                "--run-dir",
                "training_runs",
                "resume",
                "latest.json",
            ),
            stop_request=TrainingStopRequest(),
        )
    except SystemExit as error:
        exit_code = error.code

    assert exit_code == 2
    assert captured == [5]


def test_main_requires_one_subcommand(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code: int | str | None = None
    try:
        main(())
    except SystemExit as error:
        exit_code = error.code

    assert exit_code == 2
    assert "required" in capsys.readouterr().err


def test_main_init_rejects_removed_max_tokens_option(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code: int | str | None = None
    try:
        main(
            (
                "--run-dir",
                "training_runs",
                "init",
                "--max-tokens",
                "511",
            )
        )
    except SystemExit as error:
        exit_code = error.code

    assert exit_code == 2
    assert "unrecognized arguments: --max-tokens 511" in (
        capsys.readouterr().err
    )
