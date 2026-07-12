"""Black-box tests for training control server configuration."""

from pathlib import Path

import pytest

from server.training_control.config import (
    TrainingControlConfig,
    training_control_config,
)


def test_training_control_config_uses_run_directory_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "configured-run"
    monkeypatch.setenv("TRAINING_RUN_DIR", str(configured))

    config = training_control_config()

    assert config.default_run_dir == configured.resolve()


def test_training_control_config_ignores_removed_task_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRAINING_RUN_DIR", raising=False)
    monkeypatch.setenv("TRAINING_TASK_DIR", str(tmp_path / "removed"))

    config = training_control_config()

    assert (
        config.default_run_dir == (tmp_path / "training_runs").resolve()
    )


def test_training_control_config_resolves_run_directory(
    tmp_path: Path,
) -> None:
    default = tmp_path / "default"
    supplied = tmp_path / "supplied"
    config = TrainingControlConfig(
        default_run_dir=default,
        stop_timeout_seconds=30.0,
    )

    assert config.resolve_run_dir(None) == default.resolve()
    assert config.resolve_run_dir(supplied) == supplied.resolve()
