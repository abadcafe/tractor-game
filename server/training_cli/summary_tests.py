"""Process-level tests for canonical training summary output."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pydantic import ValidationError

from server.training_control.cli_client import TrainingCliSummary


def test_summary_reports_not_initialized_directory(
    tmp_path: Path,
) -> None:
    completed = _summary(tmp_path / "missing")

    assert completed.returncode == 0
    parsed = _parse_summary(completed.stdout)
    assert parsed.state == "NOT_INITIALIZED"
    assert parsed.details is None
    assert parsed.process is None


def test_summary_reports_initialized_run_with_details(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    initialized = subprocess.run(
        (
            sys.executable,
            "-m",
            "server.training_cli",
            "--run-dir",
            str(run_dir),
            "init",
            "--d-model",
            "4",
            "--layers",
            "1",
            "--heads",
            "1",
            "--max-tokens",
            "512",
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert initialized.returncode == 0

    completed = _summary(run_dir)

    assert completed.returncode == 0
    parsed = _parse_summary(completed.stdout)
    assert parsed.state == "READY"
    assert parsed.details is not None
    assert parsed.details.total_updates == 0
    assert parsed.schema_version == 2
    assert len(parsed.checkpoints.manifests) == 1


def test_summary_reports_nonempty_uninitialized_directory_as_broken(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "stderr.log").write_text(
        "old failure\n", encoding="utf-8"
    )

    completed = _summary(run_dir)

    assert completed.returncode == 0
    parsed = _parse_summary(completed.stdout)
    assert parsed.state == "BROKEN"
    assert parsed.reason is not None


def _summary(run_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (
            sys.executable,
            "-m",
            "server.training_cli",
            "--run-dir",
            str(run_dir),
            "summary",
            "--format",
            "json",
        ),
        capture_output=True,
        text=True,
        check=False,
    )


def _parse_summary(text: str) -> TrainingCliSummary:
    try:
        return TrainingCliSummary.model_validate_json(text)
    except ValidationError:
        assert False, f"invalid summary JSON: {text}"
