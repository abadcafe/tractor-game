"""Process-level tests for the one-shot CLI summary composition."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pydantic import ValidationError

from server.training_artifacts import CheckpointCatalog
from server.training_cli.summary import (
    TrainingSummary,
    format_training_summary,
)
from server.training_control.process_inspection import ProcessSnapshot
from server.training_metrics.queries import (
    MetricDatasets,
    TrainingMetrics,
)


def test_summary_composes_empty_domain_models(tmp_path: Path) -> None:
    completed = _summary(tmp_path / "missing")

    assert completed.returncode == 0
    parsed = _parse_summary(completed.stdout)
    assert parsed.schema_version == 3
    assert parsed.process is None
    assert parsed.metrics.through_sequence == 0
    assert parsed.checkpoints.manifests == ()


def test_summary_composes_metrics_and_checkpoint_catalog(
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
    assert initialized.returncode == 0, initialized.stderr

    completed = _summary(run_dir)

    assert completed.returncode == 0, completed.stderr
    parsed = _parse_summary(completed.stdout)
    assert parsed.process is None
    assert parsed.metrics.through_sequence >= 1
    assert len(parsed.checkpoints.manifests) == 1


def test_text_summary_uses_injected_time_for_process_uptime(
    tmp_path: Path,
) -> None:
    datasets = MetricDatasets(
        throughput=(),
        optimization=(),
        ppo_timing=(),
        rollout=(),
        rewards=(),
        inference=(),
        processes=(),
    )
    summary = TrainingSummary(
        run_dir=tmp_path,
        process=ProcessSnapshot(
            pid=123,
            start_ticks=456,
            started_at_ms=1_000,
            kernel_state="S",
            executable=Path("/usr/bin/python"),
            working_directory=tmp_path,
            run_dir=tmp_path,
            argv=("/usr/bin/python", "-m", "server.training_cli"),
            process_group_id=123,
            unix_session_id=123,
            command="resume",
            ready=True,
        ),
        metrics=TrainingMetrics(
            store_id="a" * 32,
            through_sequence=10,
            complete=False,
            dropped_event_count=2,
            totals={"updates": 3},
            datasets=datasets,
        ),
        checkpoints=CheckpointCatalog(
            checkpoint_directory=tmp_path / "checkpoints",
            manifests=(),
            objects=(),
            total_unique_state_bytes=0,
        ),
    )

    rendered = format_training_summary(summary, now_ms=6_000)

    assert "ready: true" in rendered
    assert "start ticks: 456" in rendered
    assert "started at: 1970-01-01T00:00:01.000+00:00" in rendered
    assert "uptime: 5s" in rendered
    assert "integrity: incomplete" in rendered
    assert "dropped events: 2" in rendered


def test_summary_import_and_query_do_not_load_torch(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            "import pathlib, sys; "
            "from server.training_cli.summary import "
            "build_training_summary; "
            "build_training_summary(pathlib.Path(sys.argv[1])); "
            "assert 'torch' not in sys.modules",
            str(tmp_path),
        ),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


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


def _parse_summary(text: str) -> TrainingSummary:
    try:
        return TrainingSummary.model_validate_json(text)
    except ValidationError:
        assert False, f"invalid summary JSON: {text}"
