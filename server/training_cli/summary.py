"""One-shot CLI composition of independent training read models."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from server.foundation import result as _result
from server.training_artifacts import (
    CheckpointCatalog,
    read_checkpoint_catalog,
)
from server.training_control.config import training_control_config
from server.training_control.process_inspection import (
    ProcessInspector,
    ProcessSnapshot,
)
from server.training_metrics.queries import (
    TrainingMetrics,
    query_training_metrics,
)

SUMMARY_SCHEMA_VERSION = 3


class TrainingSummary(BaseModel):
    """A terminal-only composition, never a Web process contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[3] = SUMMARY_SCHEMA_VERSION
    run_dir: Path
    process: ProcessSnapshot | None
    metrics: TrainingMetrics
    checkpoints: CheckpointCatalog


def build_training_summary(
    run_dir: Path,
) -> _result.Ok[TrainingSummary] | _result.Rejected:
    """Read process, metrics, and artifacts without loading Torch."""
    canonical = run_dir.resolve()
    config = training_control_config()
    process_result = ProcessInspector(
        runtime_root=config.control_runtime_dir
    ).inspect(canonical)
    if isinstance(process_result, _result.Rejected):
        return process_result
    metrics_result = query_training_metrics(
        canonical, update_limit=500, series_points=500
    )
    if isinstance(metrics_result, _result.Rejected):
        return metrics_result
    checkpoint_result = read_checkpoint_catalog(canonical)
    if isinstance(checkpoint_result, _result.Rejected):
        return checkpoint_result
    return _result.Ok(
        value=TrainingSummary(
            run_dir=canonical,
            process=process_result.value,
            metrics=metrics_result.value,
            checkpoints=checkpoint_result.value,
        )
    )


def format_training_summary(
    summary: TrainingSummary, *, now_ms: int | None = None
) -> str:
    """Render explicit domain sections for terminal users."""
    observed_now = (
        time.time_ns() // 1_000_000 if now_ms is None else now_ms
    )
    lines = [f"run directory: {summary.run_dir}", "", "process"]
    if summary.process is None:
        lines.append("  not running")
    else:
        process = summary.process
        uptime_seconds = (
            max(observed_now - process.started_at_ms, 0) // 1000
        )
        lines.extend(
            (
                f"  command: {process.command}",
                f"  ready: {str(process.ready).lower()}",
                f"  pid: {process.pid}",
                f"  start ticks: {process.start_ticks}",
                f"  started at: {_timestamp(process.started_at_ms)}",
                f"  uptime: {uptime_seconds}s",
                f"  kernel state: {process.kernel_state}",
                f"  executable: {process.executable}",
                f"  working directory: {process.working_directory}",
                f"  process group id: {process.process_group_id}",
                f"  unix session id: {process.unix_session_id}",
                f"  argv: {_shell_join(process.argv)}",
            )
        )
    lines.extend(
        (
            "",
            "metrics",
            f"  store id: {summary.metrics.store_id or '-'}",
            f"  through sequence: {summary.metrics.through_sequence}",
            "  integrity: "
            + (
                "complete" if summary.metrics.complete else "incomplete"
            ),
            f"  dropped events: {summary.metrics.dropped_event_count}",
        )
    )
    for key, value in sorted(summary.metrics.totals.items()):
        lines.append(f"  {key}: {json.dumps(value)}")
    valid_manifests = sum(
        item.valid for item in summary.checkpoints.manifests
    )
    invalid_manifests = (
        len(summary.checkpoints.manifests) - valid_manifests
    )
    orphan_objects = sum(
        item.orphan for item in summary.checkpoints.objects
    )
    lines.extend(
        (
            "",
            "checkpoints",
            f"  directory: {summary.checkpoints.checkpoint_directory}",
            f"  manifests: {len(summary.checkpoints.manifests)}",
            f"  valid manifests: {valid_manifests}",
            f"  invalid manifests: {invalid_manifests}",
            f"  objects: {len(summary.checkpoints.objects)}",
            f"  orphan objects: {orphan_objects}",
            "  unique state bytes: "
            f"{summary.checkpoints.total_unique_state_bytes}",
        )
    )
    latest = next(
        (
            item
            for item in summary.checkpoints.manifests
            if item.name == "latest.json"
        ),
        None,
    )
    if latest is not None:
        lines.append(f"  latest valid: {str(latest.valid).lower()}")
        if latest.error is not None:
            lines.append(f"  latest error: {latest.error}")
        for label, value in (
            ("latest rounds", latest.total_rounds),
            ("latest samples", latest.total_samples),
            ("latest updates", latest.total_updates),
        ):
            if value is not None:
                lines.append(f"  {label}: {value}")
    return "\n".join(lines)


def _shell_join(argv: tuple[str, ...]) -> str:
    return " ".join(json.dumps(value) for value in argv)


def _timestamp(value_ms: int) -> str:
    return datetime.fromtimestamp(value_ms / 1000, tz=UTC).isoformat(
        timespec="milliseconds"
    )
