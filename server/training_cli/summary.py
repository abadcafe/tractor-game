"""Canonical training run summary command contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from server.foundation import result as _result
from server.foundation.json_value import JsonObject
from server.training import TrainingService
from server.training_cli.process_inspection import (
    ProcessInspector,
    TrainingProcess,
)

type TrainingRunState = Literal[
    "NOT_INITIALIZED", "BROKEN", "READY", "RUNNING"
]

SUMMARY_SCHEMA_VERSION = 2


class TrainingSummary(BaseModel):
    """Process and persisted-run state without observations."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[2] = SUMMARY_SCHEMA_VERSION
    run_dir: Path
    state: TrainingRunState
    reason: str | None
    process: TrainingProcess | None
    details: JsonObject | None
    checkpoints: JsonObject


def build_training_summary(
    run_dir: Path,
) -> _result.Ok[TrainingSummary] | _result.Rejected:
    """Inspect PID identity, persisted state, and checkpoints."""
    canonical = run_dir.resolve()
    process_result = ProcessInspector().inspect(canonical)
    if isinstance(process_result, _result.Rejected):
        return process_result
    process = process_result.value
    contents_result = _run_directory_has_contents(canonical)
    if isinstance(contents_result, _result.Rejected):
        return _broken_summary(
            canonical, contents_result.reason, process=process
        )
    if not contents_result.value:
        return _result.Ok(
            value=_summary(
                run_dir=canonical,
                state="NOT_INITIALIZED",
                reason=None,
                process=None,
                details=None,
                checkpoints=_empty_checkpoints(canonical),
            )
        )
    inspected = TrainingService().inspect(canonical)
    if isinstance(inspected, _result.Rejected):
        return _broken_summary(
            canonical, inspected.reason, process=process
        )
    catalog = TrainingService().checkpoint_catalog(canonical)
    if isinstance(catalog, _result.Rejected):
        return _broken_summary(
            canonical, catalog.reason, process=process
        )
    return _result.Ok(
        value=_summary(
            run_dir=canonical,
            state="RUNNING" if process is not None else "READY",
            reason=None,
            process=process,
            details=inspected.value.model_dump(mode="json"),
            checkpoints=catalog.value,
        )
    )


def format_training_summary(summary: TrainingSummary) -> str:
    """Render the canonical model for terminal users."""
    lines = [
        f"state: {summary.state}",
        f"run directory: {summary.run_dir}",
    ]
    if summary.reason is not None:
        lines.append(f"reason: {summary.reason}")
    if summary.process is not None:
        lines.extend(
            (
                f"pid: {summary.process.pid}",
                f"command: {_shell_join(summary.process.argv)}",
            )
        )
    if summary.details is not None:
        lines.extend(
            (
                "checkpoint: "
                + _required_string(summary.details, "checkpoint_path"),
                "rounds: "
                + str(_required_int(summary.details, "total_rounds")),
                "samples: "
                + str(_required_int(summary.details, "total_samples")),
                "updates: "
                + str(_required_int(summary.details, "total_updates")),
            )
        )
    manifests = summary.checkpoints["manifests"]
    assert isinstance(manifests, list)
    lines.append(f"checkpoint manifests: {len(manifests)}")
    return "\n".join(lines)


def _broken_summary(
    run_dir: Path,
    reason: str,
    *,
    process: TrainingProcess | None,
) -> _result.Ok[TrainingSummary]:
    catalog = TrainingService().checkpoint_catalog(run_dir)
    checkpoints = (
        _empty_checkpoints(run_dir)
        if isinstance(catalog, _result.Rejected)
        else catalog.value
    )
    return _result.Ok(
        value=_summary(
            run_dir=run_dir,
            state="BROKEN",
            reason=reason,
            process=process,
            details=None,
            checkpoints=checkpoints,
        )
    )


def _summary(
    *,
    run_dir: Path,
    state: TrainingRunState,
    reason: str | None,
    process: TrainingProcess | None,
    details: JsonObject | None,
    checkpoints: JsonObject,
) -> TrainingSummary:
    return TrainingSummary(
        run_dir=run_dir,
        state=state,
        reason=reason,
        process=process,
        details=details,
        checkpoints=checkpoints,
    )


def _empty_checkpoints(run_dir: Path) -> JsonObject:
    return {
        "checkpoint_directory": str(run_dir / "checkpoints"),
        "manifests": [],
        "objects": [],
        "total_unique_state_bytes": 0,
    }


def _run_directory_has_contents(
    run_dir: Path,
) -> _result.Ok[bool] | _result.Rejected:
    try:
        if not run_dir.exists():
            return _result.Ok(value=False)
        if run_dir.is_symlink() or not run_dir.is_dir():
            return _result.Rejected(
                reason=f"training run directory is unsafe: {run_dir}"
            )
        return _result.Ok(
            value=next(run_dir.iterdir(), None) is not None
        )
    except OSError:
        return _result.Rejected(
            reason=f"training run directory is unreadable: {run_dir}"
        )


def _required_string(value: JsonObject, key: str) -> str:
    item = value[key]
    assert isinstance(item, str)
    return item


def _required_int(value: JsonObject, key: str) -> int:
    item = value[key]
    assert isinstance(item, int)
    return item


def _shell_join(argv: tuple[str, ...]) -> str:
    return " ".join(json.dumps(value) for value in argv)
