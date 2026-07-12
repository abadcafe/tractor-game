"""Canonical training run summary and JSON command contract."""

from __future__ import annotations

import json
import stat
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

SUMMARY_SCHEMA_VERSION = 1
_ARTIFACT_NAMES: tuple[str, ...] = (
    "checkpoints",
    "training.sqlite3",
    "training.sqlite3-wal",
    "training.sqlite3-shm",
)


class TrainingSummary(BaseModel):
    """Versioned output for humans and control adapters."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = SUMMARY_SCHEMA_VERSION
    run_dir: Path
    state: TrainingRunState
    reason: str | None
    process: TrainingProcess | None
    details: JsonObject | None
    metrics: tuple[JsonObject, ...]
    telemetry: tuple[JsonObject, ...]
    checkpoints: JsonObject


def build_training_summary(
    run_dir: Path,
    *,
    metric_after: int | None,
    telemetry_after: int | None,
) -> _result.Ok[TrainingSummary] | _result.Rejected:
    """Inspect process identity, persistence, and observations."""
    canonical = run_dir.resolve()
    process_result = ProcessInspector().inspect(canonical)
    if isinstance(process_result, _result.Rejected):
        return _summary_for_broken(
            canonical,
            process_result.reason,
            metric_after=metric_after,
            telemetry_after=telemetry_after,
        )
    process = process_result.value
    if process is not None:
        observation = _observe(
            canonical,
            metric_after=metric_after,
            telemetry_after=telemetry_after,
            tolerate_failure=False,
        )
        if isinstance(observation, _result.Rejected):
            return observation
        return _result.Ok(
            value=_summary(
                run_dir=canonical,
                state="RUNNING",
                reason=None,
                process=process,
                details=None,
                observation=observation.value,
            )
        )
    artifact_result = _has_training_artifacts(canonical)
    if isinstance(artifact_result, _result.Rejected):
        return _summary_for_broken(
            canonical,
            artifact_result.reason,
            metric_after=metric_after,
            telemetry_after=telemetry_after,
        )
    if not artifact_result.value:
        observation = _observe(
            canonical,
            metric_after=metric_after,
            telemetry_after=telemetry_after,
            tolerate_failure=True,
        )
        assert isinstance(observation, _result.Ok)
        return _result.Ok(
            value=_summary(
                run_dir=canonical,
                state="NOT_INITIALIZED",
                reason=None,
                process=None,
                details=None,
                observation=observation.value,
            )
        )
    inspected = TrainingService().inspect(canonical)
    if isinstance(inspected, _result.Rejected):
        return _summary_for_broken(
            canonical,
            inspected.reason,
            metric_after=metric_after,
            telemetry_after=telemetry_after,
        )
    observation = _observe(
        canonical,
        metric_after=metric_after,
        telemetry_after=telemetry_after,
        tolerate_failure=False,
    )
    if isinstance(observation, _result.Rejected):
        return observation
    return _result.Ok(
        value=_summary(
            run_dir=canonical,
            state="READY",
            reason=None,
            process=None,
            details=inspected.value.model_dump(mode="json"),
            observation=observation.value,
        )
    )


def format_training_summary(summary: TrainingSummary) -> str:
    """Render the canonical model without adding semantics."""
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
        lines.append(
            "checkpoint: "
            + _required_string(summary.details, "checkpoint_path")
        )
        lines.append(
            "updates: "
            + str(_required_int(summary.details, "total_updates"))
        )
    if summary.metrics:
        latest = summary.metrics[-1]
        lines.extend(
            (
                f"games: {_required_int(latest, 'total_games')}",
                f"samples: {_required_int(latest, 'total_samples')}",
                "current updates: "
                + str(_required_int(latest, "total_updates")),
            )
        )
    manifests = summary.checkpoints["manifests"]
    assert isinstance(manifests, list)
    lines.append(f"checkpoint manifests: {len(manifests)}")
    return "\n".join(lines)


def _summary_for_broken(
    run_dir: Path,
    reason: str,
    *,
    metric_after: int | None,
    telemetry_after: int | None,
) -> _result.Ok[TrainingSummary]:
    observation = _observe(
        run_dir,
        metric_after=metric_after,
        telemetry_after=telemetry_after,
        tolerate_failure=True,
    )
    assert isinstance(observation, _result.Ok)
    return _result.Ok(
        value=_summary(
            run_dir=run_dir,
            state="BROKEN",
            reason=reason,
            process=None,
            details=None,
            observation=observation.value,
        )
    )


class _ObservationJson(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    metrics: tuple[JsonObject, ...]
    telemetry: tuple[JsonObject, ...]
    checkpoints: JsonObject


def _observe(
    run_dir: Path,
    *,
    metric_after: int | None,
    telemetry_after: int | None,
    tolerate_failure: bool,
) -> _result.Ok[_ObservationJson] | _result.Rejected:
    observed = TrainingService().observe(
        run_dir,
        metric_after=metric_after,
        telemetry_after=telemetry_after,
    )
    if isinstance(observed, _result.Rejected):
        if not tolerate_failure:
            return observed
        return _result.Ok(value=_empty_observation(run_dir))
    value = observed.value
    return _result.Ok(
        value=_ObservationJson(
            metrics=tuple(
                item.model_dump(mode="json") for item in value.metrics
            ),
            telemetry=tuple(
                item.model_dump(mode="json") for item in value.telemetry
            ),
            checkpoints=value.checkpoints.model_dump(mode="json"),
        )
    )


def _empty_observation(run_dir: Path) -> _ObservationJson:
    return _ObservationJson(
        metrics=(),
        telemetry=(),
        checkpoints={
            "checkpoint_directory": str(run_dir / "checkpoints"),
            "manifests": [],
            "objects": [],
            "total_unique_state_bytes": 0,
        },
    )


def _summary(
    *,
    run_dir: Path,
    state: TrainingRunState,
    reason: str | None,
    process: TrainingProcess | None,
    details: JsonObject | None,
    observation: _ObservationJson,
) -> TrainingSummary:
    return TrainingSummary(
        run_dir=run_dir,
        state=state,
        reason=reason,
        process=process,
        details=details,
        metrics=observation.metrics,
        telemetry=observation.telemetry,
        checkpoints=observation.checkpoints,
    )


def _has_training_artifacts(
    run_dir: Path,
) -> _result.Ok[bool] | _result.Rejected:
    try:
        if run_dir.is_symlink():
            return _result.Rejected(
                reason=f"training run directory is a symlink: {run_dir}"
            )
        if not run_dir.exists():
            return _result.Ok(value=False)
        if not run_dir.is_dir():
            return _result.Rejected(
                reason=(
                    f"training run path is not a directory: {run_dir}"
                )
            )
        for name in _ARTIFACT_NAMES:
            path = run_dir / name
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(metadata.st_mode):
                return _result.Rejected(
                    reason=f"training artifact is a symlink: {path}"
                )
            return _result.Ok(value=True)
    except OSError:
        return _result.Rejected(
            reason=f"training run directory is unreadable: {run_dir}"
        )
    return _result.Ok(value=False)


def _required_int(value: JsonObject, key: str) -> int:
    item = value[key]
    assert isinstance(item, int)
    return item


def _required_string(value: JsonObject, key: str) -> str:
    item = value[key]
    assert isinstance(item, str)
    return item


def _shell_join(argv: tuple[str, ...]) -> str:
    return " ".join(json.dumps(item) for item in argv)
