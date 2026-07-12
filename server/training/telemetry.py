"""Low-overhead runtime telemetry persisted in SQLite."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    ValidationError,
    model_validator,
)

from server.foundation import result as _result
from server.training.persistence.schema import open_reader, open_writer

type ProcessStage = Literal[
    "coordinator",
    "loading",
    "rollout",
    "inference",
    "update",
    "checkpoint",
    "idle",
    "failed",
    "completed",
    "stopped",
]
type TelemetryMeasurementValue = int | float

_RETENTION_MILLISECONDS = 24 * 60 * 60 * 1000


class TelemetryMeasurement(BaseModel):
    """One validated numeric telemetry measurement."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    key: str = Field(min_length=1)
    value: TelemetryMeasurementValue

    def model_post_init(self, _context: object) -> None:
        assert math.isfinite(float(self.value))


class TelemetryEvent(BaseModel):
    """One runtime process observation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    process_label: str = Field(min_length=1)
    stage: ProcessStage
    total_rounds: int = Field(ge=0)
    total_updates: int = Field(ge=0)
    progress_numerator: int = Field(ge=0)
    progress_denominator: int = Field(ge=0)
    unix_seconds: float
    measurements: tuple[TelemetryMeasurement, ...] = ()

    def model_post_init(self, _context: object) -> None:
        assert self.progress_numerator <= self.progress_denominator
        assert math.isfinite(self.unix_seconds)


class StoredTelemetryEvent(TelemetryEvent):
    """One telemetry event with a stable SQLite cursor."""

    sequence: int = Field(gt=0)
    recorded_at_ms: int = Field(ge=0)


class TelemetryMeasurements(RootModel[dict[str, int | float]]):
    """Strict persisted measurement object."""

    model_config = ConfigDict(strict=True)

    @model_validator(mode="after")
    def validate_finite(self) -> TelemetryMeasurements:
        if not all(
            math.isfinite(float(value)) for value in self.root.values()
        ):
            raise ValueError("telemetry measurements must be finite")
        return self


class TelemetrySink(Protocol):
    """Telemetry sink boundary consumed by runtime code."""

    def append(
        self, event: TelemetryEvent
    ) -> _result.Ok[None] | _result.Rejected: ...


class NullTelemetrySink:
    """Telemetry sink that intentionally records nothing."""

    def append(
        self, event: TelemetryEvent
    ) -> _result.Ok[None] | _result.Rejected:
        assert event.process_label
        return _result.Ok(value=None)


def _event_map() -> dict[str, TelemetryEvent]:
    return {}


@dataclass(slots=True)
class IntervalTelemetrySink:
    """Throttle repeated telemetry writes by process label."""

    sink: TelemetrySink
    min_interval_seconds: float
    _last_emitted: dict[str, TelemetryEvent] = field(
        default_factory=_event_map
    )

    def __post_init__(self) -> None:
        assert math.isfinite(self.min_interval_seconds)
        assert self.min_interval_seconds > 0.0

    def append(
        self, event: TelemetryEvent
    ) -> _result.Ok[None] | _result.Rejected:
        last = self._last_emitted.get(event.process_label)
        if last is not None and not _should_emit(
            previous=last,
            current=event,
            min_interval_seconds=self.min_interval_seconds,
        ):
            return _result.Ok(value=None)
        result = self.sink.append(event)
        if isinstance(result, _result.Rejected):
            return result
        self._last_emitted[event.process_label] = event
        return result


@dataclass(frozen=True, slots=True)
class SqliteTelemetrySink:
    """Append runtime telemetry to the task SQLite database."""

    run_dir: Path

    def append(
        self, event: TelemetryEvent
    ) -> _result.Ok[None] | _result.Rejected:
        recorded_at_ms = int(event.unix_seconds * 1000.0)
        measurements = {
            measurement.key: measurement.value
            for measurement in event.measurements
        }
        measurements_json = json.dumps(
            measurements,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        opened = open_writer(self.run_dir)
        if isinstance(opened, _result.Rejected):
            return opened
        connection = opened.value
        try:
            connection.execute(
                "INSERT INTO runtime_telemetry ("
                "recorded_at_ms, process_label, stage, total_rounds, "
                "total_updates, progress_numerator, "
                "progress_denominator, "
                "measurements_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    recorded_at_ms,
                    event.process_label,
                    event.stage,
                    event.total_rounds,
                    event.total_updates,
                    event.progress_numerator,
                    event.progress_denominator,
                    measurements_json,
                ),
            )
            connection.commit()
        except sqlite3.Error:
            return _result.Rejected(
                reason="runtime telemetry could not be written"
            )
        finally:
            connection.close()
        return _result.Ok(value=None)


def read_telemetry_records(
    run_dir: Path,
    *,
    after_sequence: int | None = None,
    limit: int = 500,
) -> _result.Ok[tuple[StoredTelemetryEvent, ...]] | _result.Rejected:
    """Read latest or incrementally newer telemetry records."""
    if after_sequence is not None and after_sequence < 0:
        return _result.Rejected(
            reason="after_sequence must be non-negative"
        )
    if limit <= 0 or limit > 5000:
        return _result.Rejected(
            reason="limit must be between 1 and 5000"
        )
    opened = open_reader(run_dir)
    if isinstance(opened, _result.Rejected):
        return opened
    connection = opened.value
    if connection is None:
        return _result.Ok(value=())
    columns = (
        "sequence, recorded_at_ms, process_label, stage, total_rounds, "
        "total_updates, progress_numerator, progress_denominator, "
        "measurements_json"
    )
    try:
        if after_sequence is None:
            rows = connection.execute(
                f"SELECT {columns} FROM (SELECT {columns} FROM "
                "runtime_telemetry ORDER BY sequence DESC LIMIT ?) "
                "ORDER BY sequence",
                (limit,),
            ).fetchall()
        else:
            rows = connection.execute(
                f"SELECT {columns} FROM runtime_telemetry "
                "WHERE sequence > ? ORDER BY sequence LIMIT ?",
                (after_sequence, limit),
            ).fetchall()
    except sqlite3.Error:
        return _result.Rejected(
            reason="runtime telemetry could not be read"
        )
    finally:
        connection.close()
    records: list[StoredTelemetryEvent] = []
    for row in rows:
        parsed = _stored_event(tuple(row))
        if isinstance(parsed, _result.Rejected):
            return parsed
        records.append(parsed.value)
    return _result.Ok(value=tuple(records))


def prune_telemetry(
    run_dir: Path, *, now_ms: int | None = None
) -> _result.Ok[int] | _result.Rejected:
    """Delete telemetry older than the fixed retention window."""
    current_ms = (
        time.time_ns() // 1_000_000 if now_ms is None else now_ms
    )
    assert current_ms >= 0
    opened = open_writer(run_dir)
    if isinstance(opened, _result.Rejected):
        return opened
    connection = opened.value
    try:
        cursor = connection.execute(
            "DELETE FROM runtime_telemetry WHERE recorded_at_ms < ?",
            (current_ms - _RETENTION_MILLISECONDS,),
        )
        connection.commit()
        deleted = cursor.rowcount
    except sqlite3.Error:
        return _result.Rejected(
            reason="runtime telemetry retention failed"
        )
    finally:
        connection.close()
    return _result.Ok(value=deleted)


def clear_telemetry(
    run_dir: Path,
) -> _result.Ok[None] | _result.Rejected:
    """Clear process telemetry before a resumed timeline starts."""
    opened = open_writer(run_dir)
    if isinstance(opened, _result.Rejected):
        return opened
    connection = opened.value
    try:
        connection.execute("DELETE FROM runtime_telemetry")
        connection.commit()
    except sqlite3.Error:
        return _result.Rejected(
            reason="runtime telemetry could not be cleared"
        )
    finally:
        connection.close()
    return _result.Ok(value=None)


def _should_emit(
    *,
    previous: TelemetryEvent,
    current: TelemetryEvent,
    min_interval_seconds: float,
) -> bool:
    if current.stage in ("failed", "completed", "stopped"):
        return True
    if current.stage != previous.stage:
        return True
    if current.measurements and not previous.measurements:
        return True
    return (
        current.unix_seconds - previous.unix_seconds
    ) >= min_interval_seconds


def _stored_event(
    row: tuple[object, ...],
) -> _result.Ok[StoredTelemetryEvent] | _result.Rejected:
    if len(row) != 9:
        return _result.Rejected(reason="runtime telemetry is invalid")
    measurements_result = _measurements(row[8])
    if isinstance(measurements_result, _result.Rejected):
        return measurements_result
    payload: dict[str, object] = {
        "sequence": row[0],
        "recorded_at_ms": row[1],
        "process_label": row[2],
        "stage": row[3],
        "total_rounds": row[4],
        "total_updates": row[5],
        "progress_numerator": row[6],
        "progress_denominator": row[7],
        "unix_seconds": (
            float(row[1]) / 1000.0 if isinstance(row[1], int) else -1.0
        ),
        "measurements": measurements_result.value,
    }
    try:
        return _result.Ok(
            value=StoredTelemetryEvent.model_validate(payload)
        )
    except ValidationError:
        return _result.Rejected(reason="runtime telemetry is invalid")


def _measurements(
    value: object,
) -> _result.Ok[tuple[TelemetryMeasurement, ...]] | _result.Rejected:
    if not isinstance(value, str):
        return _result.Rejected(reason="runtime telemetry is invalid")
    try:
        loaded: object = json.loads(value)
    except json.JSONDecodeError:
        return _result.Rejected(reason="runtime telemetry is invalid")
    try:
        parsed = TelemetryMeasurements.model_validate(loaded)
    except ValidationError:
        return _result.Rejected(reason="runtime telemetry is invalid")
    measurement_data = parsed.root
    try:
        measurements = tuple(
            TelemetryMeasurement(key=key, value=item)
            for key, item in measurement_data.items()
        )
    except ValidationError:
        return _result.Rejected(reason="runtime telemetry is invalid")
    return _result.Ok(value=measurements)
