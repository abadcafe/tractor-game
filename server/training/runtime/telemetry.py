"""Low-overhead runtime telemetry records."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from server import result as _result
from server.training.json_types import JsonObject

type ProcessStage = Literal[
    "coordinator",
    "loading",
    "rollout",
    "inference",
    "update",
    "checkpoint",
    "idle",
    "failed",
    "complete",
]
type TelemetryMeasurementValue = int | float

TELEMETRY_FILENAME = "telemetry.jsonl"


@dataclass(frozen=True, slots=True)
class TelemetryMeasurement:
    """One numeric telemetry measurement."""

    key: str
    value: TelemetryMeasurementValue

    def __post_init__(self) -> None:
        assert self.key
        assert math.isfinite(float(self.value))


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """One append-only runtime progress sample."""

    run_id: str
    process_label: str
    stage: ProcessStage
    total_rounds: int
    total_updates: int
    progress_numerator: int
    progress_denominator: int
    unix_seconds: float
    measurements: tuple[TelemetryMeasurement, ...] = ()

    def __post_init__(self) -> None:
        assert self.run_id
        assert self.process_label
        assert self.stage in (
            "coordinator",
            "loading",
            "rollout",
            "inference",
            "update",
            "checkpoint",
            "idle",
            "failed",
            "complete",
        )
        assert self.total_rounds >= 0
        assert self.total_updates >= 0
        assert self.progress_numerator >= 0
        assert self.progress_denominator >= 0
        assert self.progress_numerator <= self.progress_denominator
        assert math.isfinite(self.unix_seconds)


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
        assert event.run_id
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
class JsonlTelemetrySink:
    """Append telemetry events to the run directory."""

    run_dir: Path

    def append(
        self, event: TelemetryEvent
    ) -> _result.Ok[None] | _result.Rejected:
        path = telemetry_path(self.run_dir)
        try:
            line = json.dumps(
                _event_to_json(event),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
        except ValueError:
            return _result.Rejected(
                reason=f"telemetry serialization failed: {path}"
            )
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(f"{line}\n")
        except OSError:
            return _result.Rejected(
                reason=f"telemetry write failed: {path}"
            )
        return _result.Ok(value=None)


def telemetry_path(run_dir: Path) -> Path:
    """Return the standard telemetry path for a run directory."""
    return run_dir / TELEMETRY_FILENAME


def _should_emit(
    *,
    previous: TelemetryEvent,
    current: TelemetryEvent,
    min_interval_seconds: float,
) -> bool:
    if current.stage in ("failed", "complete"):
        return True
    if current.stage != previous.stage:
        return True
    if current.measurements and not previous.measurements:
        return True
    return (
        current.unix_seconds - previous.unix_seconds
    ) >= min_interval_seconds


def _event_to_json(event: TelemetryEvent) -> JsonObject:
    payload: JsonObject = {
        "run_id": event.run_id,
        "process_label": event.process_label,
        "stage": event.stage,
        "total_rounds": event.total_rounds,
        "total_updates": event.total_updates,
        "progress_numerator": event.progress_numerator,
        "progress_denominator": event.progress_denominator,
        "unix_seconds": event.unix_seconds,
    }
    for measurement in event.measurements:
        payload[measurement.key] = measurement.value
    return payload
