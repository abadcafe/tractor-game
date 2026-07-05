"""Tests for runtime telemetry persistence."""

from __future__ import annotations

from pathlib import Path

from server.result import Ok
from server.training.runtime.telemetry import (
    IntervalTelemetrySink,
    JsonlTelemetrySink,
    NullTelemetrySink,
    ProcessStage,
    TelemetryEvent,
    telemetry_path,
)


def test_jsonl_telemetry_sink_appends_event(tmp_path: Path) -> None:
    event = TelemetryEvent(
        run_id="run-a",
        process_label="worker-0",
        stage="rollout",
        total_rounds=7,
        total_updates=3,
        progress_numerator=2,
        progress_denominator=5,
        unix_seconds=123.5,
    )

    appended = JsonlTelemetrySink(tmp_path).append(event)

    assert isinstance(appended, Ok)
    text = telemetry_path(tmp_path).read_text(encoding="utf-8")
    assert '"process_label": "worker-0"' in text
    assert '"stage": "rollout"' in text
    assert '"total_rounds": 7' in text
    assert text.endswith("\n")


def test_jsonl_telemetry_sink_appends_multiple_lines(
    tmp_path: Path,
) -> None:
    sink = JsonlTelemetrySink(tmp_path)
    first = sink.append(
        TelemetryEvent(
            run_id="run-a",
            process_label="coordinator",
            stage="coordinator",
            total_rounds=0,
            total_updates=0,
            progress_numerator=0,
            progress_denominator=2,
            unix_seconds=1.0,
        )
    )
    second = sink.append(
        TelemetryEvent(
            run_id="run-a",
            process_label="coordinator",
            stage="complete",
            total_rounds=2,
            total_updates=2,
            progress_numerator=2,
            progress_denominator=2,
            unix_seconds=2.0,
        )
    )

    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    lines = (
        telemetry_path(tmp_path)
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(lines) == 2
    assert '"stage": "coordinator"' in lines[0]
    assert '"stage": "complete"' in lines[1]


def test_interval_telemetry_sink_throttles_repeated_stage(
    tmp_path: Path,
) -> None:
    sink = IntervalTelemetrySink(
        sink=JsonlTelemetrySink(tmp_path),
        min_interval_seconds=10.0,
    )

    first = sink.append(
        TelemetryEvent(
            run_id="run-a",
            process_label="model-rank-0",
            stage="inference",
            total_rounds=0,
            total_updates=0,
            progress_numerator=0,
            progress_denominator=1,
            unix_seconds=1.0,
        )
    )
    repeated = sink.append(
        TelemetryEvent(
            run_id="run-a",
            process_label="model-rank-0",
            stage="inference",
            total_rounds=0,
            total_updates=0,
            progress_numerator=0,
            progress_denominator=1,
            unix_seconds=2.0,
        )
    )
    later = sink.append(
        TelemetryEvent(
            run_id="run-a",
            process_label="model-rank-0",
            stage="inference",
            total_rounds=0,
            total_updates=0,
            progress_numerator=0,
            progress_denominator=1,
            unix_seconds=11.0,
        )
    )

    assert isinstance(first, Ok)
    assert isinstance(repeated, Ok)
    assert isinstance(later, Ok)
    lines = (
        telemetry_path(tmp_path)
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(lines) == 2
    assert '"unix_seconds": 1.0' in lines[0]
    assert '"unix_seconds": 11.0' in lines[1]


def test_interval_telemetry_sink_keeps_stage_changes_and_complete(
    tmp_path: Path,
) -> None:
    sink = IntervalTelemetrySink(
        sink=JsonlTelemetrySink(tmp_path),
        min_interval_seconds=10.0,
    )
    events: tuple[tuple[ProcessStage, float], ...] = (
        ("rollout", 1.0),
        ("update", 2.0),
        ("complete", 3.0),
    )

    for stage, timestamp in events:
        appended = sink.append(
            TelemetryEvent(
                run_id="run-a",
                process_label="coordinator",
                stage=stage,
                total_rounds=1,
                total_updates=1,
                progress_numerator=1,
                progress_denominator=1,
                unix_seconds=timestamp,
            )
        )
        assert isinstance(appended, Ok)

    lines = (
        telemetry_path(tmp_path)
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(lines) == 3
    assert '"stage": "rollout"' in lines[0]
    assert '"stage": "update"' in lines[1]
    assert '"stage": "complete"' in lines[2]


def test_null_telemetry_sink_records_nothing(tmp_path: Path) -> None:
    event = TelemetryEvent(
        run_id="run-a",
        process_label="worker-0",
        stage="idle",
        total_rounds=0,
        total_updates=0,
        progress_numerator=0,
        progress_denominator=0,
        unix_seconds=1.0,
    )

    appended = NullTelemetrySink().append(event)

    assert isinstance(appended, Ok)
    assert not telemetry_path(tmp_path).exists()
