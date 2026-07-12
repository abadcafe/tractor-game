"""Black-box tests for SQLite runtime telemetry."""

from pathlib import Path

from server.foundation.result import Ok
from server.training.telemetry import (
    IntervalTelemetrySink,
    NullTelemetrySink,
    SqliteTelemetrySink,
    TelemetryEvent,
    TelemetryMeasurement,
    prune_telemetry,
    read_telemetry_records,
)


def test_sqlite_telemetry_sink_appends_typed_event(
    tmp_path: Path,
) -> None:
    sink = SqliteTelemetrySink(tmp_path)

    appended = sink.append(_event(unix_seconds=100.0))
    read = read_telemetry_records(tmp_path)

    assert isinstance(appended, Ok)
    assert isinstance(read, Ok)
    assert len(read.value) == 1
    assert read.value[0].sequence == 1
    assert read.value[0].measurements == (
        TelemetryMeasurement(key="batch_size", value=32),
    )
    assert not (tmp_path / "telemetry.jsonl").exists()


def test_read_telemetry_records_supports_cursor(tmp_path: Path) -> None:
    sink = SqliteTelemetrySink(tmp_path)
    for second in (1.0, 2.0, 3.0):
        assert isinstance(sink.append(_event(unix_seconds=second)), Ok)

    result = read_telemetry_records(tmp_path, after_sequence=1)

    assert isinstance(result, Ok)
    assert [event.sequence for event in result.value] == [2, 3]


def test_interval_sink_throttles_repeated_stage(tmp_path: Path) -> None:
    sink = IntervalTelemetrySink(
        sink=SqliteTelemetrySink(tmp_path), min_interval_seconds=2.0
    )

    assert isinstance(sink.append(_event(unix_seconds=1.0)), Ok)
    assert isinstance(sink.append(_event(unix_seconds=2.0)), Ok)
    assert isinstance(sink.append(_event(unix_seconds=3.0)), Ok)
    read = read_telemetry_records(tmp_path)

    assert isinstance(read, Ok)
    assert [event.unix_seconds for event in read.value] == [1.0, 3.0]


def test_interval_sink_always_keeps_terminal_stage(
    tmp_path: Path,
) -> None:
    sink = IntervalTelemetrySink(
        sink=SqliteTelemetrySink(tmp_path), min_interval_seconds=10.0
    )

    assert isinstance(sink.append(_event(unix_seconds=1.0)), Ok)
    assert isinstance(
        sink.append(
            _event(unix_seconds=2.0).model_copy(
                update={"stage": "stopped"}
            )
        ),
        Ok,
    )
    read = read_telemetry_records(tmp_path)

    assert isinstance(read, Ok)
    assert [event.stage for event in read.value] == [
        "rollout",
        "stopped",
    ]


def test_prune_telemetry_removes_records_older_than_one_day(
    tmp_path: Path,
) -> None:
    sink = SqliteTelemetrySink(tmp_path)
    assert isinstance(sink.append(_event(unix_seconds=1.0)), Ok)
    assert isinstance(sink.append(_event(unix_seconds=90_000.0)), Ok)

    pruned = prune_telemetry(tmp_path, now_ms=90_000_000)
    read = read_telemetry_records(tmp_path)

    assert isinstance(pruned, Ok)
    assert pruned.value == 1
    assert isinstance(read, Ok)
    assert [event.unix_seconds for event in read.value] == [90_000.0]


def test_null_telemetry_sink_records_nothing(tmp_path: Path) -> None:
    result = NullTelemetrySink().append(_event(unix_seconds=1.0))

    assert isinstance(result, Ok)
    assert not (tmp_path / "training.sqlite3").exists()


def _event(*, unix_seconds: float) -> TelemetryEvent:
    return TelemetryEvent(
        process_label="worker-0",
        stage="rollout",
        total_rounds=10,
        total_updates=2,
        progress_numerator=1,
        progress_denominator=4,
        unix_seconds=unix_seconds,
        measurements=(
            TelemetryMeasurement(key="batch_size", value=32),
        ),
    )
