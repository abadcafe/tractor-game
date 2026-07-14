"""Black-box tests for sequence-ordered full-run metrics."""

from __future__ import annotations

import math
from pathlib import Path

from server.foundation.result import Ok
from server.training_events import (
    EventContext,
    ProcessIdentity,
    StructuredEventSink,
)
from server.training_events.store import initialize_database
from server.training_metrics.queries import (
    query_metrics_cursor,
    query_training_metrics,
)


def test_metrics_project_all_updates_without_sessions(
    tmp_path: Path,
) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)
    sink = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="coordinator"),
    )
    sink.emit(
        "update",
        context=EventContext(policy_version=0, rollout_id="rollout-a"),
        fields={
            "total_rounds": 4,
            "total_samples": 32,
            "total_updates": 1,
            "process_rounds_per_second": 2.0,
            "process_samples_per_second": 16.0,
            "policy_loss": 0.25,
            "update_cycle_seconds": 3.0,
        },
    )
    sink.close()

    result = query_training_metrics(
        tmp_path, update_limit=200, series_points=200
    )

    assert isinstance(result, Ok)
    assert result.value.store_id is not None
    assert result.value.totals["total_samples"] == 32
    assert result.value.totals["samples_per_second"] == 16.0
    assert result.value.totals["update_seconds"] == 3.0
    point = result.value.datasets.throughput[0]
    assert point.update == 1
    assert point.values["rounds_per_second"] == 2.0
    dumped = result.value.model_dump()
    assert "session_id" not in dumped
    assert "sessions" not in dumped


def test_metrics_join_late_cross_process_events_by_rollout_id(
    tmp_path: Path,
) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)
    coordinator = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="coordinator"),
    )
    for rollout_id in ("rollout-a", "rollout-b"):
        coordinator.emit(
            "update",
            context=EventContext(
                policy_version=0, rollout_id=rollout_id
            ),
            fields={
                "total_rounds": 1,
                "total_samples": 4,
                "total_updates": 1,
            },
        )
    coordinator.close()
    worker = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="worker", index=1),
    )
    for rollout_id, latency in (
        ("rollout-b", 0.2),
        ("rollout-a", 0.1),
    ):
        worker.emit(
            "inference.batch",
            context=EventContext(
                policy_version=0,
                rollout_id=rollout_id,
                worker_index=1,
            ),
            fields={
                "batch_size": 4,
                "fill_ratio": 0.5,
                "inference_seconds": latency,
            },
        )
        worker.emit(
            "sampling",
            context=EventContext(
                policy_version=0,
                rollout_id=rollout_id,
                worker_index=1,
            ),
            fields={"completed_rounds": 2, "decision_count": 8},
        )
    worker.close()

    result = query_training_metrics(
        tmp_path, update_limit=200, series_points=200
    )

    assert isinstance(result, Ok)
    inference = result.value.datasets.inference
    assert [point.update for point in inference] == [1, 2]
    first_latency = inference[0].values["inference_seconds_avg"]
    second_latency = inference[1].values["inference_seconds_avg"]
    assert isinstance(first_latency, int | float)
    assert isinstance(second_latency, int | float)
    assert math.isclose(float(first_latency), 0.1)
    assert math.isclose(float(second_latency), 0.2)
    process = result.value.datasets.processes[0]
    assert process.values["completed_rounds"] == 4.0
    assert process.values["decision_count"] == 16.0


def test_metrics_project_live_inference_before_first_update(
    tmp_path: Path,
) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)
    sink = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="worker", index=0),
    )
    sink.emit(
        "inference.batch",
        context=EventContext(
            policy_version=0,
            rollout_id="rollout-a",
            worker_index=0,
        ),
        fields={
            "batch_size": 4,
            "fill_ratio": 0.5,
            "inference_seconds": 0.25,
        },
    )
    sink.close()

    result = query_training_metrics(
        tmp_path, update_limit=200, series_points=200
    )

    assert isinstance(result, Ok)
    assert len(result.value.datasets.inference) == 1
    point = result.value.datasets.inference[0]
    assert point.update == 1
    assert point.values["batch_size"] == 4.0
    assert point.values["inference_seconds_avg"] == 0.25


def test_metrics_cursor_advances_for_live_inference(
    tmp_path: Path,
) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)
    sink = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="model_rank", index=0),
    )
    sink.emit(
        "inference.batch",
        context=EventContext(
            policy_version=0,
            rollout_id="rollout-a",
        ),
        fields={"batch_size": 1},
    )
    sink.close()

    result = query_metrics_cursor(tmp_path)

    assert isinstance(result, Ok)
    assert result.value.through_sequence == 1


def test_metrics_snapshot_and_cursor_ignore_non_metric_events(
    tmp_path: Path,
) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)
    sink = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="worker", index=0),
    )
    sink.emit(
        "inference.batch",
        context=EventContext(policy_version=0, rollout_id="rollout-a"),
        fields={"batch_size": 1},
    )
    sink.emit(
        "decision",
        context=EventContext(policy_version=0, rollout_id="rollout-a"),
    )
    sink.close()

    snapshot = query_training_metrics(
        tmp_path, update_limit=200, series_points=200
    )
    cursor = query_metrics_cursor(tmp_path)

    assert isinstance(snapshot, Ok)
    assert isinstance(cursor, Ok)
    assert snapshot.value.through_sequence == 1
    assert cursor.value.through_sequence == 1


def test_metrics_keep_pending_rollout_logged_before_previous_update(
    tmp_path: Path,
) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)
    sink = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="coordinator"),
    )
    sink.emit(
        "inference.batch",
        context=EventContext(policy_version=1, rollout_id="rollout-b"),
        fields={"batch_size": 4},
    )
    sink.emit(
        "update",
        context=EventContext(policy_version=0, rollout_id="rollout-a"),
        fields={"total_updates": 1},
    )
    sink.close()

    result = query_training_metrics(
        tmp_path, update_limit=200, series_points=200
    )

    assert isinstance(result, Ok)
    assert [
        point.update for point in result.value.datasets.inference
    ] == [2]


def test_metrics_exclude_rollout_after_failed_terminal_update(
    tmp_path: Path,
) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)
    sink = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="coordinator"),
    )
    context = EventContext(policy_version=0, rollout_id="rollout-a")
    sink.emit(
        "inference.batch", context=context, fields={"batch_size": 4}
    )
    sink.emit("update", context=context, error="failed")
    sink.close()

    result = query_training_metrics(
        tmp_path, update_limit=200, series_points=200
    )

    assert isinstance(result, Ok)
    assert result.value.datasets.inference == ()


def test_failed_actions_are_excluded_and_drop_marks_incomplete(
    tmp_path: Path,
) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)
    sink = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="coordinator"),
    )
    context = EventContext(policy_version=0, rollout_id="rollout-a")
    sink.emit("update", context=context, error="failed")
    sink.emit("logging.drop", fields={"count": 3})
    sink.close()

    result = query_training_metrics(
        tmp_path, update_limit=200, series_points=200
    )

    assert isinstance(result, Ok)
    assert result.value.datasets.throughput == ()
    assert result.value.complete is False
    assert result.value.dropped_event_count == 3


def test_missing_database_is_empty(tmp_path: Path) -> None:
    result = query_training_metrics(
        tmp_path, update_limit=200, series_points=200
    )

    assert isinstance(result, Ok)
    assert result.value.store_id is None
    assert result.value.through_sequence == 0
