"""Black-box tests for SQLite-derived dashboard metrics."""

from __future__ import annotations

import math
from pathlib import Path

from server.foundation.result import Ok
from server.training.event_log import (
    EventContext,
    ProcessIdentity,
    StructuredEventSink,
)
from server.training.persistence.schema import initialize_database
from server.training_control.metric_queries import (
    query_training_metrics,
)


def test_metric_query_selects_latest_session_and_chart_series(
    tmp_path: Path,
) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)
    sink = StructuredEventSink(
        run_dir=tmp_path,
        session_id="session-1",
        process=ProcessIdentity(kind="coordinator"),
    )
    sink.emit("session.started")
    sink.emit(
        "update.completed",
        context=EventContext(policy_version=0),
        fields={
            "total_rounds": 4,
            "total_samples": 32,
            "total_updates": 1,
            "process_rounds_per_second": 2.0,
            "process_samples_per_second": 16.0,
            "rollout_decisions_per_second": 20.0,
            "policy_loss": 0.25,
            "value_loss": 0.5,
        },
    )
    sink.close()

    result = query_training_metrics(
        tmp_path,
        session_id=None,
        update_limit=200,
        series_points=200,
    )

    assert isinstance(result, Ok)
    assert result.value.session_id == "session-1"
    assert result.value.totals["total_samples"] == 32
    assert len(result.value.datasets.throughput) == 1
    assert result.value.complete is True
    assert result.value.sessions[0].session_id == "session-1"


def test_metric_query_aggregates_inference_percentiles_and_workers(
    tmp_path: Path,
) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)
    coordinator = StructuredEventSink(
        run_dir=tmp_path,
        session_id="session-aggregate",
        process=ProcessIdentity(kind="coordinator", index=0),
    )
    coordinator.emit(
        "session.started",
        fields={
            "total_rounds": 0,
            "total_samples": 0,
            "total_updates": 0,
        },
    )
    for latency in range(1, 21):
        coordinator.emit(
            "inference.batch_completed",
            context=EventContext(policy_version=0),
            fields={
                "batch_size": 4,
                "fill_ratio": 0.5,
                "recv_seconds": latency / 1000.0,
                "h2d_seconds": latency / 2000.0,
                "device_decode_seconds": latency / 3000.0,
                "inference_seconds": latency / 100.0,
            },
        )
    coordinator.close()
    worker = StructuredEventSink(
        run_dir=tmp_path,
        session_id="session-aggregate",
        process=ProcessIdentity(kind="worker", index=1),
    )
    worker.emit(
        "sampling.completed",
        context=EventContext(policy_version=0, worker_index=1),
        fields={
            "completed_rounds": 5,
            "decision_count": 40,
            "policy_wait_seconds": 2.0,
            "round_seconds": 8.0,
        },
    )
    worker.close()

    result = query_training_metrics(
        tmp_path,
        session_id="session-aggregate",
        update_limit=200,
        series_points=200,
    )

    assert isinstance(result, Ok)
    inference = result.value.datasets.inference[0].values
    inference_average = inference["inference_seconds_avg"]
    inference_p95 = inference["inference_seconds_p95"]
    assert isinstance(inference_average, float)
    assert isinstance(inference_p95, float)
    assert math.isclose(inference_average, 0.105)
    assert math.isclose(inference_p95, 0.19)
    process = result.value.datasets.processes[0].values
    assert process["worker_index"] == 1
    assert process["completed_rounds"] == 5
    assert process["decision_count"] == 40


def test_metric_query_rejects_unknown_session(tmp_path: Path) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)

    result = query_training_metrics(
        tmp_path,
        session_id="missing",
        update_limit=200,
        series_points=200,
    )

    assert not isinstance(result, Ok)
