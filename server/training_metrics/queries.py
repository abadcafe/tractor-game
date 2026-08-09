"""Sequence-ordered SQLite projections for the training metrics view."""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pydantic import (
    ConfigDict,
    TypeAdapter,
    ValidationError,
)

from server.foundation import result as _result
from server.foundation.json_value import JsonObject
from server.training_events.store import open_reader, training_store_id
from server.training_metrics.contract import (
    TRAINING_METRICS_SCHEMA_VERSION,
    MetricDatasets,
    MetricPoint,
    MetricsCursor,
    TrainingMetrics,
    TrainingMetricSummary,
)

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)
type _SqlRow = tuple[object, ...]
_SQL_ROWS: TypeAdapter[list[_SqlRow]] = TypeAdapter(
    list[_SqlRow],
    config=ConfigDict(strict=True),
)
_OPTIONAL_SQL_ROW: TypeAdapter[_SqlRow | None] = TypeAdapter(
    _SqlRow | None,
    config=ConfigDict(strict=True),
)
_METRICS_CURSOR_QUERY = (
    "SELECT coalesce(max(sequence), 0) FROM training_logs "
    "WHERE event_type IN "
    "('update', 'training', 'logging.drop', 'round', "
    "'sampling', 'inference.batch')"
)


@dataclass(frozen=True, slots=True)
class _Event:
    sequence: int
    recorded_at_ms: int
    process_index: int | None
    policy_version: int | None
    context: JsonObject
    fields: JsonObject


def query_training_metric_summary(
    run_dir: Path,
) -> _result.Ok[TrainingMetricSummary] | _result.Rejected:
    """Read latest scalar metrics without projecting chart series."""
    opened = open_reader(run_dir)
    if isinstance(opened, _result.Rejected):
        return opened
    connection = opened.value
    if connection is None:
        return _result.Ok(value=_empty_metric_summary())
    try:
        _ = connection.execute("BEGIN")
        store_id = training_store_id(connection)
        through_sequence = _metrics_through_sequence(connection)
        dropped = _scalar_int(
            connection,
            "SELECT coalesce(sum(json_extract(event_json, "
            + "'$.fields.count')), 0) FROM training_logs "
            + "WHERE event_type = 'logging.drop'",
        )
        latest_update = _latest_event(connection, event_type="update")
        connection.commit()
    except sqlite3.Error, ValidationError, ValueError:
        connection.rollback()
        return _result.Rejected(
            reason="training metric summary query failed"
        )
    finally:
        connection.close()
    return _result.Ok(
        value=TrainingMetricSummary(
            store_id=store_id,
            through_sequence=through_sequence,
            complete=dropped == 0,
            dropped_event_count=dropped,
            totals={}
            if latest_update is None
            else _metric_totals((latest_update,)),
        )
    )


def query_training_metrics(
    run_dir: Path,
    *,
    update_limit: int,
    series_points: int,
) -> _result.Ok[TrainingMetrics] | _result.Rejected:
    """Compute a sequence-ordered full-run metrics snapshot."""
    if update_limit <= 0 or update_limit > 5000:
        return _result.Rejected(
            reason="update_limit must be between 1 and 5000"
        )
    if series_points <= 0 or series_points > 1000:
        return _result.Rejected(
            reason="series_points must be between 1 and 1000"
        )
    opened = open_reader(run_dir)
    if isinstance(opened, _result.Rejected):
        return opened
    connection = opened.value
    if connection is None:
        return _result.Ok(value=_empty_metrics())
    try:
        _ = connection.execute("BEGIN")
        store_id = training_store_id(connection)
        through_sequence = _metrics_through_sequence(connection)
        started_at_ms = _scalar_int(
            connection,
            "SELECT coalesce(min(recorded_at_ms), 0) "
            + "FROM training_logs",
        )
        dropped = _scalar_int(
            connection,
            "SELECT coalesce(sum(json_extract(event_json, "
            + "'$.fields.count')), 0) FROM training_logs "
            + "WHERE event_type = 'logging.drop'",
        )
        all_updates = _events(connection, event_type="update")
        recent_updates = all_updates[-update_limit:]
        update_ordinals = {
            event.sequence: ordinal
            for ordinal, event in enumerate(all_updates, start=1)
        }
        selected_rollout_ids = {
            rollout_id
            for event in recent_updates
            if isinstance(
                rollout_id := event.context.get("rollout_id"), str
            )
        }
        selected_rollout_ids.update(_pending_rollout_ids(connection))
        selected_policy_versions = {
            policy_version
            for event in recent_updates
            if (policy_version := event.policy_version) is not None
        }
        selected_policy_versions.update(
            _pending_policy_versions(connection)
        )
        round_events = _events(
            connection,
            event_type="round",
            rollout_ids=selected_rollout_ids,
        )
        inference_events = _events(
            connection,
            event_type="inference.batch",
            policy_versions=selected_policy_versions,
        )
        sampling_events = _events(
            connection,
            event_type="sampling",
            rollout_ids=selected_rollout_ids,
        )
        connection.commit()
    except sqlite3.Error, ValidationError, ValueError:
        connection.rollback()
        return _result.Rejected(reason="training metrics query failed")
    finally:
        connection.close()

    update_points = _update_points(
        recent_updates,
        update_ordinals=update_ordinals,
        started_at_ms=started_at_ms,
        series_points=series_points,
    )
    rollout_ordinals = {
        rollout_id: update_ordinals[event.sequence]
        for event in recent_updates
        if isinstance(
            rollout_id := event.context.get("rollout_id"), str
        )
    }
    round_points = _event_points(
        round_events,
        rollout_ordinals=rollout_ordinals,
        started_at_ms=started_at_ms,
        series_points=series_points,
    )
    inference_points = _inference_points(
        inference_events,
        rollout_ordinals=rollout_ordinals,
        started_at_ms=started_at_ms,
        series_points=series_points,
    )
    process_points = _process_points(
        sampling_events,
        rollout_ordinals=rollout_ordinals,
        started_at_ms=started_at_ms,
    )
    totals = _metric_totals(recent_updates)
    return _result.Ok(
        value=TrainingMetrics(
            schema_version=TRAINING_METRICS_SCHEMA_VERSION,
            store_id=store_id,
            through_sequence=through_sequence,
            complete=dropped == 0,
            dropped_event_count=dropped,
            totals=totals,
            datasets=MetricDatasets(
                throughput=update_points,
                optimization=update_points,
                ppo_timing=update_points,
                rounds=round_points,
                rewards=round_points,
                inference=inference_points,
                processes=process_points,
            ),
        )
    )


def query_metrics_cursor(
    run_dir: Path,
) -> _result.Ok[MetricsCursor] | _result.Rejected:
    """Return the newest event cursor represented by Metrics."""
    opened = open_reader(run_dir)
    if isinstance(opened, _result.Rejected):
        return opened
    connection = opened.value
    if connection is None:
        return _result.Ok(
            value=MetricsCursor(store_id=None, through_sequence=0)
        )
    try:
        store_id = training_store_id(connection)
        through_sequence = _metrics_through_sequence(connection)
    except sqlite3.Error:
        return _result.Rejected(
            reason="training metrics revision query failed"
        )
    finally:
        connection.close()
    return _result.Ok(
        value=MetricsCursor(
            store_id=store_id, through_sequence=through_sequence
        )
    )


def _events(
    connection: sqlite3.Connection,
    *,
    event_type: str,
    rollout_ids: set[str] | None = None,
    policy_versions: set[int] | None = None,
) -> tuple[_Event, ...]:
    conditions = [
        "event_type = ?",
        "json_type(event_json, '$.error') IS NULL",
    ]
    parameters: list[str | int] = [event_type]
    if rollout_ids is not None:
        if not rollout_ids:
            return ()
        placeholders = ", ".join("?" for _item in rollout_ids)
        conditions.append(f"rollout_id IN ({placeholders})")
        parameters.extend(sorted(rollout_ids))
    if policy_versions is not None:
        if not policy_versions:
            return ()
        placeholders = ", ".join("?" for _item in policy_versions)
        conditions.append(f"policy_version IN ({placeholders})")
        parameters.extend(sorted(policy_versions))
    rows = _SQL_ROWS.validate_python(
        connection.execute(
            "SELECT sequence, recorded_at_ms, process_index, "
            + "policy_version, event_json FROM training_logs WHERE "
            + " AND ".join(conditions)
            + " ORDER BY sequence",
            parameters,
        ).fetchall()
    )
    return tuple(_event_from_row(row) for row in rows)


def _latest_event(
    connection: sqlite3.Connection, *, event_type: str
) -> _Event | None:
    row = _OPTIONAL_SQL_ROW.validate_python(
        connection.execute(
            "SELECT sequence, recorded_at_ms, process_index, "
            + "policy_version, event_json FROM training_logs "
            + "WHERE event_type = ? "
            + "AND json_type(event_json, '$.error') IS NULL "
            + "ORDER BY sequence DESC LIMIT 1",
            (event_type,),
        ).fetchone()
    )
    return None if row is None else _event_from_row(row)


def _event_from_row(row: tuple[object, ...]) -> _Event:
    if len(row) != 5:
        raise ValueError("invalid event row")
    sequence, recorded_at_ms, process_index, policy_version, raw = row
    if not isinstance(sequence, int) or not isinstance(
        recorded_at_ms, int
    ):
        raise ValueError("invalid event dimensions")
    if process_index is not None and not isinstance(process_index, int):
        raise ValueError("invalid process index")
    if policy_version is not None and not isinstance(
        policy_version, int
    ):
        raise ValueError("invalid policy version")
    if not isinstance(raw, str):
        raise ValueError("invalid event json")
    payload = _JSON_OBJECT_ADAPTER.validate_json(raw)
    context = payload.get("context")
    fields = payload.get("fields")
    if not isinstance(context, dict) or not isinstance(fields, dict):
        raise ValueError("invalid event body")
    return _Event(
        sequence=sequence,
        recorded_at_ms=recorded_at_ms,
        process_index=process_index,
        policy_version=policy_version,
        context=context,
        fields=fields,
    )


def _pending_rollout_ids(connection: sqlite3.Connection) -> set[str]:
    rows = _SQL_ROWS.validate_python(
        connection.execute(
            "SELECT DISTINCT source.rollout_id "
            + "FROM training_logs AS source "
            + "WHERE source.event_type IN "
            + "('round', 'sampling') "
            + "AND source.rollout_id IS NOT NULL "
            + "AND json_type(source.event_json, '$.error') IS NULL "
            + "AND NOT EXISTS ("
            + "SELECT 1 FROM training_logs AS terminal "
            + "WHERE terminal.rollout_id = source.rollout_id "
            + "AND terminal.event_type = 'update'"
            + ")",
        ).fetchall()
    )
    rollout_ids: set[str] = set()
    for row in rows:
        rollout_id = row[0]
        if not isinstance(rollout_id, str):
            raise ValueError("invalid pending rollout id")
        rollout_ids.add(rollout_id)
    return rollout_ids


def _pending_policy_versions(
    connection: sqlite3.Connection,
) -> set[int]:
    rows = _SQL_ROWS.validate_python(
        connection.execute(
            "SELECT DISTINCT source.policy_version "
            + "FROM training_logs AS source "
            + "WHERE source.event_type IN "
            + "('round', 'sampling', 'inference.batch') "
            + "AND source.policy_version IS NOT NULL "
            + "AND json_type(source.event_json, '$.error') IS NULL "
            + "AND NOT EXISTS ("
            + "SELECT 1 FROM training_logs AS terminal "
            + "WHERE terminal.policy_version = source.policy_version "
            + "AND terminal.event_type = 'update'"
            + ")",
        ).fetchall()
    )
    policy_versions: set[int] = set()
    for row in rows:
        policy_version = row[0]
        if not isinstance(policy_version, int):
            raise ValueError("invalid pending policy version")
        policy_versions.add(policy_version)
    return policy_versions


def _update_points(
    events: tuple[_Event, ...],
    *,
    update_ordinals: dict[int, int],
    started_at_ms: int,
    series_points: int,
) -> tuple[MetricPoint, ...]:
    groups = _bucket_groups(events, series_points)
    points: list[MetricPoint] = []
    for group in groups:
        latest = group[-1]
        values = _average_fields(group)
        for name in (
            "total_rounds",
            "total_trainable_decisions",
            "total_updates",
        ):
            if name in latest.fields:
                values[name] = latest.fields[name]
        _alias_rate(
            values,
            source="process_rounds_per_second",
            target="rounds_per_second",
        )
        _alias_rate(
            values,
            source="process_trainable_decisions_per_second",
            target="trainable_decisions_per_second",
        )
        points.append(
            _point(
                latest,
                update=update_ordinals[latest.sequence],
                started_at_ms=started_at_ms,
                values=values,
            )
        )
    return tuple(points)


def _event_points(
    events: tuple[_Event, ...],
    *,
    rollout_ordinals: dict[str, int],
    started_at_ms: int,
    series_points: int,
) -> tuple[MetricPoint, ...]:
    points: list[MetricPoint] = []
    for group in _bucket_groups(events, series_points):
        latest = group[-1]
        ordinal = _rollout_ordinal(latest, rollout_ordinals)
        if ordinal is None:
            continue
        points.append(
            _point(
                latest,
                update=ordinal,
                started_at_ms=started_at_ms,
                values=_average_fields(group),
            )
        )
    return tuple(points)


def _inference_points(
    events: tuple[_Event, ...],
    *,
    rollout_ordinals: dict[str, int],
    started_at_ms: int,
    series_points: int,
) -> tuple[MetricPoint, ...]:
    by_update: dict[int, list[_Event]] = defaultdict(list)
    for event in events:
        ordinal = _rollout_ordinal(event, rollout_ordinals)
        if ordinal is not None:
            by_update[ordinal].append(event)
    groups = _bucket_groups(
        tuple(by_update[ordinal][-1] for ordinal in sorted(by_update)),
        series_points,
    )
    points: list[MetricPoint] = []
    for bucket in groups:
        bucket_events: list[_Event] = []
        for marker in bucket:
            ordinal = _rollout_ordinal(marker, rollout_ordinals)
            assert ordinal is not None
            bucket_events.extend(by_update[ordinal])
        latest = bucket_events[-1]
        latest_ordinal = _rollout_ordinal(latest, rollout_ordinals)
        assert latest_ordinal is not None
        values: JsonObject = {}
        for source, target in (
            ("batch_size", "batch_size"),
            ("fill_ratio", "fill_ratio"),
            ("batch_wait_seconds", "batch_wait_seconds"),
            ("recv_seconds", "recv_seconds"),
            ("h2d_seconds", "h2d_seconds"),
            ("device_decode_seconds", "decode_seconds"),
            ("inference_seconds", "inference_seconds"),
        ):
            samples = _numeric_field_values(bucket_events, source)
            values[target + "_avg"] = _mean(samples)
            if target not in ("batch_size", "fill_ratio"):
                values[target + "_p95"] = _percentile_95(samples)
        values["batch_size"] = values.pop("batch_size_avg")
        values["fill_ratio"] = values.pop("fill_ratio_avg")
        points.append(
            _point(
                latest,
                update=latest_ordinal,
                started_at_ms=started_at_ms,
                values=values,
            )
        )
    return tuple(points)


def _process_points(
    events: tuple[_Event, ...],
    *,
    rollout_ordinals: dict[str, int],
    started_at_ms: int,
) -> tuple[MetricPoint, ...]:
    by_process: dict[int, list[_Event]] = defaultdict(list)
    for event in events:
        if (
            event.process_index is not None
            and _rollout_ordinal(event, rollout_ordinals) is not None
        ):
            by_process[event.process_index].append(event)
    points: list[MetricPoint] = []
    for process_index in sorted(by_process):
        group = by_process[process_index]
        latest = group[-1]
        latest_ordinal = _rollout_ordinal(latest, rollout_ordinals)
        assert latest_ordinal is not None
        values: JsonObject = {"worker_index": process_index}
        for name in (
            "completed_rounds",
            "policy_request_count",
            "policy_wait_seconds",
            "round_seconds",
        ):
            values[name] = sum(_numeric_field_values(group, name))
        points.append(
            _point(
                latest,
                update=latest_ordinal,
                started_at_ms=started_at_ms,
                values=values,
            )
        )
    return tuple(points)


def _bucket_groups(
    events: tuple[_Event, ...], maximum: int
) -> tuple[tuple[_Event, ...], ...]:
    if not events:
        return ()
    count = min(len(events), maximum)
    buckets: list[list[_Event]] = [[] for _item in range(count)]
    for index, event in enumerate(events):
        bucket = min(index * count // len(events), count - 1)
        buckets[bucket].append(event)
    return tuple(tuple(bucket) for bucket in buckets)


def _average_fields(events: tuple[_Event, ...]) -> JsonObject:
    names = {name for event in events for name in event.fields}
    values: JsonObject = {}
    for name in names:
        samples = _numeric_field_values(events, name)
        if samples:
            values[name] = _mean(samples)
        else:
            latest = events[-1].fields.get(name)
            if _is_json_value(latest):
                values[name] = latest
    return values


def _numeric_field_values(
    events: list[_Event] | tuple[_Event, ...], name: str
) -> list[float]:
    values: list[float] = []
    for event in events:
        value = event.fields.get(name)
        if isinstance(value, int | float) and not isinstance(
            value, bool
        ):
            values.append(float(value))
    return values


def _point(
    event: _Event,
    *,
    update: int,
    started_at_ms: int,
    values: JsonObject,
) -> MetricPoint:
    return MetricPoint(
        sequence=event.sequence,
        update=update,
        elapsed_seconds=max(
            (event.recorded_at_ms - started_at_ms) / 1000.0, 0.0
        ),
        recorded_at_ms=event.recorded_at_ms,
        values=values,
    )


def _rollout_ordinal(
    event: _Event, rollout_ordinals: dict[str, int]
) -> int | None:
    rollout_id = event.context.get("rollout_id")
    if isinstance(rollout_id, str):
        ordinal = rollout_ordinals.get(rollout_id)
        if ordinal is not None:
            return ordinal
    if event.policy_version is None:
        return None
    return event.policy_version + 1


def _alias_rate(
    values: JsonObject, *, source: str, target: str
) -> None:
    if source in values:
        values[target] = values[source]


def _metric_totals(events: tuple[_Event, ...]) -> JsonObject:
    if not events:
        return {}
    values = dict(events[-1].fields)
    _alias_rate(
        values,
        source="process_rounds_per_second",
        target="rounds_per_second",
    )
    _alias_rate(
        values,
        source="process_trainable_decisions_per_second",
        target="trainable_decisions_per_second",
    )
    if "update_seconds" not in values:
        for source in (
            "ppo_update_seconds",
            "update_cycle_seconds",
            "duration_seconds",
        ):
            if source in values:
                values["update_seconds"] = values[source]
                break
    return values


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(math.ceil(len(ordered) * 0.95) - 1, 0)]


def _is_json_value(value: object) -> bool:
    return value is None or isinstance(
        value, str | int | float | bool | list | dict
    )


def _scalar_int(connection: sqlite3.Connection, query: str) -> int:
    row = _OPTIONAL_SQL_ROW.validate_python(
        connection.execute(query).fetchone()
    )
    if row is None or not isinstance(row[0], int):
        raise ValueError("metric scalar is invalid")
    return row[0]


def _metrics_through_sequence(connection: sqlite3.Connection) -> int:
    return _scalar_int(connection, _METRICS_CURSOR_QUERY)


def _empty_metrics(*, through_sequence: int = 0) -> TrainingMetrics:
    return TrainingMetrics(
        schema_version=TRAINING_METRICS_SCHEMA_VERSION,
        store_id=None,
        through_sequence=through_sequence,
        complete=True,
        dropped_event_count=0,
        totals={},
        datasets=MetricDatasets(
            throughput=(),
            optimization=(),
            ppo_timing=(),
            rounds=(),
            rewards=(),
            inference=(),
            processes=(),
        ),
    )


def _empty_metric_summary() -> TrainingMetricSummary:
    return TrainingMetricSummary(
        store_id=None,
        through_sequence=0,
        complete=True,
        dropped_event_count=0,
        totals={},
    )
