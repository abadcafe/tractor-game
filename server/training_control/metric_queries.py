"""Read-only SQLite projections for the training metrics dashboard."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from server.foundation import result as _result
from server.foundation.json_value import JsonObject
from server.training_control.database import open_training_database


class MetricPoint(BaseModel):
    """One server-computed chart point."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    update: int | None = Field(default=None, ge=0)
    elapsed_seconds: float = Field(ge=0.0)
    recorded_at_ms: int = Field(ge=0)
    values: JsonObject


class MetricDatasets(BaseModel):
    """Chart-oriented metric series."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    throughput: tuple[MetricPoint, ...]
    optimization: tuple[MetricPoint, ...]
    ppo_timing: tuple[MetricPoint, ...]
    rollout: tuple[MetricPoint, ...]
    rewards: tuple[MetricPoint, ...]
    inference: tuple[MetricPoint, ...]
    processes: tuple[MetricPoint, ...]


class MetricSession(BaseModel):
    """One append-only resume session available for analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    session_id: str
    started_at_ms: int = Field(ge=0)


class TrainingMetrics(BaseModel):
    """Consistent read snapshot consumed by the Metrics SPA view."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: int = 1
    through_sequence: int = Field(ge=0)
    session_id: str | None
    sessions: tuple[MetricSession, ...]
    complete: bool
    dropped_event_count: int = Field(ge=0)
    totals: JsonObject
    datasets: MetricDatasets


def query_training_metrics(
    run_dir: Path,
    *,
    session_id: str | None,
    update_limit: int,
    series_points: int,
) -> _result.Ok[TrainingMetrics] | _result.Rejected:
    """Compute all dashboard datasets in one read transaction."""
    if update_limit <= 0 or update_limit > 5000:
        return _result.Rejected(
            reason="update_limit must be between 1 and 5000"
        )
    if series_points <= 0 or series_points > 1000:
        return _result.Rejected(
            reason="series_points must be between 1 and 1000"
        )
    opened = open_training_database(run_dir)
    if isinstance(opened, _result.Rejected):
        return opened
    connection = opened.value
    if connection is None:
        return _result.Ok(value=_empty_metrics())
    try:
        connection.execute("BEGIN")
        selected_session = session_id or _latest_session(connection)
        through_sequence = _scalar_int(
            connection,
            "SELECT coalesce(max(sequence), 0) FROM training_logs",
        )
        sessions = _sessions(connection)
        if selected_session is not None and all(
            item.session_id != selected_session for item in sessions
        ):
            connection.rollback()
            return _result.Rejected(
                reason="training metrics session does not exist"
            )
        if selected_session is None:
            connection.commit()
            return _result.Ok(
                value=_empty_metrics(through_sequence=through_sequence)
            )
        started_at_ms = _session_started_at(
            connection, selected_session
        )
        dropped = _dropped_event_count(connection, selected_session)
        update_points = _update_points(
            connection,
            session_id=selected_session,
            started_at_ms=started_at_ms,
            update_limit=update_limit,
            series_points=series_points,
        )
        rollout_points = _aggregate_event_points(
            connection,
            session_id=selected_session,
            event_type="rollout.completed",
            started_at_ms=started_at_ms,
            update_limit=update_limit,
            series_points=series_points,
            value_paths=(
                ("round_count", "$.fields.round_count"),
                ("sample_count", "$.fields.sample_count"),
                ("decision_count", "$.fields.sample_count"),
                (
                    "generated_action_count",
                    "$.fields.generated_action_count",
                ),
                (
                    "accepted_action_count",
                    "$.fields.accepted_action_count",
                ),
                ("action_choice_count", "$.fields.action_choice_count"),
                (
                    "dropped_sample_count",
                    "$.fields.dropped_sample_count",
                ),
                ("cancelled_env_count", "$.fields.cancelled_env_count"),
            ),
        )
        reward_points = _aggregate_event_points(
            connection,
            session_id=selected_session,
            event_type="rollout.completed",
            started_at_ms=started_at_ms,
            update_limit=update_limit,
            series_points=series_points,
            value_paths=(
                ("team0_reward", "$.fields.team0_reward"),
                ("team1_reward", "$.fields.team1_reward"),
                ("game_over_count", "$.fields.game_over_count"),
                ("round_count", "$.fields.round_count"),
            ),
        )
        inference_points = _inference_points(
            connection,
            session_id=selected_session,
            started_at_ms=started_at_ms,
            update_limit=update_limit,
            series_points=series_points,
        )
        process_points = _process_points(
            connection,
            session_id=selected_session,
            started_at_ms=started_at_ms,
            update_limit=update_limit,
        )
        connection.commit()
    except sqlite3.Error:
        connection.rollback()
        return _result.Rejected(reason="training metrics query failed")
    finally:
        connection.close()
    totals = {} if not update_points else update_points[-1].values
    return _result.Ok(
        value=TrainingMetrics(
            through_sequence=through_sequence,
            session_id=selected_session,
            sessions=sessions,
            complete=dropped == 0,
            dropped_event_count=dropped,
            totals=totals,
            datasets=MetricDatasets(
                throughput=update_points,
                optimization=update_points,
                ppo_timing=update_points,
                rollout=rollout_points,
                rewards=reward_points,
                inference=inference_points,
                processes=process_points,
            ),
        )
    )


def _latest_session(connection: sqlite3.Connection) -> str | None:
    row = connection.execute(
        "SELECT session_id FROM training_logs "
        "WHERE event_type = 'session.started' "
        "ORDER BY sequence DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    value = row[0]
    assert isinstance(value, str)
    return value


def _sessions(
    connection: sqlite3.Connection,
) -> tuple[MetricSession, ...]:
    rows = connection.execute(
        "SELECT session_id, recorded_at_ms FROM training_logs "
        "WHERE event_type = 'session.started' ORDER BY sequence DESC"
    ).fetchall()
    sessions: list[MetricSession] = []
    for session_id, started_at_ms in rows:
        assert isinstance(session_id, str)
        assert isinstance(started_at_ms, int)
        sessions.append(
            MetricSession(
                session_id=session_id,
                started_at_ms=started_at_ms,
            )
        )
    return tuple(sessions)


def _session_started_at(
    connection: sqlite3.Connection, session_id: str
) -> int:
    row = connection.execute(
        "SELECT recorded_at_ms FROM training_logs "
        "WHERE session_id = ? AND event_type = 'session.started' "
        "ORDER BY sequence LIMIT 1",
        (session_id,),
    ).fetchone()
    if row is None:
        return 0
    value = row[0]
    assert isinstance(value, int)
    return value


def _dropped_event_count(
    connection: sqlite3.Connection, session_id: str
) -> int:
    row = connection.execute(
        "SELECT coalesce(sum(json_extract(event_json, "
        "'$.fields.count')), 0) "
        "FROM training_logs WHERE session_id = ? "
        "AND event_type = 'logging.dropped'",
        (session_id,),
    ).fetchone()
    assert row is not None
    value = row[0]
    assert isinstance(value, int)
    return value


def _update_points(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    started_at_ms: int,
    update_limit: int,
    series_points: int,
) -> tuple[MetricPoint, ...]:
    rows = connection.execute(
        """
        WITH session_start AS (
            SELECT
                recorded_at_ms,
                json_extract(event_json, '$.fields.total_rounds')
                    AS total_rounds,
                json_extract(event_json, '$.fields.total_samples')
                    AS total_samples
            FROM training_logs
            WHERE session_id = ? AND event_type = 'session.started'
            ORDER BY sequence LIMIT 1
        ),
        recent AS (
            SELECT * FROM (
                SELECT
                    sequence,
                    recorded_at_ms,
                    json_extract(event_json, '$.fields.total_rounds')
                        AS total_rounds,
                    json_extract(event_json, '$.fields.total_samples')
                        AS total_samples,
                    json_extract(event_json, '$.fields.total_updates')
                        AS total_updates,
                    json_extract(
                        event_json,
                        '$.fields.rollout_decisions_per_second'
                    ) AS decisions_per_second,
                    json_extract(event_json, '$.fields.policy_loss')
                        AS policy_loss,
                    json_extract(event_json, '$.fields.value_loss')
                        AS value_loss,
                    json_extract(event_json, '$.fields.entropy')
                        AS entropy,
                    json_extract(event_json, '$.fields.approx_kl')
                        AS approx_kl,
                    json_extract(event_json, '$.fields.clip_fraction')
                        AS clip_fraction,
                    json_extract(
                        event_json,
                        '$.fields.ppo_update_seconds'
                    ) AS update_seconds,
                    json_extract(
                        event_json,
                        '$.fields.ppo_observation_batch_seconds'
                    ) AS observation_batch_seconds,
                    json_extract(
                        event_json,
                        '$.fields.ppo_observation_encode_seconds'
                    ) AS observation_encode_seconds,
                    json_extract(
                        event_json,
                        '$.fields.ppo_value_head_seconds'
                    ) AS value_head_seconds,
                    json_extract(
                        event_json,
                        '$.fields.ppo_argument_select_seconds'
                    ) AS argument_select_seconds,
                    json_extract(
                        event_json,
                        '$.fields.ppo_argument_decode_seconds'
                    ) AS argument_decode_seconds,
                    json_extract(
                        event_json,
                        '$.fields.ppo_argument_distribution_seconds'
                    ) AS argument_distribution_seconds,
                    json_extract(
                        event_json,
                        '$.fields.ppo_backward_seconds'
                    ) AS backward_seconds,
                    json_extract(
                        event_json,
                        '$.fields.ppo_optimizer_step_seconds'
                    ) AS optimizer_step_seconds
                FROM training_logs
                WHERE session_id = ? AND event_type = 'update.completed'
                ORDER BY sequence DESC LIMIT ?
            ) ORDER BY sequence
        ),
        intervals AS (
            SELECT
                *,
                total_rounds - lag(
                    total_rounds,
                    1,
                    coalesce(
                        (SELECT total_rounds FROM session_start), 0
                    )
                ) OVER (ORDER BY sequence) AS round_delta,
                total_samples - lag(
                    total_samples,
                    1,
                    coalesce(
                        (SELECT total_samples FROM session_start), 0
                    )
                ) OVER (ORDER BY sequence) AS sample_delta,
                recorded_at_ms - lag(
                    recorded_at_ms,
                    1,
                    coalesce(
                        (SELECT recorded_at_ms FROM session_start),
                        recorded_at_ms
                    )
                ) OVER (ORDER BY sequence) AS elapsed_ms
            FROM recent
        ),
        bounded AS (
            SELECT * FROM (
                SELECT * FROM intervals
                ORDER BY sequence DESC LIMIT ?
            ) ORDER BY sequence
        ),
        numbered AS (
            SELECT
                *,
                CAST(
                    ((row_number() OVER (ORDER BY sequence) - 1) * ?)
                    / count(*) OVER () AS INTEGER
                ) AS bucket
            FROM bounded
        )
        SELECT
            max(recorded_at_ms),
            max(total_updates),
            max(total_rounds),
            max(total_samples),
            max(total_updates),
            CASE WHEN sum(elapsed_ms) > 0
                THEN sum(round_delta) * 1000.0 / sum(elapsed_ms)
                ELSE NULL END,
            CASE WHEN sum(elapsed_ms) > 0
                THEN sum(sample_delta) * 1000.0 / sum(elapsed_ms)
                ELSE NULL END,
            avg(decisions_per_second),
            avg(policy_loss),
            avg(value_loss),
            avg(entropy),
            avg(approx_kl),
            avg(clip_fraction),
            avg(update_seconds),
            avg(observation_batch_seconds),
            avg(observation_encode_seconds),
            avg(value_head_seconds),
            avg(argument_select_seconds),
            avg(argument_decode_seconds),
            avg(argument_distribution_seconds),
            avg(backward_seconds),
            avg(optimizer_step_seconds)
        FROM numbered
        GROUP BY bucket
        ORDER BY bucket
        """,
        (
            session_id,
            session_id,
            update_limit + 1,
            update_limit,
            series_points,
        ),
    ).fetchall()
    value_names = (
        "total_rounds",
        "total_samples",
        "total_updates",
        "rounds_per_second",
        "samples_per_second",
        "decisions_per_second",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "update_seconds",
        "observation_batch_seconds",
        "observation_encode_seconds",
        "value_head_seconds",
        "argument_select_seconds",
        "argument_decode_seconds",
        "argument_distribution_seconds",
        "backward_seconds",
        "optimizer_step_seconds",
    )
    return _rows_to_points(
        rows, started_at_ms=started_at_ms, value_names=value_names
    )


def _aggregate_event_points(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    event_type: str,
    started_at_ms: int,
    update_limit: int,
    series_points: int,
    value_paths: tuple[tuple[str, str], ...],
) -> tuple[MetricPoint, ...]:
    extracts = ", ".join(
        f"avg(json_extract(event_json, '{path}'))"
        for _, path in value_paths
    )
    rows = connection.execute(
        "WITH bounded AS (SELECT * FROM (SELECT sequence, event_json, "
        "recorded_at_ms, policy_version FROM training_logs "
        "WHERE session_id = ? AND event_type = ? "
        "ORDER BY sequence DESC LIMIT ?) ORDER BY sequence), "
        "numbered AS (SELECT *, CAST(((row_number() OVER "
        "(ORDER BY sequence) - 1) * ?) / count(*) OVER () AS INTEGER) "
        "AS bucket FROM bounded) SELECT max(recorded_at_ms), "
        "max(policy_version)"
        + ("" if not extracts else ", " + extracts)
        + " FROM numbered GROUP BY bucket ORDER BY bucket",
        (session_id, event_type, update_limit, series_points),
    ).fetchall()
    return _rows_to_points(
        rows,
        started_at_ms=started_at_ms,
        value_names=tuple(name for name, _path in value_paths),
    )


def _inference_points(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    started_at_ms: int,
    update_limit: int,
    series_points: int,
) -> tuple[MetricPoint, ...]:
    rows = connection.execute(
        """
        WITH inferred AS (
            SELECT
                l.sequence,
                l.recorded_at_ms,
                coalesce(
                    l.policy_version,
                    (
                        SELECT max(r.policy_version)
                        FROM training_logs AS r
                        WHERE r.session_id = l.session_id
                            AND r.event_type = 'rollout.started'
                            AND r.sequence <= l.sequence
                    )
                ) AS update_version,
                json_extract(l.event_json, '$.fields.batch_size')
                    AS batch_size,
                json_extract(l.event_json, '$.fields.fill_ratio')
                    AS fill_ratio,
                json_extract(l.event_json, '$.fields.recv_seconds')
                    AS recv_seconds,
                json_extract(l.event_json, '$.fields.h2d_seconds')
                    AS h2d_seconds,
                json_extract(
                    l.event_json,
                    '$.fields.device_decode_seconds'
                ) AS decode_seconds,
                json_extract(l.event_json, '$.fields.inference_seconds')
                    AS inference_seconds
            FROM training_logs AS l
            WHERE l.session_id = ?
                AND l.event_type = 'inference.batch_completed'
        ),
        selected_updates AS (
            SELECT update_version
            FROM inferred
            WHERE update_version IS NOT NULL
            GROUP BY update_version
            ORDER BY update_version DESC
            LIMIT ?
        ),
        update_numbers AS (
            SELECT
                update_version,
                CAST(
                    ((
                        row_number() OVER (ORDER BY update_version) - 1
                    ) * ?)
                    / count(*) OVER () AS INTEGER
                ) AS bucket
            FROM selected_updates
            ORDER BY update_version
        ),
        bucket_raw AS (
            SELECT inferred.*, update_numbers.bucket
            FROM inferred
            JOIN update_numbers USING (update_version)
        ),
        aggregates AS (
            SELECT
                bucket,
                max(recorded_at_ms) AS recorded_at_ms,
                max(update_version) AS update_version,
                avg(batch_size) AS batch_size,
                avg(fill_ratio) AS fill_ratio,
                avg(recv_seconds) AS recv_seconds_avg,
                avg(h2d_seconds) AS h2d_seconds_avg,
                avg(decode_seconds) AS decode_seconds_avg,
                avg(inference_seconds) AS inference_seconds_avg
            FROM bucket_raw
            GROUP BY bucket
        ),
        metric_values AS (
            SELECT bucket, 'recv' AS metric, recv_seconds AS value
                FROM bucket_raw
            UNION ALL
            SELECT bucket, 'h2d', h2d_seconds FROM bucket_raw
            UNION ALL
            SELECT bucket, 'decode', decode_seconds FROM bucket_raw
            UNION ALL
            SELECT bucket, 'inference', inference_seconds
                FROM bucket_raw
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY bucket, metric ORDER BY value
                ) AS value_rank,
                count(*) OVER (
                    PARTITION BY bucket, metric
                ) AS value_count
            FROM metric_values
            WHERE value IS NOT NULL
        ),
        percentiles AS (
            SELECT
                bucket,
                max(CASE WHEN metric = 'recv' AND value_rank =
                    CAST((value_count * 95 + 99) / 100 AS INTEGER)
                    THEN value END) AS recv_seconds_p95,
                max(CASE WHEN metric = 'h2d' AND value_rank =
                    CAST((value_count * 95 + 99) / 100 AS INTEGER)
                    THEN value END) AS h2d_seconds_p95,
                max(CASE WHEN metric = 'decode' AND value_rank =
                    CAST((value_count * 95 + 99) / 100 AS INTEGER)
                    THEN value END) AS decode_seconds_p95,
                max(CASE WHEN metric = 'inference' AND value_rank =
                    CAST((value_count * 95 + 99) / 100 AS INTEGER)
                    THEN value END) AS inference_seconds_p95
            FROM ranked
            GROUP BY bucket
        )
        SELECT
            aggregates.recorded_at_ms,
            aggregates.update_version,
            aggregates.batch_size,
            aggregates.fill_ratio,
            aggregates.recv_seconds_avg,
            percentiles.recv_seconds_p95,
            aggregates.h2d_seconds_avg,
            percentiles.h2d_seconds_p95,
            aggregates.decode_seconds_avg,
            percentiles.decode_seconds_p95,
            aggregates.inference_seconds_avg,
            percentiles.inference_seconds_p95
        FROM aggregates
        JOIN percentiles USING (bucket)
        ORDER BY aggregates.bucket
        """,
        (session_id, update_limit, series_points),
    ).fetchall()
    return _rows_to_points(
        rows,
        started_at_ms=started_at_ms,
        value_names=(
            "batch_size",
            "fill_ratio",
            "recv_seconds_avg",
            "recv_seconds_p95",
            "h2d_seconds_avg",
            "h2d_seconds_p95",
            "decode_seconds_avg",
            "decode_seconds_p95",
            "inference_seconds_avg",
            "inference_seconds_p95",
        ),
    )


def _process_points(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    started_at_ms: int,
    update_limit: int,
) -> tuple[MetricPoint, ...]:
    rows = connection.execute(
        """
        WITH selected_updates AS (
            SELECT policy_version
            FROM training_logs
            WHERE session_id = ?
                AND event_type = 'sampling.completed'
                AND policy_version IS NOT NULL
            GROUP BY policy_version
            ORDER BY policy_version DESC
            LIMIT ?
        )
        SELECT
            max(recorded_at_ms),
            max(policy_version),
            process_index,
            sum(json_extract(event_json, '$.fields.completed_rounds')),
            sum(json_extract(event_json, '$.fields.decision_count')),
            sum(json_extract(
                event_json, '$.fields.policy_wait_seconds'
            )),
            sum(json_extract(event_json, '$.fields.round_seconds'))
        FROM training_logs
        JOIN selected_updates USING (policy_version)
        WHERE session_id = ? AND event_type = 'sampling.completed'
        GROUP BY process_index
        ORDER BY process_index
        """,
        (session_id, update_limit, session_id),
    ).fetchall()
    return _rows_to_points(
        rows,
        started_at_ms=started_at_ms,
        value_names=(
            "worker_index",
            "completed_rounds",
            "decision_count",
            "policy_wait_seconds",
            "round_seconds",
        ),
    )


def _rows_to_points(
    rows: list[tuple[object, ...]],
    *,
    started_at_ms: int,
    value_names: tuple[str, ...],
) -> tuple[MetricPoint, ...]:
    points: list[MetricPoint] = []
    for row in rows:
        recorded_at_ms = row[0]
        update = row[1]
        assert isinstance(recorded_at_ms, int)
        assert update is None or isinstance(update, int)
        values: JsonObject = {}
        for index, name in enumerate(value_names, start=2):
            value = row[index]
            assert value is None or isinstance(value, str | int | float)
            values[name] = value
        points.append(
            MetricPoint(
                update=update,
                elapsed_seconds=max(
                    (recorded_at_ms - started_at_ms) / 1000.0, 0.0
                ),
                recorded_at_ms=recorded_at_ms,
                values=values,
            )
        )
    return tuple(points)


def _scalar_int(connection: sqlite3.Connection, query: str) -> int:
    row = connection.execute(query).fetchone()
    assert row is not None
    value = row[0]
    assert isinstance(value, int)
    return value


def _empty_metrics(*, through_sequence: int = 0) -> TrainingMetrics:
    return TrainingMetrics(
        through_sequence=through_sequence,
        session_id=None,
        sessions=(),
        complete=True,
        dropped_event_count=0,
        totals={},
        datasets=MetricDatasets(
            throughput=(),
            optimization=(),
            ppo_timing=(),
            rollout=(),
            rewards=(),
            inference=(),
            processes=(),
        ),
    )
