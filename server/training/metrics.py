"""Strict training metric records persisted in SQLite."""

from __future__ import annotations

import math
import sqlite3
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from server.foundation import result as _result
from server.training.persistence.schema import open_reader, open_writer

_METRIC_COLUMNS: tuple[str, ...] = (
    "total_games",
    "total_samples",
    "total_updates",
    "process_games_per_second",
    "process_samples_per_second",
    "last_round_decisions_per_second",
    "last_team0_reward",
    "last_team1_reward",
    "last_generated_action_count",
    "last_accepted_action_count",
    "last_decision_count",
    "last_average_action_choices",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "ppo_update_seconds",
    "ppo_minibatch_loss_seconds",
    "ppo_observation_batch_seconds",
    "ppo_observation_encode_seconds",
    "ppo_value_head_seconds",
    "ppo_argument_select_seconds",
    "ppo_argument_decode_seconds",
    "ppo_argument_distribution_seconds",
    "ppo_backward_seconds",
    "ppo_optimizer_step_seconds",
    "ppo_argument_decode_fraction",
    "ppo_argument_trace_batch_count",
    "ppo_argument_trace_row_count",
    "ppo_argument_trace_token_count",
    "ppo_argument_trace_valid_token_count",
    "ppo_argument_trace_padding_token_count",
    "checkpoint_path",
)
_RECORD_COLUMNS: tuple[str, ...] = (
    "sequence",
    "recorded_at_ms",
    *_METRIC_COLUMNS,
)
_COLUMN_SQL = ", ".join(_METRIC_COLUMNS)
_PLACEHOLDER_SQL = ", ".join("?" for _ in _METRIC_COLUMNS)


class TrainingMetric(BaseModel):
    """One validated PPO progress sample before persistence."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    total_games: int = Field(ge=0)
    total_samples: int = Field(ge=0)
    total_updates: int = Field(ge=0)
    process_games_per_second: float
    process_samples_per_second: float
    last_round_decisions_per_second: float
    last_team0_reward: float
    last_team1_reward: float
    last_generated_action_count: int = Field(ge=0)
    last_accepted_action_count: int = Field(ge=0)
    last_decision_count: int = Field(ge=0)
    last_average_action_choices: float
    policy_loss: float | None
    value_loss: float | None
    entropy: float | None
    approx_kl: float | None
    clip_fraction: float | None
    ppo_update_seconds: float | None = Field(ge=0.0)
    ppo_minibatch_loss_seconds: float | None = Field(ge=0.0)
    ppo_observation_batch_seconds: float | None = Field(ge=0.0)
    ppo_observation_encode_seconds: float | None = Field(ge=0.0)
    ppo_value_head_seconds: float | None = Field(ge=0.0)
    ppo_argument_select_seconds: float | None = Field(ge=0.0)
    ppo_argument_decode_seconds: float | None = Field(ge=0.0)
    ppo_argument_distribution_seconds: float | None = Field(ge=0.0)
    ppo_backward_seconds: float | None = Field(ge=0.0)
    ppo_optimizer_step_seconds: float | None = Field(ge=0.0)
    ppo_argument_decode_fraction: float | None = Field(ge=0.0, le=1.0)
    ppo_argument_trace_batch_count: int | None = Field(ge=0)
    ppo_argument_trace_row_count: int | None = Field(ge=0)
    ppo_argument_trace_token_count: int | None = Field(ge=0)
    ppo_argument_trace_valid_token_count: int | None = Field(ge=0)
    ppo_argument_trace_padding_token_count: int | None = Field(ge=0)
    checkpoint_path: str | None


class StoredTrainingMetric(TrainingMetric):
    """One persisted metric with a stable incremental cursor."""

    sequence: int = Field(gt=0)
    recorded_at_ms: int = Field(ge=0)


class TrainingMetricBoundaries(BaseModel):
    """First/latest metric records and the complete row count."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    count: int = Field(gt=0)
    first: StoredTrainingMetric
    latest: StoredTrainingMetric


def append_metric(
    run_dir: Path, metric: TrainingMetric
) -> _result.Ok[StoredTrainingMetric] | _result.Rejected:
    """Append one metric in a short SQLite transaction."""
    validation = validate_training_metric(metric)
    if isinstance(validation, _result.Rejected):
        return validation
    recorded_at_ms = time.time_ns() // 1_000_000
    opened = open_writer(run_dir)
    if isinstance(opened, _result.Rejected):
        return opened
    connection = opened.value
    try:
        cursor = connection.execute(
            "INSERT INTO training_metrics "
            f"(recorded_at_ms, {_COLUMN_SQL}) "
            f"VALUES (?, {_PLACEHOLDER_SQL})",
            (recorded_at_ms, *_metric_values(metric)),
        )
        connection.commit()
        sequence = cursor.lastrowid
    except sqlite3.Error:
        return _result.Rejected(
            reason="training metric could not be written"
        )
    finally:
        connection.close()
    assert sequence is not None
    return _result.Ok(
        value=StoredTrainingMetric(
            sequence=sequence,
            recorded_at_ms=recorded_at_ms,
            **metric.model_dump(),
        )
    )


def read_metric_records(
    run_dir: Path,
    *,
    after_sequence: int | None = None,
    limit: int = 500,
) -> _result.Ok[tuple[StoredTrainingMetric, ...]] | _result.Rejected:
    """Read the latest or incrementally newer metric records."""
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
    columns = ", ".join(_RECORD_COLUMNS)
    try:
        if after_sequence is None:
            rows = connection.execute(
                f"SELECT {columns} FROM ("
                f"SELECT {columns} FROM training_metrics "
                "ORDER BY sequence DESC LIMIT ?) ORDER BY sequence",
                (limit,),
            ).fetchall()
        else:
            rows = connection.execute(
                f"SELECT {columns} FROM training_metrics "
                "WHERE sequence > ? ORDER BY sequence LIMIT ?",
                (after_sequence, limit),
            ).fetchall()
    except sqlite3.Error:
        return _result.Rejected(
            reason="training metrics could not be read"
        )
    finally:
        connection.close()
    return _parse_metric_rows(rows)


def read_metric_boundaries(
    run_dir: Path,
) -> _result.Ok[TrainingMetricBoundaries | None] | _result.Rejected:
    """Read complete count plus the first and latest metric rows."""
    opened = open_reader(run_dir)
    if isinstance(opened, _result.Rejected):
        return opened
    connection = opened.value
    if connection is None:
        return _result.Ok(value=None)
    columns = ", ".join(_RECORD_COLUMNS)
    try:
        count_row = connection.execute(
            "SELECT count(*) FROM training_metrics"
        ).fetchone()
        rows = connection.execute(
            f"SELECT {columns} FROM training_metrics "
            "WHERE sequence = (SELECT min(sequence) FROM "
            "training_metrics) OR sequence = (SELECT max(sequence) "
            "FROM training_metrics) ORDER BY sequence"
        ).fetchall()
    except sqlite3.Error:
        return _result.Rejected(
            reason="training metric boundaries could not be read"
        )
    finally:
        connection.close()
    assert count_row is not None
    count = count_row[0]
    assert isinstance(count, int)
    if count == 0:
        return _result.Ok(value=None)
    parsed = _parse_metric_rows(rows)
    if isinstance(parsed, _result.Rejected):
        return parsed
    records = parsed.value
    assert len(records) in (1, 2)
    return _result.Ok(
        value=TrainingMetricBoundaries(
            count=count,
            first=records[0],
            latest=records[-1],
        )
    )


def read_metrics(
    run_dir: Path,
) -> _result.Ok[tuple[TrainingMetric, ...]] | _result.Rejected:
    """Read every metric for training-internal validation and tests."""
    records_result = read_metric_records(run_dir, limit=5000)
    if isinstance(records_result, _result.Rejected):
        return records_result
    return _result.Ok(
        value=tuple(
            TrainingMetric.model_validate(
                record.model_dump(
                    exclude={"sequence", "recorded_at_ms"}
                )
            )
            for record in records_result.value
        )
    )


def reconcile_metrics_with_checkpoint(
    run_dir: Path,
    *,
    total_rounds: int,
    total_samples: int,
    total_updates: int,
) -> _result.Ok[None] | _result.Rejected:
    """Discard metrics outside the resumed checkpoint prefix."""
    assert total_rounds >= 0
    assert total_samples >= 0
    assert total_updates >= 0
    opened = open_writer(run_dir)
    if isinstance(opened, _result.Rejected):
        return opened
    connection = opened.value
    try:
        connection.execute(
            "DELETE FROM training_metrics WHERE total_updates > ? OR "
            "(total_updates = ? AND "
            "(total_games <> ? OR total_samples <> ?))",
            (total_updates, total_updates, total_rounds, total_samples),
        )
        connection.commit()
    except sqlite3.Error:
        return _result.Rejected(
            reason="training metrics could not be reconciled"
        )
    finally:
        connection.close()
    return _result.Ok(value=None)


def validate_training_metric(
    metric: TrainingMetric,
) -> _result.Ok[None] | _result.Rejected:
    """Reject non-finite numeric values before SQLite persistence."""
    for field, value in metric:
        if isinstance(value, float) and not math.isfinite(value):
            return _result.Rejected(
                reason=f"metric {field} must be finite"
            )
    return _result.Ok(value=None)


def _parse_metric_rows(
    rows: list[tuple[object, ...]],
) -> _result.Ok[tuple[StoredTrainingMetric, ...]] | _result.Rejected:
    records: list[StoredTrainingMetric] = []
    try:
        for row in rows:
            payload: dict[str, object] = dict(
                zip(_RECORD_COLUMNS, row, strict=True)
            )
            records.append(StoredTrainingMetric.model_validate(payload))
    except ValidationError:
        return _result.Rejected(reason="training metrics are invalid")
    return _result.Ok(value=tuple(records))


def _metric_values(metric: TrainingMetric) -> tuple[object, ...]:
    return (
        metric.total_games,
        metric.total_samples,
        metric.total_updates,
        metric.process_games_per_second,
        metric.process_samples_per_second,
        metric.last_round_decisions_per_second,
        metric.last_team0_reward,
        metric.last_team1_reward,
        metric.last_generated_action_count,
        metric.last_accepted_action_count,
        metric.last_decision_count,
        metric.last_average_action_choices,
        metric.policy_loss,
        metric.value_loss,
        metric.entropy,
        metric.approx_kl,
        metric.clip_fraction,
        metric.ppo_update_seconds,
        metric.ppo_minibatch_loss_seconds,
        metric.ppo_observation_batch_seconds,
        metric.ppo_observation_encode_seconds,
        metric.ppo_value_head_seconds,
        metric.ppo_argument_select_seconds,
        metric.ppo_argument_decode_seconds,
        metric.ppo_argument_distribution_seconds,
        metric.ppo_backward_seconds,
        metric.ppo_optimizer_step_seconds,
        metric.ppo_argument_decode_fraction,
        metric.ppo_argument_trace_batch_count,
        metric.ppo_argument_trace_row_count,
        metric.ppo_argument_trace_token_count,
        metric.ppo_argument_trace_valid_token_count,
        metric.ppo_argument_trace_padding_token_count,
        metric.checkpoint_path,
    )
