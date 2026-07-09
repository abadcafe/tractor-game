"""Training metric events and JSONL persistence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

from server import result as _result
from server.training.json_types import JsonObject

METRICS_FILENAME = "metrics.jsonl"
_REQUIRED_METRIC_SCHEMA_FIELDS: tuple[str, ...] = (
    "run_id",
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
)
_NULLABLE_FLOAT_METRIC_SCHEMA_FIELDS: tuple[str, ...] = (
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
)
_NULLABLE_INT_METRIC_SCHEMA_FIELDS: tuple[str, ...] = (
    "ppo_argument_trace_batch_count",
    "ppo_argument_trace_row_count",
    "ppo_argument_trace_token_count",
    "ppo_argument_trace_valid_token_count",
    "ppo_argument_trace_padding_token_count",
)
_NULLABLE_STR_METRIC_SCHEMA_FIELDS: tuple[str, ...] = (
    "checkpoint_path",
)


@dataclass(frozen=True, slots=True)
class TrainingMetric:
    """One append-only progress sample.

    Writers emit every field.  Readers accept missing nullable fields as
    ``None`` but reject records with malformed present fields.
    """

    run_id: str
    total_games: int
    total_samples: int
    total_updates: int
    process_games_per_second: float
    process_samples_per_second: float
    last_round_decisions_per_second: float
    last_team0_reward: float
    last_team1_reward: float
    last_generated_action_count: int
    last_accepted_action_count: int
    last_decision_count: int
    last_average_action_choices: float
    policy_loss: float | None
    value_loss: float | None
    entropy: float | None
    approx_kl: float | None
    clip_fraction: float | None
    ppo_update_seconds: float | None
    ppo_minibatch_loss_seconds: float | None
    ppo_observation_batch_seconds: float | None
    ppo_observation_encode_seconds: float | None
    ppo_value_head_seconds: float | None
    ppo_argument_select_seconds: float | None
    ppo_argument_decode_seconds: float | None
    ppo_argument_distribution_seconds: float | None
    ppo_backward_seconds: float | None
    ppo_optimizer_step_seconds: float | None
    ppo_argument_decode_fraction: float | None
    ppo_argument_trace_batch_count: int | None
    ppo_argument_trace_row_count: int | None
    ppo_argument_trace_token_count: int | None
    ppo_argument_trace_valid_token_count: int | None
    ppo_argument_trace_padding_token_count: int | None
    checkpoint_path: str | None


def metrics_path(run_dir: Path) -> Path:
    """Return the standard metrics file for a run directory."""
    return run_dir / METRICS_FILENAME


def append_metric(
    run_dir: Path, metric: TrainingMetric
) -> _result.Ok[None] | _result.Rejected:
    """Append one metric JSON object to metrics.jsonl."""
    validation = validate_training_metric(metric)
    if isinstance(validation, _result.Rejected):
        return validation
    path = metrics_path(run_dir)
    try:
        metric_json = json.dumps(
            _to_json(metric), ensure_ascii=False, allow_nan=False
        )
        line = f"{metric_json}\n"
    except ValueError:
        return _result.Rejected(
            reason=f"metric serialization failed: {path}"
        )
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(line)
    except OSError:
        return _result.Rejected(reason=f"metric write failed: {path}")
    return _result.Ok(value=None)


def validate_training_metric(
    metric: TrainingMetric,
) -> _result.Ok[None] | _result.Rejected:
    """Reject metric samples that cannot be valid JSON numbers."""
    required_floats = (
        ("process_games_per_second", metric.process_games_per_second),
        (
            "process_samples_per_second",
            metric.process_samples_per_second,
        ),
        (
            "last_round_decisions_per_second",
            metric.last_round_decisions_per_second,
        ),
        ("last_team0_reward", metric.last_team0_reward),
        ("last_team1_reward", metric.last_team1_reward),
        (
            "last_average_action_choices",
            metric.last_average_action_choices,
        ),
    )
    for field, value in required_floats:
        if not math.isfinite(value):
            return _result.Rejected(
                reason=f"metric {field} must be finite"
            )
    optional_floats = (
        ("policy_loss", metric.policy_loss),
        ("value_loss", metric.value_loss),
        ("entropy", metric.entropy),
        ("approx_kl", metric.approx_kl),
        ("clip_fraction", metric.clip_fraction),
        ("ppo_update_seconds", metric.ppo_update_seconds),
        (
            "ppo_minibatch_loss_seconds",
            metric.ppo_minibatch_loss_seconds,
        ),
        (
            "ppo_observation_batch_seconds",
            metric.ppo_observation_batch_seconds,
        ),
        (
            "ppo_observation_encode_seconds",
            metric.ppo_observation_encode_seconds,
        ),
        ("ppo_value_head_seconds", metric.ppo_value_head_seconds),
        (
            "ppo_argument_select_seconds",
            metric.ppo_argument_select_seconds,
        ),
        (
            "ppo_argument_decode_seconds",
            metric.ppo_argument_decode_seconds,
        ),
        (
            "ppo_argument_distribution_seconds",
            metric.ppo_argument_distribution_seconds,
        ),
        ("ppo_backward_seconds", metric.ppo_backward_seconds),
        (
            "ppo_optimizer_step_seconds",
            metric.ppo_optimizer_step_seconds,
        ),
        (
            "ppo_argument_decode_fraction",
            metric.ppo_argument_decode_fraction,
        ),
    )
    for field, value in optional_floats:
        if value is not None and not math.isfinite(value):
            return _result.Rejected(
                reason=f"metric {field} must be finite"
            )
    optional_nonnegative_floats = (
        ("ppo_update_seconds", metric.ppo_update_seconds),
        (
            "ppo_minibatch_loss_seconds",
            metric.ppo_minibatch_loss_seconds,
        ),
        (
            "ppo_observation_batch_seconds",
            metric.ppo_observation_batch_seconds,
        ),
        (
            "ppo_observation_encode_seconds",
            metric.ppo_observation_encode_seconds,
        ),
        ("ppo_value_head_seconds", metric.ppo_value_head_seconds),
        (
            "ppo_argument_select_seconds",
            metric.ppo_argument_select_seconds,
        ),
        (
            "ppo_argument_decode_seconds",
            metric.ppo_argument_decode_seconds,
        ),
        (
            "ppo_argument_distribution_seconds",
            metric.ppo_argument_distribution_seconds,
        ),
        ("ppo_backward_seconds", metric.ppo_backward_seconds),
        (
            "ppo_optimizer_step_seconds",
            metric.ppo_optimizer_step_seconds,
        ),
        (
            "ppo_argument_decode_fraction",
            metric.ppo_argument_decode_fraction,
        ),
    )
    for field, value in optional_nonnegative_floats:
        if value is not None and value < 0.0:
            return _result.Rejected(
                reason=f"metric {field} must be non-negative"
            )
    if (
        metric.ppo_argument_decode_fraction is not None
        and metric.ppo_argument_decode_fraction > 1.0
    ):
        return _result.Rejected(
            reason="metric ppo_argument_decode_fraction must be <= 1"
        )
    optional_ints = (
        (
            "ppo_argument_trace_batch_count",
            metric.ppo_argument_trace_batch_count,
        ),
        (
            "ppo_argument_trace_row_count",
            metric.ppo_argument_trace_row_count,
        ),
        (
            "ppo_argument_trace_token_count",
            metric.ppo_argument_trace_token_count,
        ),
        (
            "ppo_argument_trace_valid_token_count",
            metric.ppo_argument_trace_valid_token_count,
        ),
        (
            "ppo_argument_trace_padding_token_count",
            metric.ppo_argument_trace_padding_token_count,
        ),
    )
    for field, value in optional_ints:
        if value is not None and value < 0:
            return _result.Rejected(
                reason=f"metric {field} must be non-negative"
            )
    required_ints = (
        ("total_games", metric.total_games),
        ("total_samples", metric.total_samples),
        ("total_updates", metric.total_updates),
        (
            "last_generated_action_count",
            metric.last_generated_action_count,
        ),
        (
            "last_accepted_action_count",
            metric.last_accepted_action_count,
        ),
        ("last_decision_count", metric.last_decision_count),
    )
    for field, value in required_ints:
        if value < 0:
            return _result.Rejected(
                reason=f"metric {field} must be non-negative"
            )
    return _result.Ok(value=None)


def read_metrics(run_dir: Path) -> tuple[TrainingMetric, ...]:
    """Read all valid metric samples for a run directory."""
    path = metrics_path(run_dir)
    if not path.exists():
        return ()
    metrics: list[TrainingMetric] = []
    for record in path.read_bytes().splitlines():
        if not record.strip():
            continue
        metric = _metric_from_json_record(record)
        if metric is not None:
            metrics.append(metric)
    return tuple(metrics)


def _to_json(metric: TrainingMetric) -> JsonObject:
    return {
        "run_id": metric.run_id,
        "total_games": metric.total_games,
        "total_samples": metric.total_samples,
        "total_updates": metric.total_updates,
        "process_games_per_second": metric.process_games_per_second,
        "process_samples_per_second": metric.process_samples_per_second,
        "last_round_decisions_per_second": (
            metric.last_round_decisions_per_second
        ),
        "last_team0_reward": metric.last_team0_reward,
        "last_team1_reward": metric.last_team1_reward,
        "last_generated_action_count": (
            metric.last_generated_action_count
        ),
        "last_accepted_action_count": (
            metric.last_accepted_action_count
        ),
        "last_decision_count": metric.last_decision_count,
        "last_average_action_choices": (
            metric.last_average_action_choices
        ),
        "policy_loss": metric.policy_loss,
        "value_loss": metric.value_loss,
        "entropy": metric.entropy,
        "approx_kl": metric.approx_kl,
        "clip_fraction": metric.clip_fraction,
        "ppo_update_seconds": metric.ppo_update_seconds,
        "ppo_minibatch_loss_seconds": (
            metric.ppo_minibatch_loss_seconds
        ),
        "ppo_observation_batch_seconds": (
            metric.ppo_observation_batch_seconds
        ),
        "ppo_observation_encode_seconds": (
            metric.ppo_observation_encode_seconds
        ),
        "ppo_value_head_seconds": metric.ppo_value_head_seconds,
        "ppo_argument_select_seconds": (
            metric.ppo_argument_select_seconds
        ),
        "ppo_argument_decode_seconds": (
            metric.ppo_argument_decode_seconds
        ),
        "ppo_argument_distribution_seconds": (
            metric.ppo_argument_distribution_seconds
        ),
        "ppo_backward_seconds": metric.ppo_backward_seconds,
        "ppo_optimizer_step_seconds": (
            metric.ppo_optimizer_step_seconds
        ),
        "ppo_argument_decode_fraction": (
            metric.ppo_argument_decode_fraction
        ),
        "ppo_argument_trace_batch_count": (
            metric.ppo_argument_trace_batch_count
        ),
        "ppo_argument_trace_row_count": (
            metric.ppo_argument_trace_row_count
        ),
        "ppo_argument_trace_token_count": (
            metric.ppo_argument_trace_token_count
        ),
        "ppo_argument_trace_valid_token_count": (
            metric.ppo_argument_trace_valid_token_count
        ),
        "ppo_argument_trace_padding_token_count": (
            metric.ppo_argument_trace_padding_token_count
        ),
        "checkpoint_path": metric.checkpoint_path,
    }


def _metric_from_json_record(record: bytes) -> TrainingMetric | None:
    try:
        line = record.decode("utf-8")
    except UnicodeDecodeError:
        return None
    try:
        loaded: object = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not _is_object_dict(loaded):
        return None
    return _from_json(loaded)


def _from_json(data: dict[object, object]) -> TrainingMetric | None:
    if not _has_metric_schema_fields(data):
        return None
    run_id = _str_field(data, "run_id")
    total_games = _int_field(data, "total_games")
    total_samples = _int_field(data, "total_samples")
    total_updates = _int_field(data, "total_updates")
    process_games_per_second = _float_field(
        data, "process_games_per_second"
    )
    process_samples_per_second = _float_field(
        data, "process_samples_per_second"
    )
    last_round_decisions_per_second = _float_field(
        data, "last_round_decisions_per_second"
    )
    last_team0_reward = _float_field(data, "last_team0_reward")
    last_team1_reward = _float_field(data, "last_team1_reward")
    last_generated_action_count = _int_field(
        data, "last_generated_action_count"
    )
    last_accepted_action_count = _int_field(
        data, "last_accepted_action_count"
    )
    last_decision_count = _int_field(data, "last_decision_count")
    last_average_action_choices = _float_field(
        data, "last_average_action_choices"
    )
    nullable_floats_result = _nullable_float_fields(
        data, _NULLABLE_FLOAT_METRIC_SCHEMA_FIELDS
    )
    if isinstance(nullable_floats_result, _result.Rejected):
        return None
    nullable_ints_result = _nullable_int_fields(
        data, _NULLABLE_INT_METRIC_SCHEMA_FIELDS
    )
    if isinstance(nullable_ints_result, _result.Rejected):
        return None
    nullable_strs_result = _nullable_str_fields(
        data, _NULLABLE_STR_METRIC_SCHEMA_FIELDS
    )
    if isinstance(nullable_strs_result, _result.Rejected):
        return None
    nullable_floats = nullable_floats_result.value
    nullable_ints = nullable_ints_result.value
    nullable_strs = nullable_strs_result.value
    if (
        run_id is None
        or total_games is None
        or total_samples is None
        or total_updates is None
        or process_games_per_second is None
        or process_samples_per_second is None
        or last_round_decisions_per_second is None
        or last_team0_reward is None
        or last_team1_reward is None
        or last_generated_action_count is None
        or last_accepted_action_count is None
        or last_decision_count is None
        or last_average_action_choices is None
    ):
        return None
    metric = TrainingMetric(
        run_id=run_id,
        total_games=total_games,
        total_samples=total_samples,
        total_updates=total_updates,
        process_games_per_second=process_games_per_second,
        process_samples_per_second=process_samples_per_second,
        last_round_decisions_per_second=last_round_decisions_per_second,
        last_team0_reward=last_team0_reward,
        last_team1_reward=last_team1_reward,
        last_generated_action_count=last_generated_action_count,
        last_accepted_action_count=last_accepted_action_count,
        last_decision_count=last_decision_count,
        last_average_action_choices=last_average_action_choices,
        policy_loss=nullable_floats["policy_loss"],
        value_loss=nullable_floats["value_loss"],
        entropy=nullable_floats["entropy"],
        approx_kl=nullable_floats["approx_kl"],
        clip_fraction=nullable_floats["clip_fraction"],
        ppo_update_seconds=nullable_floats["ppo_update_seconds"],
        ppo_minibatch_loss_seconds=nullable_floats[
            "ppo_minibatch_loss_seconds"
        ],
        ppo_observation_batch_seconds=nullable_floats[
            "ppo_observation_batch_seconds"
        ],
        ppo_observation_encode_seconds=nullable_floats[
            "ppo_observation_encode_seconds"
        ],
        ppo_value_head_seconds=nullable_floats[
            "ppo_value_head_seconds"
        ],
        ppo_argument_select_seconds=nullable_floats[
            "ppo_argument_select_seconds"
        ],
        ppo_argument_decode_seconds=nullable_floats[
            "ppo_argument_decode_seconds"
        ],
        ppo_argument_distribution_seconds=(
            nullable_floats["ppo_argument_distribution_seconds"]
        ),
        ppo_backward_seconds=nullable_floats["ppo_backward_seconds"],
        ppo_optimizer_step_seconds=nullable_floats[
            "ppo_optimizer_step_seconds"
        ],
        ppo_argument_decode_fraction=nullable_floats[
            "ppo_argument_decode_fraction"
        ],
        ppo_argument_trace_batch_count=(
            nullable_ints["ppo_argument_trace_batch_count"]
        ),
        ppo_argument_trace_row_count=nullable_ints[
            "ppo_argument_trace_row_count"
        ],
        ppo_argument_trace_token_count=(
            nullable_ints["ppo_argument_trace_token_count"]
        ),
        ppo_argument_trace_valid_token_count=(
            nullable_ints["ppo_argument_trace_valid_token_count"]
        ),
        ppo_argument_trace_padding_token_count=(
            nullable_ints["ppo_argument_trace_padding_token_count"]
        ),
        checkpoint_path=nullable_strs["checkpoint_path"],
    )
    validation = validate_training_metric(metric)
    if isinstance(validation, _result.Rejected):
        return None
    return metric


def _has_metric_schema_fields(data: dict[object, object]) -> bool:
    return all(
        field in data for field in _REQUIRED_METRIC_SCHEMA_FIELDS
    )


def _str_field(data: dict[object, object], field: str) -> str | None:
    value = data.get(field)
    if not isinstance(value, str):
        return None
    return value


def _int_field(data: dict[object, object], field: str) -> int | None:
    value = data.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value


def _float_field(
    data: dict[object, object], field: str
) -> float | None:
    value = data.get(field)
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        return None
    return parsed


def _nullable_float_fields(
    data: dict[object, object], fields: tuple[str, ...]
) -> _result.Ok[dict[str, float | None]] | _result.Rejected:
    parsed_fields: dict[str, float | None] = {}
    for field in fields:
        result = _nullable_float_field(data, field)
        if isinstance(result, _result.Rejected):
            return result
        parsed_fields[field] = result.value
    return _result.Ok(value=parsed_fields)


def _nullable_float_field(
    data: dict[object, object], field: str
) -> _result.Ok[float | None] | _result.Rejected:
    if field not in data:
        return _result.Ok(value=None)
    value = data[field]
    if value is None:
        return _result.Ok(value=None)
    if not isinstance(value, int | float) or isinstance(value, bool):
        return _result.Rejected(reason=f"metric {field} is invalid")
    parsed = float(value)
    if not math.isfinite(parsed):
        return _result.Rejected(reason=f"metric {field} must be finite")
    return _result.Ok(value=parsed)


def _nullable_int_fields(
    data: dict[object, object], fields: tuple[str, ...]
) -> _result.Ok[dict[str, int | None]] | _result.Rejected:
    parsed_fields: dict[str, int | None] = {}
    for field in fields:
        result = _nullable_int_field(data, field)
        if isinstance(result, _result.Rejected):
            return result
        parsed_fields[field] = result.value
    return _result.Ok(value=parsed_fields)


def _nullable_int_field(
    data: dict[object, object], field: str
) -> _result.Ok[int | None] | _result.Rejected:
    if field not in data:
        return _result.Ok(value=None)
    value = data[field]
    if value is None:
        return _result.Ok(value=None)
    if not isinstance(value, int) or isinstance(value, bool):
        return _result.Rejected(reason=f"metric {field} is invalid")
    return _result.Ok(value=value)


def _nullable_str_fields(
    data: dict[object, object], fields: tuple[str, ...]
) -> _result.Ok[dict[str, str | None]] | _result.Rejected:
    parsed_fields: dict[str, str | None] = {}
    for field in fields:
        result = _nullable_str_field(data, field)
        if isinstance(result, _result.Rejected):
            return result
        parsed_fields[field] = result.value
    return _result.Ok(value=parsed_fields)


def _nullable_str_field(
    data: dict[object, object], field: str
) -> _result.Ok[str | None] | _result.Rejected:
    if field not in data:
        return _result.Ok(value=None)
    value = data[field]
    if value is None:
        return _result.Ok(value=None)
    if not isinstance(value, str):
        return _result.Rejected(reason=f"metric {field} is invalid")
    return _result.Ok(value=value)


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)
