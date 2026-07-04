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


@dataclass(frozen=True, slots=True)
class TrainingMetric:
    """One append-only progress sample for dashboards."""

    run_id: str
    total_games: int
    total_updates: int
    process_games_per_second: float
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
    )
    for field, value in optional_floats:
        if value is not None and not math.isfinite(value):
            return _result.Rejected(
                reason=f"metric {field} must be finite"
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
        "total_updates": metric.total_updates,
        "process_games_per_second": metric.process_games_per_second,
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
    run_id = _str_field(data, "run_id")
    total_games = _int_field(data, "total_games")
    total_updates = _int_field(data, "total_updates")
    process_games_per_second = _float_field(
        data, "process_games_per_second"
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
    policy_loss = _optional_float_field(data, "policy_loss")
    value_loss = _optional_float_field(data, "value_loss")
    entropy = _optional_float_field(data, "entropy")
    approx_kl = _optional_float_field(data, "approx_kl")
    clip_fraction = _optional_float_field(data, "clip_fraction")
    checkpoint_path = _optional_str_field(data, "checkpoint_path")
    if (
        run_id is None
        or total_games is None
        or total_updates is None
        or process_games_per_second is None
        or last_round_decisions_per_second is None
        or last_team0_reward is None
        or last_team1_reward is None
        or last_generated_action_count is None
        or last_accepted_action_count is None
        or last_decision_count is None
        or last_average_action_choices is None
    ):
        return None
    return TrainingMetric(
        run_id=run_id,
        total_games=total_games,
        total_updates=total_updates,
        process_games_per_second=process_games_per_second,
        last_round_decisions_per_second=last_round_decisions_per_second,
        last_team0_reward=last_team0_reward,
        last_team1_reward=last_team1_reward,
        last_generated_action_count=last_generated_action_count,
        last_accepted_action_count=last_accepted_action_count,
        last_decision_count=last_decision_count,
        last_average_action_choices=last_average_action_choices,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        checkpoint_path=checkpoint_path,
    )


def _str_field(data: dict[object, object], field: str) -> str | None:
    value = data.get(field)
    if not isinstance(value, str):
        return None
    return value


def _optional_str_field(
    data: dict[object, object], field: str
) -> str | None:
    value = data.get(field)
    if value is None:
        return None
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


def _optional_float_field(
    data: dict[object, object], field: str
) -> float | None:
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        return None
    return parsed


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)
