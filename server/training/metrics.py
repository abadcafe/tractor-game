"""Training metric events and JSONL persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

from server.training.json_types import JsonObject

METRICS_FILENAME = "metrics.jsonl"


@dataclass(frozen=True, slots=True)
class TrainingMetric:
    """One append-only progress sample for dashboards."""

    run_id: str
    total_games: int
    total_updates: int
    games_per_second: float
    decisions_per_second: float
    average_reward: float
    average_level_delta: float
    policy_loss: float | None
    value_loss: float | None
    entropy: float | None
    invalid_action_count: int
    resample_count: int
    forced_action_count: int
    legal_action_rate: float
    average_action_choices: float
    checkpoint_path: str | None


def metrics_path(run_dir: Path) -> Path:
    """Return the standard metrics file for a run directory."""
    return run_dir / METRICS_FILENAME


def append_metric(run_dir: Path, metric: TrainingMetric) -> None:
    """Append one metric JSON object to metrics.jsonl."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_path(run_dir)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_to_json(metric), ensure_ascii=False))
        file.write("\n")


def read_metrics(run_dir: Path) -> tuple[TrainingMetric, ...]:
    """Read all metric samples for a run directory."""
    path = metrics_path(run_dir)
    if not path.exists():
        return ()
    metrics: list[TrainingMetric] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        loaded: object = json.loads(line)
        assert _is_object_dict(loaded)
        metrics.append(_from_json(loaded))
    return tuple(metrics)


def _to_json(metric: TrainingMetric) -> JsonObject:
    return {
        "run_id": metric.run_id,
        "total_games": metric.total_games,
        "total_updates": metric.total_updates,
        "games_per_second": metric.games_per_second,
        "decisions_per_second": metric.decisions_per_second,
        "average_reward": metric.average_reward,
        "average_level_delta": metric.average_level_delta,
        "policy_loss": metric.policy_loss,
        "value_loss": metric.value_loss,
        "entropy": metric.entropy,
        "invalid_action_count": metric.invalid_action_count,
        "resample_count": metric.resample_count,
        "forced_action_count": metric.forced_action_count,
        "legal_action_rate": metric.legal_action_rate,
        "average_action_choices": metric.average_action_choices,
        "checkpoint_path": metric.checkpoint_path,
    }


def _from_json(data: dict[object, object]) -> TrainingMetric:
    return TrainingMetric(
        run_id=_str_field(data, "run_id"),
        total_games=_int_field(data, "total_games"),
        total_updates=_int_field(data, "total_updates"),
        games_per_second=_float_field(data, "games_per_second"),
        decisions_per_second=_float_field(data, "decisions_per_second"),
        average_reward=_float_field(data, "average_reward"),
        average_level_delta=_float_field(data, "average_level_delta"),
        policy_loss=_optional_float_field(data, "policy_loss"),
        value_loss=_optional_float_field(data, "value_loss"),
        entropy=_optional_float_field(data, "entropy"),
        invalid_action_count=_int_field(data, "invalid_action_count"),
        resample_count=_int_field(data, "resample_count"),
        forced_action_count=_int_field(data, "forced_action_count"),
        legal_action_rate=_float_field(data, "legal_action_rate"),
        average_action_choices=_float_field(
            data, "average_action_choices"
        ),
        checkpoint_path=_optional_str_field(data, "checkpoint_path"),
    )


def _str_field(data: dict[object, object], field: str) -> str:
    value = data[field]
    assert isinstance(value, str)
    return value


def _optional_str_field(
    data: dict[object, object], field: str
) -> str | None:
    value = data[field]
    if value is None:
        return None
    assert isinstance(value, str)
    return value


def _int_field(data: dict[object, object], field: str) -> int:
    value = data[field]
    assert isinstance(value, int)
    return value


def _float_field(data: dict[object, object], field: str) -> float:
    value = data[field]
    assert isinstance(value, int | float)
    return float(value)


def _optional_float_field(
    data: dict[object, object], field: str
) -> float | None:
    value = data[field]
    if value is None:
        return None
    assert isinstance(value, int | float)
    return float(value)


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)
