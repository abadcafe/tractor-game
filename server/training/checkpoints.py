"""Portable JSON checkpoint metadata for training runs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

from server.training.json_types import JsonObject, JsonValue

CHECKPOINT_SCHEMA_VERSION: int = 1


@dataclass(frozen=True, slots=True)
class TrainingCheckpoint:
    """Checkpoint payload that can move between CPU and GPU machines."""

    run_id: str
    total_games: int
    total_updates: int
    model_config: JsonObject
    train_config: JsonObject
    token_schema_version: str
    rules_progress_version: str
    model_state: JsonObject
    optimizer_state: JsonObject
    rng_state: JsonObject
    best_eval_score: float | None


def save_checkpoint(path: Path, checkpoint: TrainingCheckpoint) -> None:
    """Atomically write a checkpoint JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(
        json.dumps(_to_json(checkpoint), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def load_checkpoint(path: Path) -> TrainingCheckpoint:
    """Load a checkpoint JSON file."""
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    assert _is_object_dict(loaded)
    data = loaded
    schema_version = data.get("schema_version")
    assert schema_version == CHECKPOINT_SCHEMA_VERSION
    return TrainingCheckpoint(
        run_id=_str_field(data, "run_id"),
        total_games=_int_field(data, "total_games"),
        total_updates=_int_field(data, "total_updates"),
        model_config=_object_field(data, "model_config"),
        train_config=_object_field(data, "train_config"),
        token_schema_version=_str_field(data, "token_schema_version"),
        rules_progress_version=_str_field(
            data, "rules_progress_version"
        ),
        model_state=_object_field(data, "model_state"),
        optimizer_state=_object_field(data, "optimizer_state"),
        rng_state=_object_field(data, "rng_state"),
        best_eval_score=_optional_float_field(data, "best_eval_score"),
    )


def _to_json(checkpoint: TrainingCheckpoint) -> JsonObject:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": checkpoint.run_id,
        "total_games": checkpoint.total_games,
        "total_updates": checkpoint.total_updates,
        "model_config": checkpoint.model_config,
        "train_config": checkpoint.train_config,
        "token_schema_version": checkpoint.token_schema_version,
        "rules_progress_version": checkpoint.rules_progress_version,
        "model_state": checkpoint.model_state,
        "optimizer_state": checkpoint.optimizer_state,
        "rng_state": checkpoint.rng_state,
        "best_eval_score": checkpoint.best_eval_score,
    }


def _str_field(data: dict[object, object], field: str) -> str:
    value = data[field]
    assert isinstance(value, str)
    return value


def _int_field(data: dict[object, object], field: str) -> int:
    value = data[field]
    assert isinstance(value, int)
    return value


def _optional_float_field(
    data: dict[object, object], field: str
) -> float | None:
    value = data[field]
    if value is None:
        return None
    assert isinstance(value, int | float)
    return float(value)


def _object_field(data: dict[object, object], field: str) -> JsonObject:
    value = data[field]
    assert _is_object_dict(value)
    result: JsonObject = {}
    for key, item in value.items():
        assert isinstance(key, str)
        result[key] = _json_value(item)
    return result


def _json_value(value: object) -> JsonValue:
    if value is None:
        return None
    if isinstance(value, str | int | float | bool):
        return value
    if _is_object_list(value):
        return [_json_value(item) for item in value]
    if _is_object_dict(value):
        result: JsonObject = {}
        for key, item in value.items():
            assert isinstance(key, str)
            result[key] = _json_value(item)
        return result
    raise AssertionError("checkpoint contains non-JSON value")


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)
