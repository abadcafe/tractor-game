"""Validation of policy-model checkpoint metadata."""

from __future__ import annotations

from pathlib import Path

from server.foundation import result as _result
from server.foundation.json_value import JsonObject
from server.policy_model.checkpoint.schema import checkpoint_corruption
from server.policy_model.network import (
    MIN_ATTENTION_HEAD_DIMENSION,
    ModelConfig,
)


def model_config_from_json(
    data: JsonObject,
    path: Path,
) -> _result.Ok[ModelConfig] | _result.Rejected:
    """Parse and validate model config from checkpoint JSON."""
    if set(data) != {
        "d_model",
        "layers",
        "heads",
    }:
        return checkpoint_corruption(
            path,
            "manifest model_config fields do not match "
            + "the current schema",
        )
    d_model = _json_int_field(
        data, "d_model", path, label="model_config.d_model"
    )
    if isinstance(d_model, _result.Rejected):
        return d_model
    layers = _json_int_field(
        data, "layers", path, label="model_config.layers"
    )
    if isinstance(layers, _result.Rejected):
        return layers
    heads = _json_int_field(
        data, "heads", path, label="model_config.heads"
    )
    if isinstance(heads, _result.Rejected):
        return heads
    for label, value in (
        ("model_config.d_model", d_model.value),
        ("model_config.layers", layers.value),
        ("model_config.heads", heads.value),
    ):
        if value <= 0:
            return checkpoint_corruption(
                path, f"manifest {label} must be > 0"
            )
    if d_model.value % heads.value != 0:
        return checkpoint_corruption(
            path,
            "manifest model_config.d_model must be divisible by "
            + "model_config.heads",
        )
    if d_model.value // heads.value < MIN_ATTENTION_HEAD_DIMENSION:
        return checkpoint_corruption(
            path,
            "manifest model_config.d_model divided by "
            + "model_config.heads must be at least "
            + f"{MIN_ATTENTION_HEAD_DIMENSION}",
        )
    return _result.Ok(
        value=ModelConfig(
            d_model=d_model.value,
            layers=layers.value,
            heads=heads.value,
        )
    )


def _json_int_field(
    data: JsonObject,
    field: str,
    path: Path,
    *,
    label: str,
) -> _result.Ok[int] | _result.Rejected:
    if field not in data:
        return checkpoint_corruption(path, f"manifest missing {label}")
    value = data[field]
    if not isinstance(value, int) or isinstance(value, bool):
        return checkpoint_corruption(
            path, f"manifest {label} is not an int"
        )
    return _result.Ok(value=value)
