"""Validation for untrusted torch checkpoint data."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import TypeGuard, cast

from torch import Tensor

from server.foundation import result as _result
from server.foundation.json_value import JsonObject
from server.training.config import TrainConfig
from server.training.model import (
    MIN_ATTENTION_HEAD_DIMENSION,
    ModelConfig,
)
from server.training.torch_checkpoints.schema import (
    checkpoint_corruption,
)


def model_config_from_json(
    data: JsonObject,
    path: Path,
) -> _result.Ok[ModelConfig] | _result.Rejected:
    """Parse and validate model config from checkpoint JSON."""
    expected_fields = {"d_model", "layers", "heads"}
    if set(data) != expected_fields:
        return checkpoint_corruption(
            path,
            (
                "manifest model_config fields do not match the current "
                "schema"
            ),
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
            "model_config.heads",
        )
    if d_model.value // heads.value < MIN_ATTENTION_HEAD_DIMENSION:
        return checkpoint_corruption(
            path,
            "manifest model_config.d_model divided by "
            "model_config.heads must be at least "
            f"{MIN_ATTENTION_HEAD_DIMENSION}",
        )
    return _result.Ok(
        value=ModelConfig(
            d_model=d_model.value,
            layers=layers.value,
            heads=heads.value,
        )
    )


def train_config_from_json(
    data: JsonObject,
    path: Path,
) -> _result.Ok[TrainConfig] | _result.Rejected:
    """Parse and validate train config from checkpoint JSON."""
    seed = _json_int_field(
        data, "seed", path, label="train_config.seed"
    )
    if isinstance(seed, _result.Rejected):
        return seed
    learning_rate = _json_float_field(
        data,
        "learning_rate",
        path,
        label="train_config.learning_rate",
    )
    if isinstance(learning_rate, _result.Rejected):
        return learning_rate
    ppo_clip = _json_float_field(
        data, "ppo_clip", path, label="train_config.ppo_clip"
    )
    if isinstance(ppo_clip, _result.Rejected):
        return ppo_clip
    value_clip = _json_float_field(
        data, "value_clip", path, label="train_config.value_clip"
    )
    if isinstance(value_clip, _result.Rejected):
        return value_clip
    entropy_coef = _json_float_field(
        data, "entropy_coef", path, label="train_config.entropy_coef"
    )
    if isinstance(entropy_coef, _result.Rejected):
        return entropy_coef
    value_coef = _json_float_field(
        data, "value_coef", path, label="train_config.value_coef"
    )
    if isinstance(value_coef, _result.Rejected):
        return value_coef
    max_grad_norm = _json_float_field(
        data, "max_grad_norm", path, label="train_config.max_grad_norm"
    )
    if isinstance(max_grad_norm, _result.Rejected):
        return max_grad_norm
    ppo_epochs = _json_int_field(
        data, "ppo_epochs", path, label="train_config.ppo_epochs"
    )
    if isinstance(ppo_epochs, _result.Rejected):
        return ppo_epochs
    minibatch_size = _json_int_field(
        data,
        "minibatch_size",
        path,
        label="train_config.minibatch_size",
    )
    if isinstance(minibatch_size, _result.Rejected):
        return minibatch_size
    adam_beta1 = _json_float_field(
        data, "adam_beta1", path, label="train_config.adam_beta1"
    )
    if isinstance(adam_beta1, _result.Rejected):
        return adam_beta1
    adam_beta2 = _json_float_field(
        data, "adam_beta2", path, label="train_config.adam_beta2"
    )
    if isinstance(adam_beta2, _result.Rejected):
        return adam_beta2
    weight_decay = _json_float_field(
        data, "weight_decay", path, label="train_config.weight_decay"
    )
    if isinstance(weight_decay, _result.Rejected):
        return weight_decay
    validation = _validate_train_config_values(
        path=path,
        seed=seed.value,
        learning_rate=learning_rate.value,
        ppo_clip=ppo_clip.value,
        value_clip=value_clip.value,
        entropy_coef=entropy_coef.value,
        value_coef=value_coef.value,
        max_grad_norm=max_grad_norm.value,
        ppo_epochs=ppo_epochs.value,
        minibatch_size=minibatch_size.value,
        adam_beta1=adam_beta1.value,
        adam_beta2=adam_beta2.value,
        weight_decay=weight_decay.value,
    )
    if isinstance(validation, _result.Rejected):
        return validation
    return _result.Ok(
        value=TrainConfig(
            seed=seed.value,
            learning_rate=learning_rate.value,
            ppo_clip=ppo_clip.value,
            value_clip=value_clip.value,
            entropy_coef=entropy_coef.value,
            value_coef=value_coef.value,
            max_grad_norm=max_grad_norm.value,
            ppo_epochs=ppo_epochs.value,
            minibatch_size=minibatch_size.value,
            adam_beta1=adam_beta1.value,
            adam_beta2=adam_beta2.value,
            weight_decay=weight_decay.value,
        )
    )


def validate_optimizer_state_payload(
    *,
    state: dict[str, object],
    parameters: tuple[Tensor, ...],
    path: Path,
) -> _result.Ok[None] | _result.Rejected:
    """Validate optimizer payload before loading it into AdamW."""
    payload_state: dict[object, object] = {
        key: value for key, value in state.items()
    }
    kind = _payload_str_field(
        payload_state, "kind", path, label="optimizer_state.kind"
    )
    if isinstance(kind, _result.Rejected):
        return kind
    if kind.value != "ppo_adamw":
        return checkpoint_corruption(
            path, "state payload optimizer_state.kind is unsupported"
        )
    step_count = _payload_int_field(
        payload_state,
        "step_count",
        path,
        label="optimizer_state.step_count",
    )
    if isinstance(step_count, _result.Rejected):
        return step_count
    if step_count.value < 0:
        return checkpoint_corruption(
            path,
            "state payload optimizer_state.step_count must be >= 0",
        )
    exp_avgs = _payload_optional_tensor_list_field(
        payload_state,
        "exp_avgs",
        path,
        label="optimizer_state.exp_avgs",
    )
    if isinstance(exp_avgs, _result.Rejected):
        return exp_avgs
    exp_avg_sqs = _payload_optional_tensor_list_field(
        payload_state,
        "exp_avg_sqs",
        path,
        label="optimizer_state.exp_avg_sqs",
    )
    if isinstance(exp_avg_sqs, _result.Rejected):
        return exp_avg_sqs
    if len(exp_avgs.value) != len(parameters):
        return checkpoint_corruption(
            path,
            "state payload optimizer_state.exp_avgs length does not "
            "match model parameters",
        )
    if len(exp_avg_sqs.value) != len(parameters):
        return checkpoint_corruption(
            path,
            "state payload optimizer_state.exp_avg_sqs length does not "
            "match model parameters",
        )
    for index, parameter in enumerate(parameters):
        exp_avg = exp_avgs.value[index]
        exp_avg_sq = exp_avg_sqs.value[index]
        if exp_avg is not None:
            exp_avg_validation = _validate_optimizer_tensor(
                path=path,
                label="optimizer_state.exp_avgs",
                value=exp_avg,
                parameter=parameter,
            )
            if isinstance(exp_avg_validation, _result.Rejected):
                return exp_avg_validation
        if exp_avg_sq is not None:
            exp_avg_sq_validation = _validate_optimizer_tensor(
                path=path,
                label="optimizer_state.exp_avg_sqs",
                value=exp_avg_sq,
                parameter=parameter,
            )
            if isinstance(exp_avg_sq_validation, _result.Rejected):
                return exp_avg_sq_validation
    return _result.Ok(value=None)


def _validate_train_config_values(
    *,
    path: Path,
    seed: int,
    learning_rate: float,
    ppo_clip: float,
    value_clip: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    ppo_epochs: int,
    minibatch_size: int,
    adam_beta1: float,
    adam_beta2: float,
    weight_decay: float,
) -> _result.Ok[None] | _result.Rejected:
    if seed < 0:
        return checkpoint_corruption(
            path, "manifest train_config.seed must be >= 0"
        )
    if learning_rate <= 0.0:
        return checkpoint_corruption(
            path, "manifest train_config.learning_rate must be > 0"
        )
    if ppo_clip <= 0.0 or ppo_clip > 1.0:
        return checkpoint_corruption(
            path,
            "manifest train_config.ppo_clip must be > 0 and <= 1",
        )
    if value_clip <= 0.0:
        return checkpoint_corruption(
            path, "manifest train_config.value_clip must be > 0"
        )
    if entropy_coef < 0.0:
        return checkpoint_corruption(
            path, "manifest train_config.entropy_coef must be >= 0"
        )
    if value_coef < 0.0:
        return checkpoint_corruption(
            path, "manifest train_config.value_coef must be >= 0"
        )
    if max_grad_norm < 0.0:
        return checkpoint_corruption(
            path, "manifest train_config.max_grad_norm must be >= 0"
        )
    if ppo_epochs <= 0:
        return checkpoint_corruption(
            path, "manifest train_config.ppo_epochs must be > 0"
        )
    if minibatch_size <= 0:
        return checkpoint_corruption(
            path, "manifest train_config.minibatch_size must be > 0"
        )
    if adam_beta1 < 0.0 or adam_beta1 >= 1.0:
        return checkpoint_corruption(
            path,
            "manifest train_config.adam_beta1 must be >= 0 and < 1",
        )
    if adam_beta2 < 0.0 or adam_beta2 >= 1.0:
        return checkpoint_corruption(
            path,
            "manifest train_config.adam_beta2 must be >= 0 and < 1",
        )
    if weight_decay < 0.0:
        return checkpoint_corruption(
            path, "manifest train_config.weight_decay must be >= 0"
        )
    return _result.Ok(value=None)


def _validate_optimizer_tensor(
    *,
    path: Path,
    label: str,
    value: Tensor,
    parameter: Tensor,
) -> _result.Ok[None] | _result.Rejected:
    if value.shape != parameter.shape:
        return checkpoint_corruption(
            path,
            f"state payload {label} tensor shape does not match "
            "model parameter",
        )
    if value.dtype != parameter.dtype:
        return checkpoint_corruption(
            path,
            f"state payload {label} tensor dtype does not match "
            "model parameter",
        )
    return _result.Ok(value=None)


def _payload_int_field(
    data: Mapping[object, object],
    field: str,
    path: Path,
    *,
    label: str | None = None,
) -> _result.Ok[int] | _result.Rejected:
    field_label = field if label is None else label
    if field not in data:
        return checkpoint_corruption(
            path, f"state payload missing {field_label}"
        )
    value = data[field]
    if not isinstance(value, int) or isinstance(value, bool):
        return checkpoint_corruption(
            path, f"state payload {field_label} is not an int"
        )
    return _result.Ok(value=value)


def _payload_str_field(
    data: Mapping[object, object],
    field: str,
    path: Path,
    *,
    label: str | None = None,
) -> _result.Ok[str] | _result.Rejected:
    field_label = field if label is None else label
    if field not in data:
        return checkpoint_corruption(
            path, f"state payload missing {field_label}"
        )
    value = data[field]
    if not isinstance(value, str) or not value:
        return checkpoint_corruption(
            path, f"state payload {field_label} is not a string"
        )
    return _result.Ok(value=value)


def _payload_optional_tensor_list_field(
    data: Mapping[object, object],
    field: str,
    path: Path,
    *,
    label: str | None = None,
) -> _result.Ok[list[Tensor | None]] | _result.Rejected:
    field_label = field if label is None else label
    if field not in data:
        return checkpoint_corruption(
            path, f"state payload missing {field_label}"
        )
    value = data[field]
    if not _is_optional_tensor_list(value):
        return checkpoint_corruption(
            path,
            f"state payload {field_label} is not an optional tensor "
            "list",
        )
    return _result.Ok(value=value)


def _json_int_field(
    data: JsonObject,
    field: str,
    path: Path,
    *,
    label: str | None = None,
) -> _result.Ok[int] | _result.Rejected:
    field_label = field if label is None else label
    if field not in data:
        return checkpoint_corruption(
            path, f"manifest missing {field_label}"
        )
    value = data[field]
    if not isinstance(value, int) or isinstance(value, bool):
        return checkpoint_corruption(
            path, f"manifest {field_label} is not an int"
        )
    return _result.Ok(value=value)


def _json_float_field(
    data: JsonObject,
    field: str,
    path: Path,
    *,
    label: str | None = None,
) -> _result.Ok[float] | _result.Rejected:
    field_label = field if label is None else label
    if field not in data:
        return checkpoint_corruption(
            path, f"manifest missing {field_label}"
        )
    value = data[field]
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        return checkpoint_corruption(
            path, f"manifest {field_label} is not a finite number"
        )
    return _result.Ok(value=float(value))


def _is_optional_tensor_list(
    value: object,
) -> TypeGuard[list[Tensor | None]]:
    if not isinstance(value, list):
        return False
    items = cast(list[object], value)
    return all(
        item is None or isinstance(item, Tensor) for item in items
    )
