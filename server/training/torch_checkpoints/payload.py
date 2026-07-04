"""State payload codec for torch training checkpoints."""

from __future__ import annotations

import math
import os
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard, cast

import torch
from torch import Tensor

from server import result as _result
from server.training.model import (
    TractorPolicyModel as _TractorPolicyModel,
)
from server.training.ppo import PPOTrainer
from server.training.torch_checkpoints.schema import (
    CHECKPOINT_SCHEMA_VERSION,
    checkpoint_corruption,
)
from server.training.torch_rng import (
    TorchRngState as _TorchRngState,
)
from server.training.torch_rng import (
    capture_torch_rng_state as _capture_torch_rng_state,
)


@dataclass(frozen=True, slots=True)
class CheckpointPayload:
    """Decoded trainable state payload."""

    checkpoint_id: str
    model_state: dict[str, Tensor]
    optimizer_state: dict[str, object]
    rng_state: _TorchRngState


def write_checkpoint_payload(
    *,
    path: Path,
    checkpoint_id: str,
    model: _TractorPolicyModel,
    trainer: PPOTrainer,
) -> _result.Ok[None] | _result.Rejected:
    """Atomically write one immutable state payload."""
    tmp_state_path = path.with_suffix(f"{path.suffix}.tmp")
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": checkpoint_id,
        "model_state": model.state_dict(),
        "optimizer_state": trainer.optimizer_state(),
        "rng_state": _rng_state_to_payload(_capture_torch_rng_state()),
    }
    try:
        torch.save(payload, tmp_state_path)
        os.replace(tmp_state_path, path)
    except OSError, RuntimeError, pickle.PickleError:
        _discard_file(tmp_state_path)
        return checkpoint_corruption(path, "state payload write failed")
    return _result.Ok(value=None)


def _discard_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def read_checkpoint_payload(
    path: Path,
) -> _result.Ok[CheckpointPayload] | _result.Rejected:
    """Read and validate one immutable state payload."""
    loaded_result = _load_checkpoint_payload(path=path)
    if isinstance(loaded_result, _result.Rejected):
        return loaded_result
    loaded = loaded_result.value
    schema_version_result = _payload_int_field(
        loaded, "schema_version", path
    )
    if isinstance(schema_version_result, _result.Rejected):
        return schema_version_result
    if schema_version_result.value != CHECKPOINT_SCHEMA_VERSION:
        return checkpoint_corruption(
            path,
            "state payload schema version mismatch",
        )
    checkpoint_id_result = _payload_str_field(
        loaded, "checkpoint_id", path
    )
    if isinstance(checkpoint_id_result, _result.Rejected):
        return checkpoint_id_result
    model_state_result = _payload_tensor_state_dict_field(
        loaded, "model_state", path
    )
    if isinstance(model_state_result, _result.Rejected):
        return model_state_result
    optimizer_state_result = _payload_str_object_dict_field(
        loaded, "optimizer_state", path
    )
    if isinstance(optimizer_state_result, _result.Rejected):
        return optimizer_state_result
    rng_payload_result = _payload_object_field(
        loaded, "rng_state", path
    )
    if isinstance(rng_payload_result, _result.Rejected):
        return rng_payload_result
    rng_state_result = _rng_state_from_payload(
        path=path,
        data=rng_payload_result.value,
    )
    if isinstance(rng_state_result, _result.Rejected):
        return rng_state_result
    return _result.Ok(
        value=CheckpointPayload(
            checkpoint_id=checkpoint_id_result.value,
            model_state=model_state_result.value,
            optimizer_state=optimizer_state_result.value,
            rng_state=rng_state_result.value,
        )
    )


def _load_checkpoint_payload(
    *,
    path: Path,
) -> _result.Ok[dict[object, object]] | _result.Rejected:
    try:
        loaded: object = torch.load(
            path,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    except (
        EOFError,
        RuntimeError,
        ValueError,
        pickle.UnpicklingError,
    ):
        return checkpoint_corruption(
            path, "state payload cannot be loaded"
        )
    if not _is_object_dict(loaded):
        return checkpoint_corruption(
            path, "state payload root is not an object"
        )
    return _result.Ok(value=loaded)


def _rng_state_to_payload(state: _TorchRngState) -> dict[str, object]:
    return {
        "python_version": state.python_random_state[0],
        "python_internal_state": list(state.python_random_state[1]),
        "python_gaussian": state.python_random_state[2],
        "torch_cpu_state": state.torch_cpu_state.cpu(),
        "torch_cuda_states": [
            cuda_state.cpu() for cuda_state in state.torch_cuda_states
        ],
    }


def _rng_state_from_payload(
    *,
    path: Path,
    data: dict[object, object],
) -> _result.Ok[_TorchRngState] | _result.Rejected:
    python_version = _payload_int_field(
        data, "python_version", path, label="rng_state.python_version"
    )
    if isinstance(python_version, _result.Rejected):
        return python_version
    python_internal_state = _payload_int_list_field(
        data,
        "python_internal_state",
        path,
        label="rng_state.python_internal_state",
    )
    if isinstance(python_internal_state, _result.Rejected):
        return python_internal_state
    python_gaussian = _payload_float_or_none_field(
        data, "python_gaussian", path, label="rng_state.python_gaussian"
    )
    if isinstance(python_gaussian, _result.Rejected):
        return python_gaussian
    torch_cpu_state = _payload_tensor_field(
        data, "torch_cpu_state", path, label="rng_state.torch_cpu_state"
    )
    if isinstance(torch_cpu_state, _result.Rejected):
        return torch_cpu_state
    torch_cuda_states = _payload_tensor_list_field(
        data,
        "torch_cuda_states",
        path,
        label="rng_state.torch_cuda_states",
    )
    if isinstance(torch_cuda_states, _result.Rejected):
        return torch_cuda_states
    return _result.Ok(
        value=_TorchRngState(
            python_random_state=(
                python_version.value,
                tuple(python_internal_state.value),
                python_gaussian.value,
            ),
            torch_cpu_state=torch_cpu_state.value,
            torch_cuda_states=tuple(torch_cuda_states.value),
        )
    )


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


def _payload_object_field(
    data: Mapping[object, object],
    field: str,
    path: Path,
    *,
    label: str | None = None,
) -> _result.Ok[dict[object, object]] | _result.Rejected:
    field_label = field if label is None else label
    if field not in data:
        return checkpoint_corruption(
            path, f"state payload missing {field_label}"
        )
    value = data[field]
    if not _is_object_dict(value):
        return checkpoint_corruption(
            path, f"state payload {field_label} is not an object"
        )
    return _result.Ok(value=value)


def _payload_str_object_dict_field(
    data: Mapping[object, object],
    field: str,
    path: Path,
    *,
    label: str | None = None,
) -> _result.Ok[dict[str, object]] | _result.Rejected:
    field_label = field if label is None else label
    if field not in data:
        return checkpoint_corruption(
            path, f"state payload missing {field_label}"
        )
    value = data[field]
    if not _is_str_object_dict(value):
        return checkpoint_corruption(
            path,
            f"state payload {field_label} is not a string-key object",
        )
    return _result.Ok(value=value)


def _payload_tensor_state_dict_field(
    data: Mapping[object, object],
    field: str,
    path: Path,
    *,
    label: str | None = None,
) -> _result.Ok[dict[str, Tensor]] | _result.Rejected:
    field_label = field if label is None else label
    if field not in data:
        return checkpoint_corruption(
            path, f"state payload missing {field_label}"
        )
    value = data[field]
    if not _is_tensor_state_dict(value):
        return checkpoint_corruption(
            path,
            f"state payload {field_label} is not a tensor state dict",
        )
    return _result.Ok(value=value)


def _payload_tensor_field(
    data: Mapping[object, object],
    field: str,
    path: Path,
    *,
    label: str | None = None,
) -> _result.Ok[Tensor] | _result.Rejected:
    field_label = field if label is None else label
    if field not in data:
        return checkpoint_corruption(
            path, f"state payload missing {field_label}"
        )
    value = data[field]
    if not isinstance(value, Tensor):
        return checkpoint_corruption(
            path, f"state payload {field_label} is not a tensor"
        )
    return _result.Ok(value=value)


def _payload_int_list_field(
    data: Mapping[object, object],
    field: str,
    path: Path,
    *,
    label: str | None = None,
) -> _result.Ok[list[int]] | _result.Rejected:
    field_label = field if label is None else label
    if field not in data:
        return checkpoint_corruption(
            path, f"state payload missing {field_label}"
        )
    value = data[field]
    if not _is_int_list(value):
        return checkpoint_corruption(
            path, f"state payload {field_label} is not an int list"
        )
    return _result.Ok(value=value)


def _payload_tensor_list_field(
    data: Mapping[object, object],
    field: str,
    path: Path,
    *,
    label: str | None = None,
) -> _result.Ok[list[Tensor]] | _result.Rejected:
    field_label = field if label is None else label
    if field not in data:
        return checkpoint_corruption(
            path, f"state payload missing {field_label}"
        )
    value = data[field]
    if not _is_tensor_list(value):
        return checkpoint_corruption(
            path, f"state payload {field_label} is not a tensor list"
        )
    return _result.Ok(value=value)


def _payload_float_or_none_field(
    data: Mapping[object, object],
    field: str,
    path: Path,
    *,
    label: str | None = None,
) -> _result.Ok[float | None] | _result.Rejected:
    field_label = field if label is None else label
    if field not in data:
        return checkpoint_corruption(
            path, f"state payload missing {field_label}"
        )
    value = data[field]
    if value is None:
        return _result.Ok(value=None)
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        return checkpoint_corruption(
            path, f"state payload {field_label} is not a finite number"
        )
    return _result.Ok(value=float(value))


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _is_str_object_dict(value: object) -> TypeGuard[dict[str, object]]:
    if not _is_object_dict(value):
        return False
    return all(isinstance(key, str) for key in value)


def _is_int_list(value: object) -> TypeGuard[list[int]]:
    if not isinstance(value, list):
        return False
    items = cast(list[object], value)
    return all(
        isinstance(item, int) and not isinstance(item, bool)
        for item in items
    )


def _is_tensor_list(value: object) -> TypeGuard[list[Tensor]]:
    if not isinstance(value, list):
        return False
    items = cast(list[object], value)
    return all(isinstance(item, Tensor) for item in items)


def _is_tensor_state_dict(
    value: object,
) -> TypeGuard[dict[str, Tensor]]:
    if not _is_object_dict(value):
        return False
    for key, item in value.items():
        if not isinstance(key, str):
            return False
        if not isinstance(item, Tensor):
            return False
    return True
