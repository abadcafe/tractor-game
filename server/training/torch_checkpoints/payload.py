"""State payload codec for torch training checkpoints."""

from __future__ import annotations

import os
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

import torch
from torch import Tensor

from server.foundation import result as _result
from server.training.model import (
    TractorPolicyModel as _TractorPolicyModel,
)
from server.training.ppo import PPOTrainer
from server.training.torch_checkpoints.schema import (
    CHECKPOINT_SCHEMA_VERSION,
    checkpoint_corruption,
)


@dataclass(frozen=True, slots=True)
class CheckpointPayload:
    """Decoded trainable state payload."""

    checkpoint_id: str
    model_state: dict[str, Tensor]
    optimizer_state: dict[str, object]


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
    return _result.Ok(
        value=CheckpointPayload(
            checkpoint_id=checkpoint_id_result.value,
            model_state=model_state_result.value,
            optimizer_state=optimizer_state_result.value,
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


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _is_str_object_dict(value: object) -> TypeGuard[dict[str, object]]:
    if not _is_object_dict(value):
        return False
    return all(isinstance(key, str) for key in value)


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
