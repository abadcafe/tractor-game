"""Strict schema-27 policy and trainer payload codecs."""

from __future__ import annotations

import os
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard, cast

import torch
from pydantic import ConfigDict, TypeAdapter, ValidationError
from torch import Tensor

from server.checkpoint_contract import CHECKPOINT_SCHEMA_VERSION
from server.foundation import result as _result
from server.policy_model.checkpoint.schema import checkpoint_corruption

_OBJECT_DICT = TypeAdapter(
    dict[object, object], config=ConfigDict(strict=True)
)


@dataclass(frozen=True, slots=True)
class PolicyPayload:
    """Inference-only actor state."""

    checkpoint_id: str
    policy_state: dict[str, Tensor]


@dataclass(frozen=True, slots=True)
class TrainerPayload:
    """Training-only critic and joint optimizer state."""

    checkpoint_id: str
    value_state: dict[str, Tensor]
    optimizer_state: dict[str, object]


def write_policy_payload(
    *,
    path: Path,
    checkpoint_id: str,
    policy_state: Mapping[str, Tensor],
) -> _result.Ok[None] | _result.Rejected:
    """Atomically write the immutable policy payload."""
    valid = _validate_tensor_state(policy_state, path)
    if isinstance(valid, _result.Rejected):
        return valid
    return _write_payload(
        path=path,
        payload={
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_id": checkpoint_id,
            "policy_state": dict(policy_state),
        },
        label="policy",
    )


def write_trainer_payload(
    *,
    path: Path,
    checkpoint_id: str,
    value_state: Mapping[str, Tensor],
    optimizer_state: dict[str, object],
) -> _result.Ok[None] | _result.Rejected:
    """Atomically write the immutable trainer payload."""
    valid = _validate_tensor_state(value_state, path)
    if isinstance(valid, _result.Rejected):
        return valid
    return _write_payload(
        path=path,
        payload={
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_id": checkpoint_id,
            "value_state": dict(value_state),
            "optimizer_state": optimizer_state,
        },
        label="trainer",
    )


def read_policy_payload(
    path: Path,
) -> _result.Ok[PolicyPayload] | _result.Rejected:
    """Read exactly the schema-27 inference payload."""
    loaded = _load_payload(path)
    if isinstance(loaded, _result.Rejected):
        return loaded
    data = loaded.value
    if set(data) != {
        "schema_version",
        "checkpoint_id",
        "policy_state",
    }:
        return checkpoint_corruption(
            path, "policy payload fields do not match schema 27"
        )
    common = _validate_common(data, path)
    if isinstance(common, _result.Rejected):
        return common
    policy_state = _tensor_state_field(data, "policy_state", path)
    if isinstance(policy_state, _result.Rejected):
        return policy_state
    return _result.Ok(
        PolicyPayload(
            checkpoint_id=common.value,
            policy_state=policy_state.value,
        )
    )


def read_trainer_payload(
    path: Path,
) -> _result.Ok[TrainerPayload] | _result.Rejected:
    """Read exactly the schema-27 training payload."""
    loaded = _load_payload(path)
    if isinstance(loaded, _result.Rejected):
        return loaded
    data = loaded.value
    if set(data) != {
        "schema_version",
        "checkpoint_id",
        "value_state",
        "optimizer_state",
    }:
        return checkpoint_corruption(
            path, "trainer payload fields do not match schema 27"
        )
    common = _validate_common(data, path)
    if isinstance(common, _result.Rejected):
        return common
    value_state = _tensor_state_field(data, "value_state", path)
    if isinstance(value_state, _result.Rejected):
        return value_state
    optimizer_state = _object_dict_field(data, "optimizer_state", path)
    if isinstance(optimizer_state, _result.Rejected):
        return optimizer_state
    return _result.Ok(
        TrainerPayload(
            checkpoint_id=common.value,
            value_state=value_state.value,
            optimizer_state=optimizer_state.value,
        )
    )


def _write_payload(
    *, path: Path, payload: dict[str, object], label: str
) -> _result.Ok[None] | _result.Rejected:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except OSError, RuntimeError, pickle.PickleError:
        _discard(temporary)
        return checkpoint_corruption(
            path, f"{label} payload write failed"
        )
    return _result.Ok(None)


def _load_payload(
    path: Path,
) -> _result.Ok[dict[object, object]] | _result.Rejected:
    try:
        value = cast(
            object,
            torch.load(
                path,
                map_location=torch.device("cpu"),
                weights_only=True,
            ),
        )
        return _result.Ok(_OBJECT_DICT.validate_python(value))
    except (
        EOFError,
        RuntimeError,
        ValueError,
        pickle.UnpicklingError,
        ValidationError,
    ):
        return checkpoint_corruption(path, "payload cannot be loaded")


def _validate_common(
    data: Mapping[object, object], path: Path
) -> _result.Ok[str] | _result.Rejected:
    version = data.get("schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        return checkpoint_corruption(
            path, "payload schema version mismatch"
        )
    checkpoint_id = data.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        return checkpoint_corruption(
            path, "payload checkpoint id invalid"
        )
    return _result.Ok(checkpoint_id)


def _tensor_state_field(
    data: Mapping[object, object], field: str, path: Path
) -> _result.Ok[dict[str, Tensor]] | _result.Rejected:
    value = data.get(field)
    if not _is_tensor_state(value):
        return checkpoint_corruption(
            path, f"payload {field} is invalid"
        )
    valid = _validate_tensor_state(value, path)
    if isinstance(valid, _result.Rejected):
        return valid
    return _result.Ok(value)


def _object_dict_field(
    data: Mapping[object, object], field: str, path: Path
) -> _result.Ok[dict[str, object]] | _result.Rejected:
    value = data.get(field)
    if not isinstance(value, dict):
        return checkpoint_corruption(
            path, f"payload {field} is invalid"
        )
    items = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in items):
        return checkpoint_corruption(
            path, f"payload {field} is invalid"
        )
    return _result.Ok(cast(dict[str, object], value))


def _validate_tensor_state(
    state: Mapping[str, Tensor], path: Path
) -> _result.Ok[None] | _result.Rejected:
    for name, tensor in state.items():
        if bool(torch.isfinite(tensor).all().item()):
            continue
        return checkpoint_corruption(
            path, f"parameter {name} must be finite"
        )
    return _result.Ok(None)


def _is_tensor_state(
    value: object,
) -> TypeGuard[dict[str, Tensor]]:
    if not isinstance(value, dict):
        return False
    items = cast(dict[object, object], value)
    return all(
        isinstance(key, str) and isinstance(item, Tensor)
        for key, item in items.items()
    )


def _discard(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError, OSError:
        return


__all__ = (
    "PolicyPayload",
    "TrainerPayload",
    "read_policy_payload",
    "read_trainer_payload",
    "write_policy_payload",
    "write_trainer_payload",
)
