"""Torch checkpoint save/load for trainable models."""

from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard, cast

import torch
from torch import Tensor

from server.training.config import ModelConfig, TrainConfig
from server.training.json_types import JsonObject, JsonValue
from server.training.model import TractorPolicyModel
from server.training.ppo import PPOTrainer

type PythonRandomState = tuple[int, tuple[int, ...], float | None]

CHECKPOINT_SCHEMA_VERSION = 12
CHECKPOINT_OBJECTS_DIR = "objects"
CHECKPOINT_STATE_FILENAME = "state.pt"


@dataclass(frozen=True, slots=True)
class LoadedTrainingState:
    """Loaded model/trainer state and progress counters."""

    model: TractorPolicyModel
    trainer: PPOTrainer
    total_rounds: int
    total_updates: int


@dataclass(frozen=True, slots=True)
class TorchCheckpointMetadata:
    """Portable checkpoint metadata needed before model creation."""

    model_config: ModelConfig
    train_config: TrainConfig
    total_rounds: int
    total_updates: int


@dataclass(frozen=True, slots=True)
class TorchRngState:
    """Random generator states needed for exact training resume."""

    python_random_state: PythonRandomState
    torch_cpu_state: Tensor
    torch_cuda_states: tuple[Tensor, ...]


@dataclass(frozen=True, slots=True)
class _TorchCheckpointManifest:
    checkpoint_id: str
    state_path: Path
    state_sha256: str
    metadata: TorchCheckpointMetadata


def create_model(
    config: ModelConfig, device: torch.device
) -> TractorPolicyModel:
    """Create a model on the requested device."""
    model = TractorPolicyModel(
        d_model=config.d_model,
        layers=config.layers,
        heads=config.heads,
        dropout=config.dropout,
    )
    return model.to(device)


def create_training_state(
    *,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> LoadedTrainingState:
    """Create fresh model and PPO trainer."""
    model = create_model(model_config, device)
    trainer = PPOTrainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=device,
    )
    return LoadedTrainingState(
        model=model,
        trainer=trainer,
        total_rounds=0,
        total_updates=0,
    )


def save_torch_checkpoint(
    *,
    path: Path,
    model: TractorPolicyModel,
    trainer: PPOTrainer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    total_rounds: int,
    total_updates: int,
) -> None:
    """Atomically save trainable state."""
    assert path.suffix == ".json"
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_id = uuid.uuid4().hex
    relative_state_path = (
        Path(CHECKPOINT_OBJECTS_DIR)
        / checkpoint_id
        / CHECKPOINT_STATE_FILENAME
    )
    state_path = path.parent / relative_state_path
    state_path.parent.mkdir(parents=True, exist_ok=False)
    tmp_state_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
    tmp_manifest_path = path.with_suffix(f"{path.suffix}.tmp")
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": checkpoint_id,
        "model_state": model.state_dict(),
        "optimizer_state": trainer.optimizer_state(),
        "rng_state": _rng_state_to_payload(capture_torch_rng_state()),
    }
    torch.save(payload, tmp_state_path)
    os.replace(tmp_state_path, state_path)
    manifest = _manifest_to_json(
        _TorchCheckpointManifest(
            checkpoint_id=checkpoint_id,
            state_path=relative_state_path,
            state_sha256=_sha256_file(state_path),
            metadata=TorchCheckpointMetadata(
                model_config=model_config,
                train_config=train_config,
                total_rounds=total_rounds,
                total_updates=total_updates,
            ),
        )
    )
    tmp_manifest_path.write_text(
        f"{json.dumps(manifest, ensure_ascii=False, sort_keys=True)}\n",
        encoding="utf-8",
    )
    os.replace(tmp_manifest_path, path)


def load_torch_checkpoint(
    *,
    path: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> LoadedTrainingState:
    """Load trainable state from a torch checkpoint."""
    manifest = _read_checkpoint_manifest(path)
    metadata = manifest.metadata
    assert metadata.model_config == model_config
    state_path = _state_file_path(manifest_path=path, manifest=manifest)
    assert _sha256_file(state_path) == manifest.state_sha256
    loaded = _load_checkpoint_payload(path=state_path)
    schema_version = loaded["schema_version"]
    assert schema_version == CHECKPOINT_SCHEMA_VERSION
    checkpoint_id = loaded["checkpoint_id"]
    assert checkpoint_id == manifest.checkpoint_id
    model = create_model(model_config, device)
    model_state = loaded["model_state"]
    assert _is_tensor_state_dict(model_state)
    model.load_state_dict(model_state)
    trainer = PPOTrainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=device,
    )
    optimizer_state = loaded["optimizer_state"]
    assert _is_str_object_dict(optimizer_state)
    trainer.load_optimizer_state(optimizer_state)
    rng_payload = loaded["rng_state"]
    assert _is_object_dict(rng_payload)
    rng_state = _rng_state_from_payload(rng_payload)
    restore_torch_rng_state(rng_state)
    return LoadedTrainingState(
        model=model,
        trainer=trainer,
        total_rounds=metadata.total_rounds,
        total_updates=metadata.total_updates,
    )


def read_torch_checkpoint_metadata(
    path: Path,
) -> TorchCheckpointMetadata:
    """Read checkpoint metadata without binding to a training device."""
    return _read_checkpoint_manifest(path).metadata


def capture_torch_rng_state() -> TorchRngState:
    """Capture Python and torch RNG states for a checkpoint."""
    python_state = random.getstate()
    assert _is_python_random_state(python_state)
    cuda_states = (
        tuple(torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else ()
    )
    return TorchRngState(
        python_random_state=python_state,
        torch_cpu_state=torch.get_rng_state(),
        torch_cuda_states=cuda_states,
    )


def restore_torch_rng_state(state: TorchRngState) -> None:
    """Restore Python and torch RNG states from a checkpoint."""
    random.setstate(state.python_random_state)
    assert state.torch_cpu_state.device.type == "cpu"
    torch.set_rng_state(state.torch_cpu_state)
    if state.torch_cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(list(state.torch_cuda_states))


def _load_checkpoint_payload(
    *,
    path: Path,
) -> dict[object, object]:
    loaded: object = torch.load(
        path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    assert _is_object_dict(loaded)
    return loaded


def _manifest_to_json(manifest: _TorchCheckpointManifest) -> JsonObject:
    metadata = manifest.metadata
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": manifest.checkpoint_id,
        "state_path": manifest.state_path.as_posix(),
        "state_sha256": manifest.state_sha256,
        "model_config": metadata.model_config.to_json(),
        "train_config": metadata.train_config.to_json(),
        "total_rounds": metadata.total_rounds,
        "total_updates": metadata.total_updates,
    }


def _read_checkpoint_manifest(path: Path) -> _TorchCheckpointManifest:
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    assert _is_json_object(loaded)
    schema_version = loaded["schema_version"]
    assert schema_version == CHECKPOINT_SCHEMA_VERSION
    model_config = _json_object_field(loaded, "model_config")
    train_config = _json_object_field(loaded, "train_config")
    state_path = Path(_json_str_field(loaded, "state_path"))
    assert not state_path.is_absolute()
    assert ".." not in state_path.parts
    return _TorchCheckpointManifest(
        checkpoint_id=_json_str_field(loaded, "checkpoint_id"),
        state_path=state_path,
        state_sha256=_json_str_field(loaded, "state_sha256"),
        metadata=TorchCheckpointMetadata(
            model_config=ModelConfig.from_json(model_config),
            train_config=TrainConfig.from_json(train_config),
            total_rounds=_json_int_field(loaded, "total_rounds"),
            total_updates=_json_int_field(loaded, "total_updates"),
        ),
    )


def _state_file_path(
    *,
    manifest_path: Path,
    manifest: _TorchCheckpointManifest,
) -> Path:
    return manifest_path.parent / manifest.state_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _rng_state_to_payload(state: TorchRngState) -> dict[str, object]:
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
    data: dict[object, object],
) -> TorchRngState:
    python_version = data["python_version"]
    python_internal_state = data["python_internal_state"]
    python_gaussian = data["python_gaussian"]
    torch_cpu_state = data["torch_cpu_state"]
    torch_cuda_states = data["torch_cuda_states"]
    assert isinstance(python_version, int)
    assert _is_int_list(python_internal_state)
    assert python_gaussian is None or (
        isinstance(python_gaussian, int | float)
        and not isinstance(python_gaussian, bool)
    )
    assert isinstance(torch_cpu_state, Tensor)
    assert _is_tensor_list(torch_cuda_states)
    return TorchRngState(
        python_random_state=(
            python_version,
            tuple(python_internal_state),
            None if python_gaussian is None else float(python_gaussian),
        ),
        torch_cpu_state=torch_cpu_state,
        torch_cuda_states=tuple(torch_cuda_states),
    )


def _json_int_field(data: JsonObject, field: str) -> int:
    value = data[field]
    assert isinstance(value, int)
    return value


def _json_str_field(data: JsonObject, field: str) -> str:
    value = data[field]
    assert isinstance(value, str)
    assert value
    return value


def _json_object_field(data: JsonObject, field: str) -> JsonObject:
    value = data[field]
    assert _is_json_object(value)
    return value


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _is_str_object_dict(value: object) -> TypeGuard[dict[str, object]]:
    if not _is_object_dict(value):
        return False
    return all(isinstance(key, str) for key in value)


def _is_json_object(value: object) -> TypeGuard[JsonObject]:
    if not _is_object_dict(value):
        return False
    for key, item in value.items():
        if not isinstance(key, str):
            return False
        if not _is_json_value(item):
            return False
    return True


def _is_json_value(value: object) -> TypeGuard[JsonValue]:
    if value is None:
        return True
    if isinstance(value, str | int | float | bool):
        return True
    if _is_object_list(value):
        return all(_is_json_value(item) for item in value)
    if _is_object_dict(value):
        return all(
            isinstance(key, str) and _is_json_value(item)
            for key, item in value.items()
        )
    return False


def _is_python_random_state(
    value: object,
) -> TypeGuard[PythonRandomState]:
    if not isinstance(value, tuple):
        return False
    items = cast(tuple[object, ...], value)
    if len(items) != 3:
        return False
    version, internal_state, gaussian = items
    return (
        isinstance(version, int)
        and _is_int_tuple(internal_state)
        and (gaussian is None or isinstance(gaussian, float))
    )


def _is_int_tuple(value: object) -> TypeGuard[tuple[int, ...]]:
    if not isinstance(value, tuple):
        return False
    items = cast(tuple[object, ...], value)
    return all(isinstance(item, int) for item in items)


def _is_int_list(value: object) -> TypeGuard[list[int]]:
    if not isinstance(value, list):
        return False
    items = cast(list[object], value)
    return all(isinstance(item, int) for item in items)


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


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
