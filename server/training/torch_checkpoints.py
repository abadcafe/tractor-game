"""Torch checkpoint save/load for trainable models."""

from __future__ import annotations

import os
import random
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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    payload: dict[str, object] = {
        "schema_version": 9,
        "model_config": model_config.to_json(),
        "train_config": train_config.to_json(),
        "model_state": model.state_dict(),
        "optimizer_state": trainer.optimizer_state(),
        "rng_state": capture_torch_rng_state(),
        "total_rounds": total_rounds,
        "total_updates": total_updates,
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_torch_checkpoint(
    *,
    path: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> LoadedTrainingState:
    """Load trainable state from a torch checkpoint."""
    loaded = _load_checkpoint_payload(path=path, map_location=device)
    schema_version = loaded["schema_version"]
    assert schema_version == 9
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
    rng_state = loaded["rng_state"]
    assert isinstance(rng_state, TorchRngState)
    restore_torch_rng_state(rng_state)
    return LoadedTrainingState(
        model=model,
        trainer=trainer,
        total_rounds=_int_field(loaded, "total_rounds"),
        total_updates=_int_field(loaded, "total_updates"),
    )


def read_torch_checkpoint_metadata(
    path: Path,
) -> TorchCheckpointMetadata:
    """Read checkpoint metadata without binding to a training device."""
    loaded = _load_checkpoint_payload(
        path=path,
        map_location=torch.device("cpu"),
    )
    schema_version = loaded["schema_version"]
    assert schema_version == 9
    model_config = loaded["model_config"]
    train_config = loaded["train_config"]
    assert _is_json_object(model_config)
    assert _is_json_object(train_config)
    return TorchCheckpointMetadata(
        model_config=ModelConfig.from_json(model_config),
        train_config=TrainConfig.from_json(train_config),
        total_rounds=_int_field(loaded, "total_rounds"),
        total_updates=_int_field(loaded, "total_updates"),
    )


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
    torch.set_rng_state(state.torch_cpu_state)
    if state.torch_cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(list(state.torch_cuda_states))


def _load_checkpoint_payload(
    *,
    path: Path,
    map_location: torch.device,
) -> dict[object, object]:
    loaded: object = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )
    assert _is_object_dict(loaded)
    return loaded


def _int_field(data: dict[object, object], field: str) -> int:
    value = data[field]
    assert isinstance(value, int)
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


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


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
