"""Torch checkpoint save/load for trainable models."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
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

CHECKPOINT_SCHEMA_VERSION = 15
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


class CheckpointCorruptionError(Exception):
    """Checkpoint files are missing, malformed, or inconsistent."""


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
    )
    return model.to(device)


def create_training_state(
    *,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> LoadedTrainingState:
    """Create fresh model and PPO trainer."""
    seed_training_rng(train_config.seed)
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
    manifest_paths: tuple[Path, ...],
    model: TractorPolicyModel,
    trainer: PPOTrainer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    total_rounds: int,
    total_updates: int,
    retained_update_count: int,
) -> None:
    """Atomically save trainable state behind one or more manifests."""
    assert retained_update_count >= 0
    checkpoint_dir = _checkpoint_dir_from_manifest_paths(manifest_paths)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_id = uuid.uuid4().hex
    relative_state_path = (
        Path(CHECKPOINT_OBJECTS_DIR)
        / checkpoint_id
        / CHECKPOINT_STATE_FILENAME
    )
    state_path = checkpoint_dir / relative_state_path
    state_path.parent.mkdir(parents=True, exist_ok=False)
    tmp_state_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": checkpoint_id,
        "model_state": model.state_dict(),
        "optimizer_state": trainer.optimizer_state(),
        "rng_state": _rng_state_to_payload(capture_torch_rng_state()),
    }
    torch.save(payload, tmp_state_path)
    os.replace(tmp_state_path, state_path)
    manifest = _TorchCheckpointManifest(
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
    for manifest_path in manifest_paths:
        _write_checkpoint_manifest(
            path=manifest_path, manifest=manifest
        )
    _prune_torch_checkpoints(
        checkpoint_dir=checkpoint_dir,
        retained_update_count=retained_update_count,
    )


def _prune_torch_checkpoints(
    *,
    checkpoint_dir: Path,
    retained_update_count: int,
) -> None:
    """Delete expired manifests and unreferenced state objects."""
    assert retained_update_count >= 0
    if not checkpoint_dir.exists():
        return
    assert checkpoint_dir.is_dir()
    _prune_update_manifests(
        checkpoint_dir=checkpoint_dir,
        retained_update_count=retained_update_count,
    )
    _remove_unreferenced_checkpoint_objects(checkpoint_dir)


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
    assert metadata.train_config.seed == train_config.seed
    state_path = _state_file_path(manifest_path=path, manifest=manifest)
    if _sha256_file(state_path) != manifest.state_sha256:
        raise _checkpoint_corruption(
            state_path,
            f"state sha256 does not match manifest {path}",
        )
    loaded = _load_checkpoint_payload(path=state_path)
    schema_version = loaded["schema_version"]
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise _checkpoint_corruption(
            state_path,
            "state payload schema version mismatch",
        )
    checkpoint_id = loaded["checkpoint_id"]
    if checkpoint_id != manifest.checkpoint_id:
        raise _checkpoint_corruption(
            state_path,
            f"state checkpoint id does not match manifest {path}",
        )
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


def seed_training_rng(seed: int) -> None:
    """Seed Python and torch RNGs for a fresh training run."""
    assert seed >= 0
    random.seed(seed)
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)
    torch.set_rng_state(cpu_generator.get_state())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _checkpoint_dir_from_manifest_paths(
    manifest_paths: tuple[Path, ...],
) -> Path:
    assert manifest_paths
    checkpoint_dir = manifest_paths[0].parent
    seen_paths: set[Path] = set()
    for path in manifest_paths:
        assert path.suffix == ".json"
        assert path.parent == checkpoint_dir
        assert path not in seen_paths
        seen_paths.add(path)
    return checkpoint_dir


def _write_checkpoint_manifest(
    *,
    path: Path,
    manifest: _TorchCheckpointManifest,
) -> None:
    manifest_json = _manifest_to_json(manifest)
    manifest_text = json.dumps(
        manifest_json,
        ensure_ascii=False,
        sort_keys=True,
    )
    tmp_manifest_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_manifest_path.write_text(
        f"{manifest_text}\n",
        encoding="utf-8",
    )
    os.replace(tmp_manifest_path, path)


def _prune_update_manifests(
    *,
    checkpoint_dir: Path,
    retained_update_count: int,
) -> None:
    update_paths = _update_checkpoint_manifest_paths(checkpoint_dir)
    if retained_update_count == 0:
        expired_paths = tuple(path for _, path in update_paths)
    else:
        expired_paths = tuple(
            path for _, path in update_paths[:-retained_update_count]
        )
    for path in expired_paths:
        path.unlink()


def _update_checkpoint_manifest_paths(
    checkpoint_dir: Path,
) -> tuple[tuple[int, Path], ...]:
    paths: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("update-*.json"):
        update_number = _update_number_from_manifest_path(path)
        if update_number is not None:
            paths.append((update_number, path))
    return tuple(sorted(paths, key=lambda item: item[0]))


def _update_number_from_manifest_path(path: Path) -> int | None:
    if path.suffix != ".json":
        return None
    update_text = path.stem.removeprefix("update-")
    if update_text == path.stem:
        return None
    if not update_text.isdecimal():
        return None
    return int(update_text)


def _remove_unreferenced_checkpoint_objects(
    checkpoint_dir: Path,
) -> None:
    objects_dir = checkpoint_dir / CHECKPOINT_OBJECTS_DIR
    if not objects_dir.exists():
        return
    assert objects_dir.is_dir()
    live_checkpoint_ids = _live_checkpoint_ids(checkpoint_dir)
    for child in tuple(objects_dir.iterdir()):
        assert not child.is_symlink()
        if child.name in live_checkpoint_ids:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _live_checkpoint_ids(checkpoint_dir: Path) -> set[str]:
    checkpoint_ids: set[str] = set()
    for path in _managed_checkpoint_manifest_paths(checkpoint_dir):
        manifest = _read_checkpoint_manifest(path)
        _assert_manifest_state_path(manifest)
        checkpoint_ids.add(manifest.checkpoint_id)
    return checkpoint_ids


def _managed_checkpoint_manifest_paths(
    checkpoint_dir: Path,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    latest = checkpoint_dir / "latest.json"
    if latest.exists():
        paths.append(latest)
    paths.extend(
        path
        for _, path in _update_checkpoint_manifest_paths(checkpoint_dir)
    )
    return tuple(paths)


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
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _checkpoint_corruption(
            path, "manifest is not valid JSON"
        ) from exc
    if not _is_json_object(loaded):
        raise _checkpoint_corruption(
            path, "manifest root is not an object"
        )
    schema_version = _json_int_field(loaded, "schema_version", path)
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise _checkpoint_corruption(
            path,
            "manifest schema version mismatch",
        )
    model_config = _json_object_field(loaded, "model_config", path)
    train_config = _json_object_field(loaded, "train_config", path)
    state_path = Path(_json_str_field(loaded, "state_path", path))
    if state_path.is_absolute() or ".." in state_path.parts:
        raise _checkpoint_corruption(
            path, "manifest state path escapes checkpoint directory"
        )
    manifest = _TorchCheckpointManifest(
        checkpoint_id=_json_str_field(loaded, "checkpoint_id", path),
        state_path=state_path,
        state_sha256=_json_str_field(loaded, "state_sha256", path),
        metadata=TorchCheckpointMetadata(
            model_config=ModelConfig.from_json(model_config),
            train_config=TrainConfig.from_json(train_config),
            total_rounds=_json_int_field(loaded, "total_rounds", path),
            total_updates=_json_int_field(
                loaded, "total_updates", path
            ),
        ),
    )
    _assert_manifest_state_path(manifest)
    return manifest


def _assert_manifest_state_path(
    manifest: _TorchCheckpointManifest,
) -> None:
    expected = (
        Path(CHECKPOINT_OBJECTS_DIR)
        / manifest.checkpoint_id
        / CHECKPOINT_STATE_FILENAME
    )
    if manifest.state_path != expected:
        raise CheckpointCorruptionError(
            "checkpoint corruption: manifest state path does not match "
            f"checkpoint id {manifest.checkpoint_id}"
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


def _json_int_field(data: JsonObject, field: str, path: Path) -> int:
    if field not in data:
        raise _checkpoint_corruption(path, f"manifest missing {field}")
    value = data[field]
    if not isinstance(value, int):
        raise _checkpoint_corruption(
            path, f"manifest {field} is not an int"
        )
    return value


def _json_str_field(data: JsonObject, field: str, path: Path) -> str:
    if field not in data:
        raise _checkpoint_corruption(path, f"manifest missing {field}")
    value = data[field]
    if not isinstance(value, str) or not value:
        raise _checkpoint_corruption(
            path, f"manifest {field} is not a string"
        )
    return value


def _json_object_field(
    data: JsonObject, field: str, path: Path
) -> JsonObject:
    if field not in data:
        raise _checkpoint_corruption(path, f"manifest missing {field}")
    value = data[field]
    if not _is_json_object(value):
        raise _checkpoint_corruption(
            path, f"manifest {field} is not an object"
        )
    return value


def _checkpoint_corruption(
    path: Path, reason: str
) -> CheckpointCorruptionError:
    return CheckpointCorruptionError(
        f"checkpoint corruption: {path}: {reason}"
    )


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
