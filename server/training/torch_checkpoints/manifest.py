"""Manifest codec for torch training checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeGuard, cast

from server import result as _result
from server.training.json_types import JsonObject, JsonValue
from server.training.torch_checkpoints.filesystem import (
    validate_checkpoint_manifest_file,
)
from server.training.torch_checkpoints.schema import (
    CHECKPOINT_OBJECTS_DIR,
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_STATE_FILENAME,
    CheckpointManifest,
    TorchCheckpointMetadata,
    checkpoint_corruption,
)
from server.training.torch_checkpoints.validation import (
    model_config_from_json,
    train_config_from_json,
)


def checkpoint_dir_from_manifest_paths(
    manifest_paths: tuple[Path, ...],
) -> _result.Ok[Path] | _result.Rejected:
    """Validate and return the shared dir for managed manifests."""
    assert manifest_paths
    checkpoint_dir = manifest_paths[0].parent
    seen_paths: set[Path] = set()
    for path in manifest_paths:
        if path.parent != checkpoint_dir:
            return checkpoint_corruption(
                path,
                "manifest paths must share one checkpoint directory",
            )
        if path in seen_paths:
            return checkpoint_corruption(
                path, "manifest path is duplicated"
            )
        if not _is_managed_manifest_path(path):
            return checkpoint_corruption(
                path,
                "manifest path must be latest.json or "
                "update-<positive n>.json",
            )
        seen_paths.add(path)
    return _result.Ok(value=checkpoint_dir)


def write_checkpoint_manifest(
    *,
    path: Path,
    manifest: CheckpointManifest,
) -> _result.Ok[None] | _result.Rejected:
    """Atomically write one manifest file."""
    manifest_json = _manifest_to_json(manifest)
    manifest_text = json.dumps(
        manifest_json,
        ensure_ascii=False,
        sort_keys=True,
    )
    tmp_manifest_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        tmp_manifest_path.write_text(
            f"{manifest_text}\n",
            encoding="utf-8",
        )
        os.replace(tmp_manifest_path, path)
    except OSError:
        _discard_file(tmp_manifest_path)
        return checkpoint_corruption(path, "manifest write failed")
    return _result.Ok(value=None)


def _discard_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def read_checkpoint_manifest(
    path: Path,
) -> _result.Ok[CheckpointManifest] | _result.Rejected:
    """Read and validate one checkpoint manifest."""
    path_check = validate_checkpoint_manifest_file(path)
    if isinstance(path_check, _result.Rejected):
        return path_check
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return checkpoint_corruption(path, "manifest file is missing")
    except UnicodeDecodeError:
        return checkpoint_corruption(
            path, "manifest is not valid UTF-8"
        )
    except json.JSONDecodeError:
        return checkpoint_corruption(path, "manifest is not valid JSON")
    except OSError:
        return checkpoint_corruption(
            path, "manifest file is not readable"
        )
    if not _is_json_object(loaded):
        return checkpoint_corruption(
            path, "manifest root is not an object"
        )
    schema_version_result = _json_int_field(
        loaded, "schema_version", path
    )
    if isinstance(schema_version_result, _result.Rejected):
        return schema_version_result
    if schema_version_result.value != CHECKPOINT_SCHEMA_VERSION:
        return checkpoint_corruption(
            path,
            "manifest schema version mismatch",
        )
    model_config_result = _json_object_field(
        loaded, "model_config", path
    )
    if isinstance(model_config_result, _result.Rejected):
        return model_config_result
    train_config_result = _json_object_field(
        loaded, "train_config", path
    )
    if isinstance(train_config_result, _result.Rejected):
        return train_config_result
    state_path_result = _json_str_field(loaded, "state_path", path)
    if isinstance(state_path_result, _result.Rejected):
        return state_path_result
    state_path = Path(state_path_result.value)
    if state_path.is_absolute() or ".." in state_path.parts:
        return checkpoint_corruption(
            path, "manifest state path escapes checkpoint directory"
        )
    checkpoint_id_result = _json_str_field(
        loaded, "checkpoint_id", path
    )
    if isinstance(checkpoint_id_result, _result.Rejected):
        return checkpoint_id_result
    state_sha_result = _json_str_field(loaded, "state_sha256", path)
    if isinstance(state_sha_result, _result.Rejected):
        return state_sha_result
    total_rounds_result = _json_non_negative_int_field(
        loaded, "total_rounds", path
    )
    if isinstance(total_rounds_result, _result.Rejected):
        return total_rounds_result
    total_samples_result = _json_non_negative_int_field(
        loaded, "total_samples", path
    )
    if isinstance(total_samples_result, _result.Rejected):
        return total_samples_result
    total_updates_result = _json_non_negative_int_field(
        loaded, "total_updates", path
    )
    if isinstance(total_updates_result, _result.Rejected):
        return total_updates_result
    parsed_model_config = model_config_from_json(
        model_config_result.value, path
    )
    if isinstance(parsed_model_config, _result.Rejected):
        return parsed_model_config
    parsed_train_config = train_config_from_json(
        train_config_result.value, path
    )
    if isinstance(parsed_train_config, _result.Rejected):
        return parsed_train_config
    manifest = CheckpointManifest(
        checkpoint_id=checkpoint_id_result.value,
        state_path=state_path,
        state_sha256=state_sha_result.value,
        metadata=TorchCheckpointMetadata(
            model_config=parsed_model_config.value,
            train_config=parsed_train_config.value,
            total_rounds=total_rounds_result.value,
            total_samples=total_samples_result.value,
            total_updates=total_updates_result.value,
        ),
    )
    state_path_check = _assert_manifest_state_path(
        manifest=manifest,
        path=path,
    )
    if isinstance(state_path_check, _result.Rejected):
        return state_path_check
    return _result.Ok(value=manifest)


def manifest_state_file_path(
    *,
    manifest_path: Path,
    manifest: CheckpointManifest,
) -> Path:
    """Return the state file path for a manifest."""
    return manifest_path.parent / manifest.state_path


def managed_checkpoint_manifest_paths(
    checkpoint_dir: Path,
) -> tuple[Path, ...]:
    """Return managed latest/update manifest paths."""
    paths: list[Path] = []
    latest = checkpoint_dir / "latest.json"
    if latest.exists():
        paths.append(latest)
    paths.extend(
        path
        for _, path in update_checkpoint_manifest_paths(checkpoint_dir)
    )
    return tuple(paths)


def update_checkpoint_manifest_paths(
    checkpoint_dir: Path,
) -> tuple[tuple[int, Path], ...]:
    """Return managed update manifests sorted by update number."""
    paths: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("update-*.json"):
        update_number = managed_update_number_from_manifest_path(path)
        if update_number is not None:
            paths.append((update_number, path))
    return tuple(sorted(paths, key=lambda item: item[0]))


def _manifest_to_json(manifest: CheckpointManifest) -> JsonObject:
    metadata = manifest.metadata
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": manifest.checkpoint_id,
        "state_path": manifest.state_path.as_posix(),
        "state_sha256": manifest.state_sha256,
        "model_config": metadata.model_config.to_json(),
        "train_config": metadata.train_config.to_json(),
        "total_rounds": metadata.total_rounds,
        "total_samples": metadata.total_samples,
        "total_updates": metadata.total_updates,
    }


def _assert_manifest_state_path(
    *,
    manifest: CheckpointManifest,
    path: Path,
) -> _result.Ok[None] | _result.Rejected:
    expected = (
        Path(CHECKPOINT_OBJECTS_DIR)
        / manifest.checkpoint_id
        / CHECKPOINT_STATE_FILENAME
    )
    if manifest.state_path != expected:
        return checkpoint_corruption(
            path,
            "manifest state path does not match checkpoint id "
            f"{manifest.checkpoint_id}",
        )
    return _result.Ok(value=None)


def managed_update_number_from_manifest_path(path: Path) -> int | None:
    """Return the update number for canonical update manifests only."""
    if path.suffix != ".json":
        return None
    update_text = path.stem.removeprefix("update-")
    if update_text == path.stem:
        return None
    if not update_text.isdecimal():
        return None
    update_number = int(update_text)
    if update_number <= 0:
        return None
    if path.name != f"update-{update_number}.json":
        return None
    return update_number


def _is_managed_manifest_path(path: Path) -> bool:
    if path.name == "latest.json":
        return True
    return managed_update_number_from_manifest_path(path) is not None


def _json_int_field(
    data: JsonObject,
    field: str,
    path: Path,
) -> _result.Ok[int] | _result.Rejected:
    if field not in data:
        return checkpoint_corruption(path, f"manifest missing {field}")
    value = data[field]
    if not isinstance(value, int) or isinstance(value, bool):
        return checkpoint_corruption(
            path, f"manifest {field} is not an int"
        )
    return _result.Ok(value=value)


def _json_non_negative_int_field(
    data: JsonObject,
    field: str,
    path: Path,
) -> _result.Ok[int] | _result.Rejected:
    value_result = _json_int_field(data, field, path)
    if isinstance(value_result, _result.Rejected):
        return value_result
    if value_result.value < 0:
        return checkpoint_corruption(
            path, f"manifest {field} is negative"
        )
    return value_result


def _json_str_field(
    data: JsonObject,
    field: str,
    path: Path,
) -> _result.Ok[str] | _result.Rejected:
    if field not in data:
        return checkpoint_corruption(path, f"manifest missing {field}")
    value = data[field]
    if not isinstance(value, str) or not value:
        return checkpoint_corruption(
            path, f"manifest {field} is not a string"
        )
    return _result.Ok(value=value)


def _json_object_field(
    data: JsonObject, field: str, path: Path
) -> _result.Ok[JsonObject] | _result.Rejected:
    if field not in data:
        return checkpoint_corruption(path, f"manifest missing {field}")
    value = data[field]
    if not _is_json_object(value):
        return checkpoint_corruption(
            path, f"manifest {field} is not an object"
        )
    return _result.Ok(value=value)


def _is_json_object(value: object) -> TypeGuard[JsonObject]:
    if not isinstance(value, dict):
        return False
    items = cast(dict[object, object], value)
    for key, item in items.items():
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
    if isinstance(value, list):
        items = cast(list[object], value)
        return all(_is_json_value(item) for item in items)
    if isinstance(value, dict):
        items = cast(dict[object, object], value)
        return all(
            isinstance(key, str) and _is_json_value(item)
            for key, item in items.items()
        )
    return False
