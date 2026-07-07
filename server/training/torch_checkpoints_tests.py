"""Tests for torch training checkpoint metadata."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import TypeGuard

import pytest
import torch

from server.result import Ok, Rejected
from server.training import training_state
from server.training.config import ModelConfig, TrainConfig
from server.training.model import TractorPolicyModel
from server.training.ppo import PPOTrainer
from server.training.runtime import (
    ExecutionConfig,
    ExecutionTimeouts,
    ModelRankKind,
    ModelRankPlacement,
)
from server.training.torch_checkpoints import (
    TorchCheckpointMetadata,
)
from server.training.torch_checkpoints import (
    load_torch_checkpoint as _load_torch_checkpoint,
)
from server.training.torch_checkpoints import (
    pruning as _checkpoint_pruning,
)
from server.training.torch_checkpoints import (
    read_torch_checkpoint_metadata as _read_torch_checkpoint_metadata,
)
from server.training.torch_checkpoints import save as _checkpoint_save
from server.training.torch_checkpoints import (
    save_torch_checkpoint as _save_torch_checkpoint,
)
from server.training.torch_checkpoints.schema import CheckpointManifest
from server.training.train import (
    ExecutionConfigOverrides,
    TrainConfigOverrides,
    resolve_execution_config,
    resolve_model_config,
    resolve_train_config,
)
from server.training.training_state import (
    LoadedTrainingState,
)
from server.training.training_state import (
    create_training_state as _create_training_state,
)


def create_training_state(
    *,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig | None = None,
    device: torch.device,
) -> LoadedTrainingState:
    """Create state through the execution-config public boundary."""
    resolved_execution_config = _execution_config_for_device(
        device=device,
        execution_config=execution_config,
    )
    return _create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=resolved_execution_config,
    )


def test_torch_checkpoint_metadata_drives_resume_model_config(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig(
        learning_rate=0.0003,
    )
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"

    save_torch_checkpoint(
        manifest_paths=(path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=7,
        total_samples=0,
        total_updates=3,
        retained_update_count=5,
    )

    metadata = read_torch_checkpoint_metadata(path)
    assert metadata.model_config == model_config
    assert metadata.train_config == train_config
    assert metadata.total_rounds == 7
    assert metadata.total_updates == 3
    resolved_model_config = resolve_model_config(
        cli_model_config=ModelConfig(d_model=128),
        resume_path=path,
    )
    assert isinstance(resolved_model_config, Ok)
    assert resolved_model_config.value == model_config


def test_read_metadata_uses_manifest_without_torch_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=11,
        total_samples=0,
        total_updates=5,
        retained_update_count=5,
    )

    def fail_torch_load(*args: object, **kwargs: object) -> object:
        assert False

    monkeypatch.setattr(torch, "load", fail_torch_load)

    metadata = read_torch_checkpoint_metadata(path)

    assert metadata.model_config == model_config
    assert metadata.train_config == train_config
    assert metadata.total_rounds == 11
    assert metadata.total_updates == 5


def test_torch_checkpoint_save_rejects_payload_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )

    def fail_torch_save(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail_torch_save)

    result = _save_torch_checkpoint(
        manifest_paths=(tmp_path / "latest.json",),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=1,
        total_samples=0,
        total_updates=1,
        retained_update_count=5,
    )

    assert isinstance(result, Rejected)
    assert "state payload write failed" in result.reason
    assert _state_paths(tmp_path) == ()
    assert _object_entries(tmp_path) == ()


def test_torch_checkpoint_save_rejects_checkpoint_dir_symlink(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    target_dir = tmp_path / "external-checkpoints"
    target_dir.mkdir()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.symlink_to(target_dir, target_is_directory=True)

    result = _save_torch_checkpoint(
        manifest_paths=(checkpoint_dir / "latest.json",),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=1,
        total_samples=0,
        total_updates=1,
        retained_update_count=5,
    )

    assert isinstance(result, Rejected)
    assert "checkpoint path is a symlink" in result.reason
    assert not (target_dir / "latest.json").exists()
    assert not (target_dir / "objects").exists()


def test_torch_checkpoint_save_rejects_objects_dir_symlink(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    target_dir = tmp_path / "external-objects"
    target_dir.mkdir()
    objects_dir = tmp_path / "objects"
    objects_dir.symlink_to(target_dir, target_is_directory=True)

    result = _save_torch_checkpoint(
        manifest_paths=(tmp_path / "latest.json",),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=1,
        total_samples=0,
        total_updates=1,
        retained_update_count=5,
    )

    assert isinstance(result, Rejected)
    assert "checkpoint objects path is a symlink" in result.reason
    assert not (tmp_path / "latest.json").exists()
    assert tuple(target_dir.iterdir()) == ()


def test_torch_checkpoint_save_rolls_back_manifest_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    latest_path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(latest_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=1,
        total_samples=0,
        total_updates=1,
        retained_update_count=5,
    )
    latest_before = latest_path.read_bytes()
    state_paths_before = _state_paths(tmp_path)
    original_writer = _checkpoint_save.write_checkpoint_manifest
    write_count = 0

    def fail_second_manifest_write(
        *,
        path: Path,
        manifest: CheckpointManifest,
    ) -> Ok[None] | Rejected:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            reason = (
                f"checkpoint corruption: {path}: manifest write failed"
            )
            return Rejected(reason=reason)
        return original_writer(path=path, manifest=manifest)

    monkeypatch.setattr(
        _checkpoint_save,
        "write_checkpoint_manifest",
        fail_second_manifest_write,
    )
    update_path = tmp_path / "update-2.json"

    result = _save_torch_checkpoint(
        manifest_paths=(update_path, latest_path),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=2,
        total_samples=0,
        total_updates=2,
        retained_update_count=5,
    )

    assert isinstance(result, Rejected)
    assert "manifest write failed" in result.reason
    assert write_count == 2
    assert latest_path.read_bytes() == latest_before
    assert not update_path.exists()
    assert _state_paths(tmp_path) == state_paths_before


def test_torch_checkpoint_save_reports_post_commit_prune_unlink_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    latest_path = tmp_path / "latest.json"
    update_path = tmp_path / "update-1.json"
    save_torch_checkpoint(
        manifest_paths=(update_path, latest_path),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=1,
        total_samples=0,
        total_updates=1,
        retained_update_count=5,
    )
    original_unlink = Path.unlink

    def fail_update_unlink(
        self: Path,
        missing_ok: bool = False,
    ) -> None:
        if self.name == "update-1.json":
            raise OSError("busy")
        original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_update_unlink)

    result = _save_torch_checkpoint(
        manifest_paths=(latest_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=2,
        total_samples=0,
        total_updates=2,
        retained_update_count=0,
    )

    assert isinstance(result, Ok)
    prune_failure = result.value.post_commit_prune_failure
    assert isinstance(prune_failure, Rejected)
    assert "expired update manifest cannot be deleted" in (
        prune_failure.reason
    )
    metadata = read_torch_checkpoint_metadata(latest_path)
    assert metadata.total_rounds == 2
    assert metadata.total_updates == 2
    assert update_path.exists()


def test_torch_checkpoint_save_rejects_prune_symlink_before_commit(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    latest_path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(latest_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=1,
        total_samples=0,
        total_updates=1,
        retained_update_count=5,
    )
    latest_before = latest_path.read_bytes()
    state_paths_before = _state_paths(tmp_path)
    target = tmp_path / "external-state"
    target.mkdir()
    symlink_path = tmp_path / "objects" / "symlink-object"
    symlink_path.symlink_to(target, target_is_directory=True)

    result = _save_torch_checkpoint(
        manifest_paths=(latest_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=2,
        total_samples=0,
        total_updates=2,
        retained_update_count=5,
    )

    assert isinstance(result, Rejected)
    assert "checkpoint object is a symlink" in result.reason
    assert latest_path.read_bytes() == latest_before
    assert _state_paths(tmp_path) == state_paths_before


def test_torch_checkpoint_save_rejects_retained_symlink_before_commit(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    latest_path = tmp_path / "latest.json"
    update_path = tmp_path / "update-1.json"
    save_torch_checkpoint(
        manifest_paths=(update_path, latest_path),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=1,
        total_samples=0,
        total_updates=1,
        retained_update_count=5,
    )
    latest_before = latest_path.read_bytes()
    retained_object_dir = _single_state_path(latest_path).parent
    retained_object_name = retained_object_dir.name
    target = tmp_path / "external-state"
    target.mkdir()
    shutil.rmtree(retained_object_dir)
    retained_object_dir.symlink_to(target, target_is_directory=True)

    result = _save_torch_checkpoint(
        manifest_paths=(latest_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=2,
        total_samples=0,
        total_updates=2,
        retained_update_count=1,
    )

    assert isinstance(result, Rejected)
    assert "checkpoint object is a symlink" in result.reason
    assert latest_path.read_bytes() == latest_before
    assert tuple(path.name for path in _object_entries(tmp_path)) == (
        retained_object_name,
    )


def test_torch_checkpoint_save_rejects_retained_file_before_commit(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    latest_path = tmp_path / "latest.json"
    update_path = tmp_path / "update-1.json"
    save_torch_checkpoint(
        manifest_paths=(update_path, latest_path),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=1,
        total_samples=0,
        total_updates=1,
        retained_update_count=5,
    )
    latest_before = latest_path.read_bytes()
    retained_object_dir = _single_state_path(latest_path).parent
    retained_object_name = retained_object_dir.name
    shutil.rmtree(retained_object_dir)
    retained_object_dir.write_bytes(
        b"not-a-checkpoint-object-directory"
    )

    result = _save_torch_checkpoint(
        manifest_paths=(latest_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=2,
        total_samples=0,
        total_updates=2,
        retained_update_count=1,
    )

    assert isinstance(result, Rejected)
    assert "checkpoint object is not a directory" in result.reason
    assert latest_path.read_bytes() == latest_before
    assert tuple(path.name for path in _object_entries(tmp_path)) == (
        retained_object_name,
    )


def test_torch_checkpoint_save_reports_post_commit_prune_rmtree_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    latest_path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(latest_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=1,
        total_samples=0,
        total_updates=1,
        retained_update_count=5,
    )
    orphan_dir = tmp_path / "objects" / "orphan"
    orphan_dir.mkdir()

    def fail_rmtree(path: Path) -> None:
        assert path == orphan_dir
        raise OSError("busy")

    monkeypatch.setattr(
        _checkpoint_pruning.shutil, "rmtree", fail_rmtree
    )

    result = _save_torch_checkpoint(
        manifest_paths=(latest_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=2,
        total_samples=0,
        total_updates=2,
        retained_update_count=5,
    )

    assert isinstance(result, Ok)
    prune_failure = result.value.post_commit_prune_failure
    assert isinstance(prune_failure, Rejected)
    assert "checkpoint object cannot be deleted" in prune_failure.reason
    metadata = read_torch_checkpoint_metadata(latest_path)
    assert metadata.total_rounds == 2
    assert metadata.total_updates == 2
    assert orphan_dir.exists()


def test_torch_checkpoint_state_payload_is_weights_only_safe(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=11,
        total_samples=0,
        total_updates=5,
        retained_update_count=5,
    )

    loaded: object = torch.load(
        _single_state_path(path),
        map_location=torch.device("cpu"),
        weights_only=True,
    )

    assert isinstance(loaded, dict)
    assert loaded["schema_version"] == 19
    assert isinstance(loaded["checkpoint_id"], str)
    assert "model_config" not in loaded
    assert "train_config" not in loaded
    assert "total_rounds" not in loaded
    assert "total_updates" not in loaded


def test_torch_checkpoint_alias_manifests_share_one_state_object(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    latest_path = tmp_path / "latest.json"
    update_path = tmp_path / "update-5.json"

    save_torch_checkpoint(
        manifest_paths=(update_path, latest_path),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=13,
        total_samples=0,
        total_updates=5,
        retained_update_count=5,
    )

    assert latest_path.exists()
    assert update_path.exists()
    assert _state_paths(tmp_path) == (_single_state_path(latest_path),)
    latest_metadata = read_torch_checkpoint_metadata(latest_path)
    update_metadata = read_torch_checkpoint_metadata(update_path)
    assert latest_metadata == update_metadata
    assert latest_metadata.total_rounds == 13
    assert latest_metadata.total_updates == 5


def test_torch_checkpoint_save_rejects_unmanaged_manifest_path(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    alias_path = tmp_path / "alias.json"

    result = _save_torch_checkpoint(
        manifest_paths=(alias_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=13,
        total_samples=0,
        total_updates=5,
        retained_update_count=5,
    )

    assert isinstance(result, Rejected)
    assert "latest.json or update-<positive n>.json" in result.reason
    assert not alias_path.exists()
    assert not (tmp_path / "objects").exists()


def test_torch_checkpoint_save_rejects_zero_update_manifest_path(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    update_path = tmp_path / "update-0.json"

    result = _save_torch_checkpoint(
        manifest_paths=(update_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=13,
        total_samples=0,
        total_updates=0,
        retained_update_count=5,
    )

    assert isinstance(result, Rejected)
    assert "update-<positive n>.json" in result.reason
    assert not update_path.exists()
    assert not (tmp_path / "objects").exists()


def test_torch_checkpoint_save_rejects_update_number_mismatch(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    latest_path = tmp_path / "latest.json"
    update_path = tmp_path / "update-4.json"

    result = _save_torch_checkpoint(
        manifest_paths=(update_path, latest_path),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=13,
        total_samples=0,
        total_updates=5,
        retained_update_count=5,
    )

    assert isinstance(result, Rejected)
    assert "update manifest number must equal total_updates" in (
        result.reason
    )
    assert not update_path.exists()
    assert not latest_path.exists()
    assert not (tmp_path / "objects").exists()


def test_torch_checkpoint_save_removes_overwritten_latest_object(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    latest_path = tmp_path / "latest.json"

    save_torch_checkpoint(
        manifest_paths=(latest_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=1,
        total_samples=0,
        total_updates=1,
        retained_update_count=0,
    )
    first_state_paths = _state_paths(tmp_path)
    assert len(first_state_paths) == 1
    save_torch_checkpoint(
        manifest_paths=(latest_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=2,
        total_samples=0,
        total_updates=2,
        retained_update_count=0,
    )

    assert len(_state_paths(tmp_path)) == 1
    assert _state_paths(tmp_path) != first_state_paths
    metadata = read_torch_checkpoint_metadata(latest_path)
    assert metadata.total_rounds == 2
    assert metadata.total_updates == 2


def test_torch_checkpoint_save_keeps_latest_and_recent_updates(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    latest_path = tmp_path / "latest.json"

    for update_number in range(1, 4):
        update_path = tmp_path / f"update-{update_number}.json"
        save_torch_checkpoint(
            manifest_paths=(update_path, latest_path),
            model=state.model,
            trainer=state.trainer,
            model_config=model_config,
            train_config=train_config,
            total_rounds=update_number,
            total_samples=0,
            total_updates=update_number,
            retained_update_count=2,
        )

    assert not (tmp_path / "update-1.json").exists()
    assert (tmp_path / "update-2.json").exists()
    assert (tmp_path / "update-3.json").exists()
    assert latest_path.exists()
    assert len(_state_paths(tmp_path)) == 2
    latest_metadata = read_torch_checkpoint_metadata(latest_path)
    old_update_metadata = read_torch_checkpoint_metadata(
        tmp_path / "update-2.json"
    )
    assert latest_metadata.total_updates == 3
    assert old_update_metadata.total_updates == 2


def test_torch_checkpoint_save_ignores_unmanaged_json(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    latest_path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(latest_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=1,
        total_samples=0,
        total_updates=1,
        retained_update_count=5,
    )
    (tmp_path / "notes.json").write_text(
        "{not checkpoint json", encoding="utf-8"
    )

    save_torch_checkpoint(
        manifest_paths=(latest_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=2,
        total_samples=0,
        total_updates=2,
        retained_update_count=5,
    )

    metadata = read_torch_checkpoint_metadata(latest_path)
    assert metadata.total_rounds == 2
    assert (tmp_path / "notes.json").exists()


def test_torch_checkpoint_save_ignores_noncanonical_update_json(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    latest_path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(latest_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=1,
        total_samples=0,
        total_updates=1,
        retained_update_count=5,
    )
    noncanonical_path = tmp_path / "update-001.json"
    noncanonical_path.write_text(
        "{not checkpoint json", encoding="utf-8"
    )
    zero_update_path = tmp_path / "update-0.json"
    zero_update_path.write_text(
        "{not checkpoint json", encoding="utf-8"
    )
    update_path = tmp_path / "update-2.json"

    save_torch_checkpoint(
        manifest_paths=(update_path, latest_path),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=2,
        total_samples=0,
        total_updates=2,
        retained_update_count=5,
    )

    latest_metadata = read_torch_checkpoint_metadata(latest_path)
    update_metadata = read_torch_checkpoint_metadata(update_path)
    assert latest_metadata.total_updates == 2
    assert update_metadata.total_updates == 2
    assert noncanonical_path.exists()
    assert zero_update_path.exists()


def test_torch_checkpoint_save_reports_corrupt_update_manifest(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    latest_path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(latest_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=1,
        total_samples=0,
        total_updates=1,
        retained_update_count=5,
    )
    latest_before = latest_path.read_text(encoding="utf-8")
    state_paths_before = _state_paths(tmp_path)
    (tmp_path / "update-1.json").write_text(
        "{not checkpoint json", encoding="utf-8"
    )

    result = _save_torch_checkpoint(
        manifest_paths=(latest_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=2,
        total_samples=0,
        total_updates=2,
        retained_update_count=5,
    )

    assert isinstance(result, Rejected)
    assert "checkpoint corruption:" in result.reason
    assert "update-1.json" in result.reason
    assert latest_path.read_text(encoding="utf-8") == latest_before
    assert _state_paths(tmp_path) == state_paths_before
    metadata = read_torch_checkpoint_metadata(latest_path)
    assert metadata.total_updates == 1


def test_torch_checkpoint_read_rejects_bad_model_config(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=11,
        total_samples=0,
        total_updates=5,
        retained_update_count=5,
    )
    manifest = _read_json_object(path)
    manifest["model_config"] = {}
    _write_json_object(path, manifest)

    result = _read_torch_checkpoint_metadata(path)

    assert isinstance(result, Rejected)
    assert "checkpoint corruption:" in result.reason
    assert "model_config.d_model" in result.reason


def test_torch_checkpoint_manifest_excludes_execution_fields(
    tmp_path: Path,
) -> None:
    _, _, path = _saved_checkpoint(tmp_path)
    manifest = _read_json_object(path)
    train_config_json: object = manifest["train_config"]
    assert _is_object_dict(train_config_json)

    assert "device" not in train_config_json
    assert "ppo_profile" not in train_config_json
    assert "timeouts" not in train_config_json
    assert "round_seconds" not in train_config_json
    assert "update_seconds" not in train_config_json


def test_torch_checkpoint_read_rejects_invalid_utf8_manifest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "latest.json"
    path.write_bytes(b"\xff")

    result = _read_torch_checkpoint_metadata(path)

    assert isinstance(result, Rejected)
    assert "checkpoint corruption:" in result.reason
    assert "manifest is not valid UTF-8" in result.reason


def test_torch_checkpoint_read_rejects_directory_manifest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "latest.json"
    path.mkdir()

    result = _read_torch_checkpoint_metadata(path)

    assert isinstance(result, Rejected)
    assert "checkpoint corruption:" in result.reason
    assert "manifest file is not readable" in result.reason


def test_torch_checkpoint_read_rejects_manifest_symlink(
    tmp_path: Path,
) -> None:
    _, _, path = _saved_checkpoint(tmp_path)
    external_manifest_path = tmp_path / "external-latest.json"
    path.rename(external_manifest_path)
    path.symlink_to(external_manifest_path)

    result = _read_torch_checkpoint_metadata(path)

    assert isinstance(result, Rejected)
    assert "checkpoint corruption:" in result.reason
    assert "manifest file is a symlink" in result.reason


def test_torch_checkpoint_read_rejects_negative_total_rounds(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=11,
        total_samples=0,
        total_updates=5,
        retained_update_count=5,
    )
    manifest = _read_json_object(path)
    manifest["total_rounds"] = -7
    _write_json_object(path, manifest)

    result = _read_torch_checkpoint_metadata(path)

    assert isinstance(result, Rejected)
    assert "checkpoint corruption:" in result.reason
    assert "manifest total_rounds is negative" in result.reason


def test_torch_checkpoint_load_rejects_negative_total_updates(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=11,
        total_samples=0,
        total_updates=5,
        retained_update_count=5,
    )
    manifest = _read_json_object(path)
    manifest["total_updates"] = -3
    _write_json_object(path, manifest)

    result = _load_torch_checkpoint(
        path=path,
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )

    assert isinstance(result, Rejected)
    assert "checkpoint corruption:" in result.reason
    assert "manifest total_updates is negative" in result.reason


def test_torch_checkpoint_load_rejects_state_hash_mismatch(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=11,
        total_samples=0,
        total_updates=5,
        retained_update_count=5,
    )
    with _single_state_path(path).open("ab") as file:
        file.write(b"corrupt")

    result = _load_torch_checkpoint(
        path=path,
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )

    assert isinstance(result, Rejected)
    assert "checkpoint corruption:" in result.reason
    assert "state sha256 does not match manifest" in result.reason


def test_torch_checkpoint_load_rejects_checkpoint_dir_symlink(
    tmp_path: Path,
) -> None:
    real_dir = tmp_path / "real-checkpoints"
    model_config, train_config, _ = _saved_checkpoint(real_dir)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.symlink_to(real_dir, target_is_directory=True)

    result = _load_torch_checkpoint(
        path=checkpoint_dir / "latest.json",
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )

    assert isinstance(result, Rejected)
    assert "checkpoint corruption:" in result.reason
    assert "checkpoint path is a symlink" in result.reason


def test_torch_checkpoint_load_rejects_objects_dir_symlink(
    tmp_path: Path,
) -> None:
    model_config, train_config, path = _saved_checkpoint(tmp_path)
    objects_dir = tmp_path / "objects"
    external_objects_dir = tmp_path / "external-objects"
    objects_dir.rename(external_objects_dir)
    objects_dir.symlink_to(
        external_objects_dir, target_is_directory=True
    )

    result = _load_torch_checkpoint(
        path=path,
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )

    assert isinstance(result, Rejected)
    assert "checkpoint corruption:" in result.reason
    assert "checkpoint objects path is a symlink" in result.reason


def test_torch_checkpoint_load_rejects_checkpoint_object_symlink(
    tmp_path: Path,
) -> None:
    model_config, train_config, path = _saved_checkpoint(tmp_path)
    object_dir = _single_state_path(path).parent
    external_object_dir = tmp_path / "external-object"
    object_dir.rename(external_object_dir)
    object_dir.symlink_to(external_object_dir, target_is_directory=True)

    result = _load_torch_checkpoint(
        path=path,
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )

    assert isinstance(result, Rejected)
    assert "checkpoint corruption:" in result.reason
    assert "checkpoint object is a symlink" in result.reason


def test_torch_checkpoint_load_rejects_state_file_symlink(
    tmp_path: Path,
) -> None:
    model_config, train_config, path = _saved_checkpoint(tmp_path)
    state_path = _single_state_path(path)
    external_state_path = tmp_path / "external-state.pt"
    state_path.rename(external_state_path)
    state_path.symlink_to(external_state_path)

    result = _load_torch_checkpoint(
        path=path,
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )

    assert isinstance(result, Rejected)
    assert "checkpoint corruption:" in result.reason
    assert "state file is a symlink" in result.reason


def test_torch_checkpoint_load_rejects_directory_state_payload(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=11,
        total_samples=0,
        total_updates=5,
        retained_update_count=5,
    )
    state_path = _single_state_path(path)
    state_path.unlink()
    state_path.mkdir()

    result = _load_torch_checkpoint(
        path=path,
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )

    assert isinstance(result, Rejected)
    assert "checkpoint corruption:" in result.reason
    assert "state file is not readable" in result.reason


def test_torch_checkpoint_load_rejects_non_torch_state_payload(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=11,
        total_samples=0,
        total_updates=5,
        retained_update_count=5,
    )
    state_path = _single_state_path(path)
    state_path.write_bytes(b"not a torch checkpoint")
    _update_manifest_state_sha(path, state_path)

    result = _load_torch_checkpoint(
        path=path,
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )

    assert isinstance(result, Rejected)
    assert "checkpoint corruption:" in result.reason
    assert "state payload cannot be loaded" in result.reason


def test_torch_checkpoint_load_rejects_missing_payload_field(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=11,
        total_samples=0,
        total_updates=5,
        retained_update_count=5,
    )
    state_path = _single_state_path(path)
    torch.save({}, state_path)
    _update_manifest_state_sha(path, state_path)

    result = _load_torch_checkpoint(
        path=path,
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )

    assert isinstance(result, Rejected)
    assert "checkpoint corruption:" in result.reason
    assert "state payload missing schema_version" in result.reason


def test_torch_checkpoint_save_payload_excludes_rng_state(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=11,
        total_samples=0,
        total_updates=5,
        retained_update_count=5,
    )
    payload = _load_state_payload(_single_state_path(path))

    assert "rng_state" not in payload


def test_torch_checkpoint_load_rejects_optimizer_dtype_mismatch(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=11,
        total_samples=0,
        total_updates=5,
        retained_update_count=5,
    )
    state_path = _single_state_path(path)
    payload = _load_state_payload(state_path)
    optimizer_payload = payload["optimizer_state"]
    assert _is_object_dict(optimizer_payload)
    exp_avgs = optimizer_payload["exp_avgs"]
    assert _is_object_list(exp_avgs)
    first_parameter = next(iter(state.model.parameters()))
    updated_exp_avgs = list(exp_avgs)
    updated_exp_avgs[0] = torch.ones(
        first_parameter.shape, dtype=torch.int64
    )
    updated_optimizer_payload: dict[object, object] = dict(
        optimizer_payload
    )
    updated_optimizer_payload["exp_avgs"] = updated_exp_avgs
    payload["optimizer_state"] = updated_optimizer_payload
    _write_state_payload(path, state_path, payload)

    result = _load_torch_checkpoint(
        path=path,
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )

    assert isinstance(result, Rejected)
    assert "checkpoint corruption:" in result.reason
    assert "optimizer_state.exp_avgs tensor dtype" in result.reason


def test_torch_checkpoint_load_does_not_restore_global_rng_state(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"

    save_torch_checkpoint(
        manifest_paths=(path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=7,
        total_samples=0,
        total_updates=3,
        retained_update_count=5,
    )
    python_before = random.getstate()
    torch_before = torch.get_rng_state().clone()

    loaded = load_torch_checkpoint(
        path=path,
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )

    assert loaded.total_rounds == 7
    assert loaded.total_updates == 3
    assert random.getstate() == python_before
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_create_training_state_seeds_initial_model() -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )

    first = create_training_state(
        model_config=model_config,
        train_config=TrainConfig(seed=11),
        device=torch.device("cpu"),
    )
    second = create_training_state(
        model_config=model_config,
        train_config=TrainConfig(seed=11),
        device=torch.device("cpu"),
    )
    third = create_training_state(
        model_config=model_config,
        train_config=TrainConfig(seed=12),
        device=torch.device("cpu"),
    )

    first_parameters = _model_parameters(first.model)
    second_parameters = _model_parameters(second.model)
    third_parameters = _model_parameters(third.model)
    assert _all_tensors_equal(first_parameters, second_parameters)
    assert not _all_tensors_equal(first_parameters, third_parameters)


def test_torch_checkpoint_load_rejects_seed_mismatch(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig(seed=7)
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=7,
        total_samples=0,
        total_updates=3,
        retained_update_count=5,
    )

    result = _load_torch_checkpoint(
        path=path,
        model_config=model_config,
        train_config=TrainConfig(seed=8),
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )

    assert isinstance(result, Rejected)
    assert "checkpoint mismatch:" in result.reason
    assert "train_config.seed" in result.reason


def test_torch_checkpoint_load_rejects_model_config_mismatch(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig(seed=7)
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=7,
        total_samples=0,
        total_updates=3,
        retained_update_count=5,
    )

    result = _load_torch_checkpoint(
        path=path,
        model_config=ModelConfig(
            d_model=16,
            layers=1,
            heads=2,
            max_tokens=192,
        ),
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )

    assert isinstance(result, Rejected)
    assert "checkpoint mismatch:" in result.reason
    assert "model_config" in result.reason


def test_torch_checkpoint_cuda_resume_loads_payload_on_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=7,
        total_samples=0,
        total_updates=3,
        retained_update_count=5,
    )
    state_path = _single_state_path(path)
    saved_payload: object = torch.load(
        state_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    assert _is_object_dict(saved_payload)
    payload: dict[object, object] = dict(saved_payload)
    load_map_locations: list[torch.device] = []
    model_devices: list[torch.device] = []

    def fake_torch_load(
        file_path: object,
        *,
        map_location: object,
        weights_only: object,
    ) -> object:
        assert file_path == state_path
        assert isinstance(map_location, torch.device)
        assert weights_only is True
        load_map_locations.append(map_location)
        return payload

    def fake_create_model(
        config: ModelConfig,
        device: torch.device,
    ) -> TractorPolicyModel:
        model_devices.append(device)
        return TractorPolicyModel(
            d_model=config.d_model,
            layers=config.layers,
            heads=config.heads,
        )

    def fake_cuda_available() -> bool:
        return True

    monkeypatch.setattr(torch, "load", fake_torch_load)
    monkeypatch.setattr(
        training_state,
        "create_model",
        fake_create_model,
    )
    monkeypatch.setattr(torch.cuda, "is_available", fake_cuda_available)

    loaded = load_torch_checkpoint(
        path=path,
        model_config=model_config,
        train_config=TrainConfig(),
        execution_config=ExecutionConfig(
            model_ranks=ModelRankPlacement(
                kind="cuda", devices=("cuda:0",)
            )
        ),
        device=torch.device("cuda"),
    )

    assert loaded.total_rounds == 7
    assert loaded.total_updates == 3
    assert load_map_locations == [torch.device("cpu")]
    assert model_devices == [torch.device("cuda")]


def test_resolve_train_config_defaults_and_resume_overrides(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig(
        learning_rate=0.0007,
        ppo_epochs=7,
        minibatch_size=11,
    )
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=7,
        total_samples=0,
        total_updates=3,
        retained_update_count=5,
    )

    fresh = resolve_train_config(
        cli_overrides=TrainConfigOverrides(),
        resume_path=None,
    )
    resumed = resolve_train_config(
        cli_overrides=TrainConfigOverrides(),
        resume_path=path,
    )
    resumed_with_same_seed = resolve_train_config(
        cli_overrides=TrainConfigOverrides(seed=train_config.seed),
        resume_path=path,
    )
    execution = resolve_execution_config(
        ExecutionConfigOverrides(
            model_ranks=ModelRankPlacement(
                kind="cuda", devices=("cuda:0",)
            ),
            ppo_profile="detailed",
            round_timeout_seconds=111.0,
            rollout_response_timeout_seconds=222.0,
            state_sync_timeout_seconds=333.0,
            update_timeout_seconds=444.0,
        )
    )
    assert isinstance(fresh, Ok)
    assert isinstance(resumed, Ok)
    assert isinstance(resumed_with_same_seed, Ok)
    assert isinstance(execution, Ok)
    assert fresh.value == TrainConfig()
    assert resumed.value == train_config
    assert resumed_with_same_seed.value.seed == train_config.seed
    assert execution.value.model_ranks.kind == "cuda"
    assert execution.value.ppo_profile == "detailed"
    assert execution.value.timeouts == ExecutionTimeouts(
        round_seconds=111.0,
        rollout_response_seconds=222.0,
        state_sync_seconds=333.0,
        update_seconds=444.0,
    )


def save_torch_checkpoint(
    *,
    manifest_paths: tuple[Path, ...],
    model: TractorPolicyModel,
    trainer: PPOTrainer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    total_rounds: int,
    total_samples: int = 0,
    total_updates: int,
    retained_update_count: int,
) -> None:
    result = _save_torch_checkpoint(
        manifest_paths=manifest_paths,
        model=model,
        trainer=trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=total_rounds,
        total_samples=total_samples,
        total_updates=total_updates,
        retained_update_count=retained_update_count,
    )
    assert isinstance(result, Ok)
    assert result.value.post_commit_prune_failure is None


def load_torch_checkpoint(
    *,
    path: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig | None = None,
    device: torch.device,
) -> LoadedTrainingState:
    resolved_execution_config = _execution_config_for_device(
        device=device,
        execution_config=execution_config,
    )
    result = _load_torch_checkpoint(
        path=path,
        model_config=model_config,
        train_config=train_config,
        execution_config=resolved_execution_config,
        device=device,
    )
    assert isinstance(result, Ok)
    return result.value


def read_torch_checkpoint_metadata(
    path: Path,
) -> TorchCheckpointMetadata:
    result = _read_torch_checkpoint_metadata(path)
    assert isinstance(result, Ok)
    return result.value


def _model_rank_kind_from_torch(
    device: torch.device,
) -> ModelRankKind:
    if device.type == "cpu":
        return "none"
    if device.type == "cuda":
        return "cuda"
    if device.type == "mps":
        return "mps"
    assert False


def _execution_config_for_device(
    *,
    device: torch.device,
    execution_config: ExecutionConfig | None,
) -> ExecutionConfig:
    expected_kind = _model_rank_kind_from_torch(device)
    if execution_config is None:
        return _execution_config_for_model_rank_kind(expected_kind)
    assert execution_config.model_ranks.kind == expected_kind
    return execution_config


def _execution_config_for_model_rank_kind(
    model_rank_kind: ModelRankKind,
) -> ExecutionConfig:
    if model_rank_kind == "none":
        return ExecutionConfig()
    if model_rank_kind == "cuda":
        return ExecutionConfig(
            model_ranks=ModelRankPlacement(
                kind="cuda", devices=("cuda:0",)
            ),
        )
    return ExecutionConfig(
        model_ranks=ModelRankPlacement(kind="mps", devices=("mps",)),
    )


def _saved_checkpoint(
    checkpoint_dir: Path,
) -> tuple[ModelConfig, TrainConfig, Path]:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=192,
    )
    train_config = TrainConfig()
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    path = checkpoint_dir / "latest.json"
    save_torch_checkpoint(
        manifest_paths=(path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=11,
        total_samples=0,
        total_updates=5,
        retained_update_count=5,
    )
    return model_config, train_config, path


def _single_state_path(checkpoint_path: Path) -> Path:
    state_paths = _state_paths(checkpoint_path.parent)
    assert len(state_paths) == 1
    return state_paths[0]


def _read_json_object(path: Path) -> dict[object, object]:
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    assert _is_object_dict(loaded)
    return dict(loaded)


def _write_json_object(path: Path, data: dict[object, object]) -> None:
    path.write_text(
        f"{json.dumps(data, ensure_ascii=False, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _update_manifest_state_sha(
    manifest_path: Path,
    state_path: Path,
) -> None:
    manifest = _read_json_object(manifest_path)
    manifest["state_sha256"] = hashlib.sha256(
        state_path.read_bytes()
    ).hexdigest()
    _write_json_object(manifest_path, manifest)


def _load_state_payload(path: Path) -> dict[object, object]:
    loaded: object = torch.load(
        path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    assert _is_object_dict(loaded)
    return dict(loaded)


def _write_state_payload(
    manifest_path: Path,
    state_path: Path,
    payload: dict[object, object],
) -> None:
    torch.save(payload, state_path)
    _update_manifest_state_sha(manifest_path, state_path)


def _state_paths(checkpoint_dir: Path) -> tuple[Path, ...]:
    return tuple(
        sorted((checkpoint_dir / "objects").glob("*/state.pt"))
    )


def _object_entries(checkpoint_dir: Path) -> tuple[Path, ...]:
    objects_dir = checkpoint_dir / "objects"
    if not objects_dir.exists():
        return ()
    return tuple(sorted(objects_dir.iterdir()))


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _model_parameters(
    model: TractorPolicyModel,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        parameter.detach().cpu().clone()
        for parameter in model.parameters()
    )


def _all_tensors_equal(
    left: tuple[torch.Tensor, ...],
    right: tuple[torch.Tensor, ...],
) -> bool:
    assert len(left) == len(right)
    return all(
        torch.equal(left_tensor, right_tensor)
        for left_tensor, right_tensor in zip(left, right, strict=True)
    )
