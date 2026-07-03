"""Tests for torch training checkpoint metadata."""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path
from typing import TypeGuard

import pytest
import torch

from server.rules.cards import Rank
from server.sm.required_progress import RequiredLevelPlan
from server.training import torch_checkpoints
from server.training.config import ModelConfig, TrainConfig
from server.training.model import TractorPolicyModel
from server.training.torch_checkpoints import (
    create_training_state,
    load_torch_checkpoint,
    read_torch_checkpoint_metadata,
    save_torch_checkpoint,
)
from server.training.train import (
    TrainConfigOverrides,
    resolve_model_config,
    resolve_train_config,
)


def test_torch_checkpoint_metadata_drives_resume_model_config(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        dropout=0.0,
        max_tokens=192,
    )
    train_config = TrainConfig(
        device="cpu",
        learning_rate=0.0003,
        max_round_seconds=30.0,
        required_level_plan=RequiredLevelPlan(
            required_levels=(Rank.JACK, Rank.ACE)
        ),
    )
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"

    save_torch_checkpoint(
        path=path,
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=7,
        total_updates=3,
    )

    metadata = read_torch_checkpoint_metadata(path)
    assert metadata.model_config == model_config
    assert metadata.train_config == train_config
    assert metadata.total_rounds == 7
    assert metadata.total_updates == 3
    assert (
        resolve_model_config(
            cli_model_config=ModelConfig(d_model=128),
            resume_path=path,
        )
        == model_config
    )


def test_read_metadata_uses_manifest_without_torch_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        dropout=0.0,
        max_tokens=192,
    )
    train_config = TrainConfig(device="cpu")
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"
    save_torch_checkpoint(
        path=path,
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=11,
        total_updates=5,
    )

    def fail_torch_load(*args: object, **kwargs: object) -> object:
        assert False

    monkeypatch.setattr(torch, "load", fail_torch_load)

    metadata = read_torch_checkpoint_metadata(path)

    assert metadata.model_config == model_config
    assert metadata.train_config == train_config
    assert metadata.total_rounds == 11
    assert metadata.total_updates == 5


def test_torch_checkpoint_state_payload_is_weights_only_safe(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        dropout=0.0,
        max_tokens=192,
    )
    train_config = TrainConfig(device="cpu")
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"
    save_torch_checkpoint(
        path=path,
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=11,
        total_updates=5,
    )

    loaded: object = torch.load(
        _single_state_path(path),
        map_location=torch.device("cpu"),
        weights_only=True,
    )

    assert isinstance(loaded, dict)
    assert loaded["schema_version"] == 12
    assert isinstance(loaded["checkpoint_id"], str)
    assert "model_config" not in loaded
    assert "train_config" not in loaded
    assert "total_rounds" not in loaded
    assert "total_updates" not in loaded


def test_torch_checkpoint_load_rejects_state_hash_mismatch(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        dropout=0.0,
        max_tokens=192,
    )
    train_config = TrainConfig(device="cpu")
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"
    save_torch_checkpoint(
        path=path,
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=11,
        total_updates=5,
    )
    with _single_state_path(path).open("ab") as file:
        file.write(b"corrupt")

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path\n"
                "import torch\n"
                "from server.training.config import ModelConfig, "
                "TrainConfig\n"
                "from server.training.torch_checkpoints import "
                "load_torch_checkpoint\n"
                "load_torch_checkpoint(\n"
                f"    path=Path({str(path)!r}),\n"
                "    model_config=ModelConfig(\n"
                "        d_model=8,\n"
                "        layers=1,\n"
                "        heads=2,\n"
                "        dropout=0.0,\n"
                "        max_tokens=192,\n"
                "    ),\n"
                "    train_config=TrainConfig(device='cpu'),\n"
                "    device=torch.device('cpu'),\n"
                ")\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "AssertionError" in completed.stderr


def test_torch_checkpoint_load_restores_rng_state(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        dropout=0.0,
        max_tokens=192,
    )
    train_config = TrainConfig(device="cpu")
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"

    save_torch_checkpoint(
        path=path,
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=7,
        total_updates=3,
    )
    expected_python = random.random()
    expected_torch = torch.rand(3)
    for _ in range(17):
        random.random()
    torch.rand(17)

    loaded = load_torch_checkpoint(
        path=path,
        model_config=model_config,
        train_config=train_config,
        device=torch.device("cpu"),
    )

    assert loaded.total_rounds == 7
    assert loaded.total_updates == 3
    assert random.random() == expected_python
    assert torch.equal(torch.rand(3), expected_torch)


def test_torch_checkpoint_cuda_resume_loads_payload_on_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        dropout=0.0,
        max_tokens=192,
    )
    train_config = TrainConfig(device="cpu")
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"
    save_torch_checkpoint(
        path=path,
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=7,
        total_updates=3,
    )
    state_path = _single_state_path(path)
    saved_payload: object = torch.load(
        state_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    assert _is_object_dict(saved_payload)
    payload: dict[object, object] = dict(saved_payload)
    rng_payload = payload["rng_state"]
    assert _is_object_dict(rng_payload)
    updated_rng_payload: dict[object, object] = dict(rng_payload)
    fake_cuda_state = torch.get_rng_state()
    updated_rng_payload["torch_cuda_states"] = [fake_cuda_state]
    payload["rng_state"] = updated_rng_payload
    load_map_locations: list[torch.device] = []
    model_devices: list[torch.device] = []
    restored_cuda_states: list[tuple[torch.Tensor, ...]] = []

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
            dropout=config.dropout,
        )

    def fake_cuda_available() -> bool:
        return True

    def fake_set_rng_state_all(states: list[torch.Tensor]) -> None:
        restored_cuda_states.append(tuple(states))

    monkeypatch.setattr(torch, "load", fake_torch_load)
    monkeypatch.setattr(
        torch_checkpoints,
        "create_model",
        fake_create_model,
    )
    monkeypatch.setattr(torch.cuda, "is_available", fake_cuda_available)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        fake_set_rng_state_all,
    )

    loaded = load_torch_checkpoint(
        path=path,
        model_config=model_config,
        train_config=TrainConfig(device="cuda"),
        device=torch.device("cuda"),
    )

    assert loaded.total_rounds == 7
    assert loaded.total_updates == 3
    assert load_map_locations == [torch.device("cpu")]
    assert model_devices == [torch.device("cuda")]
    assert len(restored_cuda_states) == 1
    assert len(restored_cuda_states[0]) == 1
    assert torch.equal(restored_cuda_states[0][0], fake_cuda_state)


def test_resolve_train_config_defaults_and_resume_overrides(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        dropout=0.0,
        max_tokens=192,
    )
    train_config = TrainConfig(
        device="cpu",
        learning_rate=0.0007,
        max_round_seconds=333.0,
        ppo_epochs=7,
        minibatch_size=11,
    )
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.json"
    save_torch_checkpoint(
        path=path,
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=7,
        total_updates=3,
    )

    fresh = resolve_train_config(
        cli_overrides=TrainConfigOverrides(),
        resume_path=None,
    )
    resumed = resolve_train_config(
        cli_overrides=TrainConfigOverrides(),
        resume_path=path,
    )
    resumed_with_device = resolve_train_config(
        cli_overrides=TrainConfigOverrides(device="cuda"),
        resume_path=path,
    )
    resumed_with_required_levels = resolve_train_config(
        cli_overrides=TrainConfigOverrides(
            required_level_plan=RequiredLevelPlan(
                required_levels=(
                    Rank.FIVE,
                    Rank.TEN,
                    Rank.KING,
                    Rank.ACE,
                )
            )
        ),
        resume_path=path,
    )

    assert fresh.max_round_seconds == 120.0
    assert resumed == train_config
    assert resumed_with_device.device == "cuda"
    assert (
        resumed_with_device.learning_rate == train_config.learning_rate
    )
    assert resumed_with_device.max_round_seconds == (
        train_config.max_round_seconds
    )
    assert resumed_with_required_levels.required_level_plan == (
        RequiredLevelPlan(
            required_levels=(
                Rank.FIVE,
                Rank.TEN,
                Rank.KING,
                Rank.ACE,
            )
        )
    )


def _single_state_path(checkpoint_path: Path) -> Path:
    state_paths = tuple(
        (checkpoint_path.parent / "objects").glob("*/state.pt")
    )
    assert len(state_paths) == 1
    return state_paths[0]


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)
