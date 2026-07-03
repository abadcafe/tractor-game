"""Tests for torch training checkpoint metadata."""

from __future__ import annotations

import random
from pathlib import Path

import torch

from server.training.config import ModelConfig, TrainConfig
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
    )
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        device=torch.device("cpu"),
    )
    path = tmp_path / "latest.pt"

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
    path = tmp_path / "latest.pt"

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
    path = tmp_path / "latest.pt"
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

    assert fresh.max_round_seconds == 120.0
    assert resumed == train_config
    assert resumed_with_device.device == "cuda"
    assert (
        resumed_with_device.learning_rate == train_config.learning_rate
    )
    assert resumed_with_device.max_round_seconds == (
        train_config.max_round_seconds
    )
