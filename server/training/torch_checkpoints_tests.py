"""Tests for torch training checkpoint metadata."""

from __future__ import annotations

from pathlib import Path

import torch

from server.training.config import ModelConfig, TrainConfig
from server.training.torch_checkpoints import (
    create_training_state,
    read_torch_checkpoint_metadata,
    save_torch_checkpoint,
)
from server.training.train import resolve_model_config


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
