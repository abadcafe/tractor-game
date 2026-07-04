"""Trainable model and optimizer state construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from server import result as _result
from server.training.config import (
    ModelConfig,
    TrainConfig,
    TrainingDevice,
)
from server.training.model import TractorPolicyModel
from server.training.ppo import PPOTrainer
from server.training.torch_rng import seed_training_rng


@dataclass(frozen=True, slots=True)
class LoadedTrainingState:
    """Loaded model/trainer state and progress counters."""

    model: TractorPolicyModel
    trainer: PPOTrainer
    total_rounds: int
    total_updates: int


def resolve_training_device(
    device: TrainingDevice,
) -> _result.Ok[torch.device] | _result.Rejected:
    """Resolve a configured training device before model creation."""
    if device == "cpu":
        return _result.Ok(value=torch.device("cpu"))
    if not torch.cuda.is_available():
        return _result.Rejected(
            reason=(
                "--device cuda is unavailable in this PyTorch runtime"
            )
        )
    return _result.Ok(value=torch.device("cuda"))


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
