"""Trainable model and optimizer state construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from server import result as _result
from server.training.config import (
    ModelConfig,
    TrainConfig,
)
from server.training.model import TractorPolicyModel
from server.training.ppo import PPOTrainer
from server.training.ppo.distributed import PPOUpdatePartition
from server.training.runtime.config import ExecutionConfig
from server.training.runtime.seeding import seed_training_rng


@dataclass(frozen=True, slots=True)
class LoadedTrainingState:
    """Loaded model/trainer state and progress counters."""

    model: TractorPolicyModel
    trainer: PPOTrainer
    total_rounds: int
    total_samples: int
    total_updates: int


def validate_model_rank_runtime(
    execution_config: ExecutionConfig,
) -> _result.Ok[None] | _result.Rejected:
    """Validate model-rank device availability before setup."""
    model_ranks = execution_config.model_ranks
    if model_ranks.kind == "none":
        return _result.Ok(value=None)
    if model_ranks.kind == "cuda":
        if not torch.cuda.is_available():
            return _result.Rejected(
                reason=(
                    "--model-ranks cuda is unavailable in this "
                    "PyTorch runtime"
                )
            )
        for device_name in model_ranks.devices:
            device = torch.device(device_name)
            if device.type != "cuda":
                return _result.Rejected(
                    reason=f"invalid CUDA model rank: {device_name}"
                )
            if device.index >= torch.cuda.device_count():
                return _result.Rejected(
                    reason=(
                        f"CUDA model rank is unavailable: {device_name}"
                    )
                )
        return _result.Ok(value=None)
    if not torch.backends.mps.is_available():
        return _result.Rejected(
            reason=(
                "--model-ranks mps is unavailable in this "
                "PyTorch runtime"
            )
        )
    return _result.Ok(value=None)


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
    execution_config: ExecutionConfig,
    device: torch.device | None = None,
    update_partition: PPOUpdatePartition | None = None,
) -> LoadedTrainingState:
    """Create fresh model and PPO trainer."""
    resolved_device = device
    if resolved_device is None:
        resolved_device = torch.device("cpu")
    seed_training_rng(train_config.seed)
    model = create_model(model_config, resolved_device)
    trainer = PPOTrainer(
        model=model,
        train_config=train_config,
        device=resolved_device,
        profile_mode=execution_config.ppo_profile,
        update_partition=update_partition,
    )
    return LoadedTrainingState(
        model=model,
        trainer=trainer,
        total_rounds=0,
        total_samples=0,
        total_updates=0,
    )
