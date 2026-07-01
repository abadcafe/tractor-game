"""Top-level self-play training loop."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from server.training.config import ModelConfig, TrainConfig
from server.training.metrics import TrainingMetric, append_metric
from server.training.runner import SelfPlaySession
from server.training.torch_checkpoints import (
    LoadedTrainingState,
    create_training_state,
    load_torch_checkpoint,
    save_torch_checkpoint,
)
from server.training.torch_policy import TorchTrainingPolicy


@dataclass(frozen=True, slots=True)
class TrainingLoopResult:
    """Final counters and checkpoint path from a training run."""

    total_rounds: int
    total_updates: int
    checkpoint_path: Path


async def train_self_play(
    *,
    run_dir: Path,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    max_rounds: int,
    resume: Path | None,
) -> TrainingLoopResult:
    """Train by self-play for a fixed number of rounds."""
    assert max_rounds >= 0
    device = torch.device(train_config.device)
    state = _load_or_create_state(
        resume=resume,
        model_config=model_config,
        train_config=train_config,
        device=device,
    )
    policy = TorchTrainingPolicy(
        model=state.model,
        config=model_config,
        device=device,
    )
    start = time.monotonic()
    total_rounds = state.total_rounds
    total_updates = state.total_updates
    latest_checkpoint = run_dir / "checkpoints" / "latest.pt"
    session = SelfPlaySession(policy=policy)

    for _ in range(max_rounds):
        round_result = await session.play_round(
            max_seconds=train_config.max_round_seconds
        )
        if round_result.rewarded_steps:
            stats = state.trainer.update(round_result.rewarded_steps)
            total_updates += 1
        else:
            stats = None
        total_rounds += 1
        elapsed = max(time.monotonic() - start, 0.000001)
        decision_count = len(round_result.rewarded_steps)
        checkpoint_path = _checkpoint_path(
            run_dir=run_dir,
            total_updates=total_updates,
            latest_checkpoint=latest_checkpoint,
            train_config=train_config,
        )
        save_torch_checkpoint(
            path=checkpoint_path,
            model=state.model,
            trainer=state.trainer,
            model_config=model_config,
            train_config=train_config,
            total_rounds=total_rounds,
            total_updates=total_updates,
        )
        if checkpoint_path != latest_checkpoint:
            save_torch_checkpoint(
                path=latest_checkpoint,
                model=state.model,
                trainer=state.trainer,
                model_config=model_config,
                train_config=train_config,
                total_rounds=total_rounds,
                total_updates=total_updates,
            )
        append_metric(
            run_dir,
            TrainingMetric(
                run_id=run_id,
                total_games=total_rounds,
                total_updates=total_updates,
                games_per_second=total_rounds / elapsed,
                decisions_per_second=decision_count
                / round_result.elapsed_seconds,
                average_reward=round_result.team0_reward,
                average_level_delta=round_result.team0_reward,
                policy_loss=None
                if stats is None
                else stats.policy_loss,
                value_loss=None if stats is None else stats.value_loss,
                entropy=None if stats is None else stats.entropy,
                invalid_action_count=round_result.invalid_action_count,
                resample_count=round_result.resample_count,
                forced_action_count=round_result.forced_action_count,
                legal_action_rate=_legal_action_rate(
                    accepted=round_result.accepted_action_count,
                    invalid=round_result.invalid_action_count,
                ),
                average_action_tokens=round_result.average_action_tokens,
                checkpoint_path=str(checkpoint_path),
            ),
        )
        if round_result.game_over:
            session = SelfPlaySession(policy=policy)
    if max_rounds == 0:
        save_torch_checkpoint(
            path=latest_checkpoint,
            model=state.model,
            trainer=state.trainer,
            model_config=model_config,
            train_config=train_config,
            total_rounds=total_rounds,
            total_updates=total_updates,
        )
    return TrainingLoopResult(
        total_rounds=total_rounds,
        total_updates=total_updates,
        checkpoint_path=latest_checkpoint,
    )


def _legal_action_rate(*, accepted: int, invalid: int) -> float:
    total = accepted + invalid
    if total == 0:
        return 1.0
    return accepted / total


def _load_or_create_state(
    *,
    resume: Path | None,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> LoadedTrainingState:
    if resume is None:
        return create_training_state(
            model_config=model_config,
            train_config=train_config,
            device=device,
        )
    return load_torch_checkpoint(
        path=resume,
        model_config=model_config,
        train_config=train_config,
        device=device,
    )


def _checkpoint_path(
    *,
    run_dir: Path,
    total_updates: int,
    latest_checkpoint: Path,
    train_config: TrainConfig,
) -> Path:
    if (
        total_updates > 0
        and total_updates % train_config.checkpoint_every_updates == 0
    ):
        return run_dir / "checkpoints" / f"update-{total_updates}.pt"
    return latest_checkpoint


def run_training_loop(
    *,
    run_dir: Path,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    max_rounds: int,
    resume: Path | None,
) -> TrainingLoopResult:
    """Synchronous wrapper for CLI entry points."""
    return asyncio.run(
        train_self_play(
            run_dir=run_dir,
            run_id=run_id,
            model_config=model_config,
            train_config=train_config,
            max_rounds=max_rounds,
            resume=resume,
        )
    )
