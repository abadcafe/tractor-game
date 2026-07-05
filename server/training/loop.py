"""Top-level self-play training loop."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from server import result as _result
from server.training.config import ModelConfig, TrainConfig
from server.training.metrics import (
    TrainingMetric,
    append_metric,
    validate_training_metric,
)
from server.training.runner import SelfPlaySession
from server.training.torch_checkpoints import (
    load_torch_checkpoint,
    save_torch_checkpoint,
)
from server.training.torch_policy import TorchTrainingPolicy
from server.training.training_state import (
    LoadedTrainingState,
    create_training_state,
    resolve_training_device,
)


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
) -> _result.Ok[TrainingLoopResult] | _result.Rejected:
    """Train by self-play for a fixed number of rounds."""
    assert max_rounds >= 0
    device_result = resolve_training_device(train_config.device)
    if isinstance(device_result, _result.Rejected):
        return device_result
    device = device_result.value
    state_result = _load_or_create_state(
        resume=resume,
        model_config=model_config,
        train_config=train_config,
        device=device,
    )
    if isinstance(state_result, _result.Rejected):
        return state_result
    state = state_result.value
    policy = TorchTrainingPolicy(
        model=state.model,
        config=model_config,
        device=device,
    )
    start = _monotonic()
    total_rounds = state.total_rounds
    total_updates = state.total_updates
    start_total_rounds = total_rounds
    checkpoint_dir = run_dir / "checkpoints"
    latest_checkpoint = checkpoint_dir / "latest.json"
    session = SelfPlaySession(policy=policy)

    for _ in range(max_rounds):
        round_result = await session.play_round(
            max_seconds=train_config.max_round_seconds
        )
        if isinstance(round_result, _result.Rejected):
            return round_result
        round_data = round_result.value
        if not round_data.rollout.is_empty():
            update_result = state.trainer.update(round_data.rollout)
            if isinstance(update_result, _result.Rejected):
                return update_result
            stats = update_result.value
            total_updates += 1
        else:
            stats = None
        total_rounds += 1
        elapsed = max(_monotonic() - start, 0.000001)
        process_rounds = total_rounds - start_total_rounds
        assert process_rounds > 0
        decision_count = round_data.rollout.transition_count()
        archive_checkpoint = (
            None
            if stats is None
            else _archive_checkpoint_path(
                run_dir=run_dir,
                total_updates=total_updates,
                train_config=train_config,
            )
        )
        checkpoint_path = (
            latest_checkpoint
            if archive_checkpoint is None
            else archive_checkpoint
        )
        metric = TrainingMetric(
            run_id=run_id,
            total_games=total_rounds,
            total_updates=total_updates,
            process_games_per_second=process_rounds / elapsed,
            last_round_decisions_per_second=(
                decision_count / round_data.elapsed_seconds
            ),
            last_team0_reward=round_data.team0_reward,
            last_team1_reward=round_data.team1_reward,
            last_generated_action_count=(
                round_data.generated_action_count
            ),
            last_accepted_action_count=(
                round_data.accepted_action_count
            ),
            last_decision_count=decision_count,
            last_average_action_choices=(
                round_data.average_action_choices
            ),
            policy_loss=None if stats is None else stats.policy_loss,
            value_loss=None if stats is None else stats.value_loss,
            entropy=None if stats is None else stats.entropy,
            approx_kl=None if stats is None else stats.approx_kl,
            clip_fraction=None
            if stats is None
            else stats.clip_fraction,
            ppo_update_seconds=None
            if stats is None
            else stats.profile.update_seconds,
            ppo_minibatch_loss_seconds=None
            if stats is None
            else stats.profile.minibatch_loss_seconds,
            ppo_observation_batch_seconds=None
            if stats is None
            else stats.profile.observation_batch_seconds,
            ppo_observation_encode_seconds=None
            if stats is None
            else stats.profile.observation_encode_seconds,
            ppo_value_head_seconds=None
            if stats is None
            else stats.profile.value_head_seconds,
            ppo_argument_select_seconds=None
            if stats is None
            else stats.profile.argument_select_seconds,
            ppo_argument_prefix_tensorize_seconds=None
            if stats is None
            else stats.profile.argument_prefix_tensorize_seconds,
            ppo_argument_decode_seconds=None
            if stats is None
            else stats.profile.argument_decode_seconds,
            ppo_argument_distribution_seconds=None
            if stats is None
            else stats.profile.argument_distribution_seconds,
            ppo_backward_seconds=None
            if stats is None
            else stats.profile.backward_seconds,
            ppo_optimizer_step_seconds=None
            if stats is None
            else stats.profile.optimizer_step_seconds,
            ppo_argument_decode_fraction=None
            if stats is None
            else stats.profile.argument_decode_fraction,
            ppo_argument_prefix_batch_count=None
            if stats is None
            else stats.profile.argument_prefix_batch_count,
            ppo_argument_prefix_row_count=None
            if stats is None
            else stats.profile.argument_prefix_row_count,
            ppo_argument_prefix_token_count=None
            if stats is None
            else stats.profile.argument_prefix_token_count,
            ppo_argument_prefix_valid_token_count=None
            if stats is None
            else stats.profile.argument_prefix_valid_token_count,
            ppo_argument_prefix_padding_token_count=None
            if stats is None
            else stats.profile.argument_prefix_padding_token_count,
            checkpoint_path=str(checkpoint_path),
        )
        metric_validation = validate_training_metric(metric)
        if isinstance(metric_validation, _result.Rejected):
            return metric_validation
        save_result = save_torch_checkpoint(
            manifest_paths=(
                (latest_checkpoint,)
                if archive_checkpoint is None
                else (archive_checkpoint, latest_checkpoint)
            ),
            model=state.model,
            trainer=state.trainer,
            model_config=model_config,
            train_config=train_config,
            total_rounds=total_rounds,
            total_updates=total_updates,
            retained_update_count=(
                train_config.checkpoint_retention_updates
            ),
        )
        if isinstance(save_result, _result.Rejected):
            return save_result
        metric_result = append_metric(run_dir, metric)
        prune_failure = save_result.value.post_commit_prune_failure
        if isinstance(metric_result, _result.Rejected):
            if prune_failure is not None:
                return _result.Rejected(
                    reason=(
                        f"{metric_result.reason}; "
                        f"{prune_failure.reason}"
                    )
                )
            return metric_result
        if prune_failure is not None:
            return prune_failure
        if round_data.game_over:
            session = SelfPlaySession(policy=policy)
    if max_rounds == 0:
        save_result = save_torch_checkpoint(
            manifest_paths=(latest_checkpoint,),
            model=state.model,
            trainer=state.trainer,
            model_config=model_config,
            train_config=train_config,
            total_rounds=total_rounds,
            total_updates=total_updates,
            retained_update_count=(
                train_config.checkpoint_retention_updates
            ),
        )
        if isinstance(save_result, _result.Rejected):
            return save_result
        prune_failure = save_result.value.post_commit_prune_failure
        if prune_failure is not None:
            return prune_failure
    return _result.Ok(
        value=TrainingLoopResult(
            total_rounds=total_rounds,
            total_updates=total_updates,
            checkpoint_path=latest_checkpoint,
        )
    )


def _load_or_create_state(
    *,
    resume: Path | None,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> _result.Ok[LoadedTrainingState] | _result.Rejected:
    if resume is None:
        return _result.Ok(
            value=create_training_state(
                model_config=model_config,
                train_config=train_config,
                device=device,
            )
        )
    return load_torch_checkpoint(
        path=resume,
        model_config=model_config,
        train_config=train_config,
        device=device,
    )


def _archive_checkpoint_path(
    *,
    run_dir: Path,
    total_updates: int,
    train_config: TrainConfig,
) -> Path | None:
    if train_config.checkpoint_retention_updates == 0:
        return None
    if (
        total_updates > 0
        and total_updates % train_config.checkpoint_every_updates == 0
    ):
        return run_dir / "checkpoints" / f"update-{total_updates}.json"
    return None


def _monotonic() -> float:
    return time.monotonic()


def run_training_loop(
    *,
    run_dir: Path,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    max_rounds: int,
    resume: Path | None,
) -> _result.Ok[TrainingLoopResult] | _result.Rejected:
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
