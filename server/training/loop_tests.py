"""Tests for actual self-play training loop."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from server.result import Ok, Rejected
from server.training import loop
from server.training.config import ModelConfig, TrainConfig
from server.training.loop import train_self_play
from server.training.metrics import read_metrics
from server.training.ppo import PPOTrainer, PPOUpdateStats
from server.training.runner import TrainingRoundResult
from server.training.torch_checkpoints import (
    read_torch_checkpoint_metadata,
    save_torch_checkpoint,
)
from server.training.training_state import create_training_state
from server.training.trajectory import RolloutBatch


@pytest.mark.asyncio
async def test_train_self_play_one_round_writes_checkpoint_and_metrics(
    tmp_path: Path,
) -> None:
    result = await train_self_play(
        run_dir=tmp_path,
        run_id="loop-test",
        model_config=ModelConfig(
            d_model=4,
            layers=1,
            heads=1,
            max_tokens=512,
        ),
        train_config=TrainConfig(
            device="cpu",
            learning_rate=0.0003,
            checkpoint_every_updates=1,
            checkpoint_retention_updates=1,
            max_round_seconds=120.0,
            ppo_epochs=1,
            minibatch_size=512,
        ),
        max_rounds=1,
        resume=None,
    )
    assert isinstance(result, Ok)
    loop_result = result.value

    assert loop_result.total_rounds == 1
    assert loop_result.total_updates == 1
    assert loop_result.checkpoint_path.exists()
    assert (tmp_path / "checkpoints" / "update-1.json").exists()
    assert len(_checkpoint_state_paths(tmp_path)) == 1
    metrics = read_metrics(tmp_path)
    assert metrics
    assert metrics[-1].total_games == 1
    assert metrics[-1].total_updates == 1


@pytest.mark.asyncio
async def test_train_self_play_resumes_and_continues_counters(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=4,
        layers=1,
        heads=1,
        max_tokens=512,
    )
    train_config = TrainConfig(
        device="cpu",
        learning_rate=0.0003,
        max_round_seconds=120.0,
        ppo_epochs=1,
        minibatch_size=512,
    )
    first_result = await train_self_play(
        run_dir=tmp_path,
        run_id="loop-resume-test",
        model_config=model_config,
        train_config=train_config,
        max_rounds=1,
        resume=None,
    )
    assert isinstance(first_result, Ok)
    first = first_result.value

    resumed_result = await train_self_play(
        run_dir=tmp_path,
        run_id="loop-resume-test",
        model_config=model_config,
        train_config=train_config,
        max_rounds=1,
        resume=first.checkpoint_path,
    )
    assert isinstance(resumed_result, Ok)
    resumed = resumed_result.value

    assert resumed.total_rounds == 2
    assert resumed.total_updates == 2
    metadata = read_torch_checkpoint_metadata(resumed.checkpoint_path)
    assert isinstance(metadata, Ok)
    assert metadata.value.total_rounds == 2
    assert metadata.value.total_updates == 2
    metrics = read_metrics(tmp_path)
    assert [metric.total_games for metric in metrics] == [1, 2]
    assert [metric.total_updates for metric in metrics] == [1, 2]


@pytest.mark.asyncio
async def test_train_self_play_resume_speed_uses_process_rounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = ModelConfig(
        d_model=4,
        layers=1,
        heads=1,
        max_tokens=512,
    )
    train_config = TrainConfig(
        device="cpu",
        learning_rate=0.0003,
        max_round_seconds=120.0,
        ppo_epochs=1,
        minibatch_size=512,
    )
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        device=torch.device("cpu"),
    )
    latest_path = tmp_path / "checkpoints" / "latest.json"
    save_result = save_torch_checkpoint(
        manifest_paths=(latest_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=100,
        total_updates=7,
        retained_update_count=train_config.checkpoint_retention_updates,
    )
    assert isinstance(save_result, Ok)
    times = iter((0.0, 10.0))
    monkeypatch.setattr(loop, "_monotonic", lambda: next(times))
    monkeypatch.setattr(loop, "SelfPlaySession", _NoRewardSession)

    result = await train_self_play(
        run_dir=tmp_path,
        run_id="loop-resume-speed-test",
        model_config=model_config,
        train_config=train_config,
        max_rounds=1,
        resume=latest_path,
    )
    assert isinstance(result, Ok)
    loop_result = result.value

    metrics = read_metrics(tmp_path)
    assert loop_result.total_rounds == 101
    assert metrics[-1].total_games == 101
    assert metrics[-1].process_games_per_second == 0.1


@pytest.mark.asyncio
async def test_train_self_play_keeps_archive_without_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = ModelConfig(
        d_model=4,
        layers=1,
        heads=1,
        max_tokens=512,
    )
    train_config = TrainConfig(
        device="cpu",
        learning_rate=0.0003,
        checkpoint_every_updates=1,
        checkpoint_retention_updates=1,
        max_round_seconds=120.0,
        ppo_epochs=1,
        minibatch_size=512,
    )
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        device=torch.device("cpu"),
    )
    latest_path = tmp_path / "checkpoints" / "latest.json"
    update_path = tmp_path / "checkpoints" / "update-1.json"
    save_result = save_torch_checkpoint(
        manifest_paths=(update_path, latest_path),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=1,
        total_updates=1,
        retained_update_count=train_config.checkpoint_retention_updates,
    )
    assert isinstance(save_result, Ok)
    monkeypatch.setattr(loop, "SelfPlaySession", _NoRewardSession)

    result = await train_self_play(
        run_dir=tmp_path,
        run_id="loop-no-update-test",
        model_config=model_config,
        train_config=train_config,
        max_rounds=1,
        resume=latest_path,
    )
    assert isinstance(result, Ok)
    loop_result = result.value

    latest_metadata = read_torch_checkpoint_metadata(latest_path)
    update_metadata = read_torch_checkpoint_metadata(update_path)
    assert isinstance(latest_metadata, Ok)
    assert isinstance(update_metadata, Ok)
    metrics = read_metrics(tmp_path)
    assert loop_result.total_rounds == 2
    assert loop_result.total_updates == 1
    assert latest_metadata.value.total_rounds == 2
    assert latest_metadata.value.total_updates == 1
    assert update_metadata.value.total_rounds == 1
    assert update_metadata.value.total_updates == 1
    assert metrics[-1].checkpoint_path == str(latest_path)
    assert len(_checkpoint_state_paths(tmp_path)) == 2


@pytest.mark.asyncio
async def test_train_self_play_metrics_follow_post_commit_prune_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = ModelConfig(
        d_model=4,
        layers=1,
        heads=1,
        max_tokens=512,
    )
    train_config = TrainConfig(
        device="cpu",
        learning_rate=0.0003,
        checkpoint_retention_updates=0,
        max_round_seconds=120.0,
        ppo_epochs=1,
        minibatch_size=512,
    )
    state = create_training_state(
        model_config=model_config,
        train_config=train_config,
        device=torch.device("cpu"),
    )
    latest_path = tmp_path / "checkpoints" / "latest.json"
    update_path = tmp_path / "checkpoints" / "update-1.json"
    save_result = save_torch_checkpoint(
        manifest_paths=(update_path, latest_path),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=1,
        total_updates=1,
        retained_update_count=5,
    )
    assert isinstance(save_result, Ok)
    original_unlink = Path.unlink

    def fail_update_unlink(
        self: Path,
        missing_ok: bool = False,
    ) -> None:
        if self.name == "update-1.json":
            raise OSError("busy")
        original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_update_unlink)
    monkeypatch.setattr(loop, "SelfPlaySession", _NoRewardSession)

    result = await train_self_play(
        run_dir=tmp_path,
        run_id="loop-prune-failure-test",
        model_config=model_config,
        train_config=train_config,
        max_rounds=1,
        resume=latest_path,
    )

    assert isinstance(result, Ok)
    latest_metadata = read_torch_checkpoint_metadata(latest_path)
    assert isinstance(latest_metadata, Ok)
    metrics = read_metrics(tmp_path)
    assert latest_metadata.value.total_rounds == 2
    assert metrics[-1].total_games == 2
    assert update_path.exists()


@pytest.mark.asyncio
async def test_train_self_play_returns_round_timeout_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(loop, "SelfPlaySession", _TimeoutSession)

    result = await train_self_play(
        run_dir=tmp_path,
        run_id="loop-timeout-test",
        model_config=ModelConfig(
            d_model=4,
            layers=1,
            heads=1,
            max_tokens=512,
        ),
        train_config=TrainConfig(
            device="cpu",
            learning_rate=0.0003,
            max_round_seconds=0.001,
            ppo_epochs=1,
            minibatch_size=512,
        ),
        max_rounds=1,
        resume=None,
    )

    assert isinstance(result, Rejected)
    assert "training round timed out" in result.reason
    assert read_metrics(tmp_path) == ()


@pytest.mark.asyncio
async def test_train_self_play_rejects_non_finite_ppo_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def update_with_nan(
        self: PPOTrainer,
        batch: RolloutBatch,
    ) -> Ok[PPOUpdateStats] | Rejected:
        assert self.train_config.ppo_epochs > 0
        assert not batch.is_empty()
        return Rejected(reason="PPO value_loss must be finite")

    monkeypatch.setattr(PPOTrainer, "update", update_with_nan)

    result = await train_self_play(
        run_dir=tmp_path,
        run_id="loop-nan-test",
        model_config=ModelConfig(
            d_model=4,
            layers=1,
            heads=1,
            max_tokens=512,
        ),
        train_config=TrainConfig(
            device="cpu",
            learning_rate=0.0003,
            checkpoint_every_updates=1,
            checkpoint_retention_updates=1,
            max_round_seconds=120.0,
            ppo_epochs=1,
            minibatch_size=512,
        ),
        max_rounds=1,
        resume=None,
    )

    assert isinstance(result, Rejected)
    assert "PPO value_loss must be finite" in result.reason
    assert read_metrics(tmp_path) == ()
    assert _checkpoint_state_paths(tmp_path) == ()


class _NoRewardSession:
    def __init__(self, *, policy: object) -> None:
        self._policy = policy

    async def play_round(
        self,
        *,
        max_seconds: float,
    ) -> Ok[TrainingRoundResult] | Rejected:
        return Ok(
            value=TrainingRoundResult(
                rollout=RolloutBatch(trajectories=()),
                team0_reward=0.0,
                team1_reward=0.0,
                generated_action_count=0,
                accepted_action_count=0,
                average_action_choices=0.0,
                elapsed_seconds=max_seconds,
                game_over=False,
            )
        )


class _TimeoutSession:
    def __init__(self, *, policy: object) -> None:
        self._policy = policy

    async def play_round(
        self,
        *,
        max_seconds: float,
    ) -> Ok[TrainingRoundResult] | Rejected:
        assert max_seconds > 0.0
        reason = (
            f"training round timed out after {max_seconds:g} seconds"
        )
        return Rejected(reason=reason)


def _checkpoint_state_paths(run_dir: Path) -> tuple[Path, ...]:
    return tuple(
        sorted((run_dir / "checkpoints" / "objects").glob("*/state.pt"))
    )
