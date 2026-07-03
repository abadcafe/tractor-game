"""Tests for actual self-play training loop."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from server.training import loop
from server.training.config import ModelConfig, TrainConfig
from server.training.loop import train_self_play
from server.training.metrics import read_metrics
from server.training.runner import TrainingRoundResult
from server.training.torch_checkpoints import (
    create_training_state,
    read_torch_checkpoint_metadata,
    save_torch_checkpoint,
)


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

    assert result.total_rounds == 1
    assert result.total_updates == 1
    assert result.checkpoint_path.exists()
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
    first = await train_self_play(
        run_dir=tmp_path,
        run_id="loop-resume-test",
        model_config=model_config,
        train_config=train_config,
        max_rounds=1,
        resume=None,
    )

    resumed = await train_self_play(
        run_dir=tmp_path,
        run_id="loop-resume-test",
        model_config=model_config,
        train_config=train_config,
        max_rounds=1,
        resume=first.checkpoint_path,
    )

    assert resumed.total_rounds == 2
    assert resumed.total_updates == 2
    metadata = read_torch_checkpoint_metadata(resumed.checkpoint_path)
    assert metadata.total_rounds == 2
    assert metadata.total_updates == 2
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
    save_torch_checkpoint(
        manifest_paths=(latest_path,),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=100,
        total_updates=7,
        retained_update_count=train_config.checkpoint_retention_updates,
    )
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

    metrics = read_metrics(tmp_path)
    assert result.total_rounds == 101
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
    save_torch_checkpoint(
        manifest_paths=(update_path, latest_path),
        model=state.model,
        trainer=state.trainer,
        model_config=model_config,
        train_config=train_config,
        total_rounds=1,
        total_updates=1,
        retained_update_count=train_config.checkpoint_retention_updates,
    )
    monkeypatch.setattr(loop, "SelfPlaySession", _NoRewardSession)

    result = await train_self_play(
        run_dir=tmp_path,
        run_id="loop-no-update-test",
        model_config=model_config,
        train_config=train_config,
        max_rounds=1,
        resume=latest_path,
    )

    latest_metadata = read_torch_checkpoint_metadata(latest_path)
    update_metadata = read_torch_checkpoint_metadata(update_path)
    metrics = read_metrics(tmp_path)
    assert result.total_rounds == 2
    assert result.total_updates == 1
    assert latest_metadata.total_rounds == 2
    assert latest_metadata.total_updates == 1
    assert update_metadata.total_rounds == 1
    assert update_metadata.total_updates == 1
    assert metrics[-1].checkpoint_path == str(latest_path)
    assert len(_checkpoint_state_paths(tmp_path)) == 2


class _NoRewardSession:
    def __init__(self, *, policy: object) -> None:
        self._policy = policy

    async def play_round(
        self,
        *,
        max_seconds: float,
    ) -> TrainingRoundResult:
        return TrainingRoundResult(
            rewarded_steps=(),
            team0_reward=0.0,
            team1_reward=0.0,
            generated_action_count=0,
            accepted_action_count=0,
            average_action_choices=0.0,
            elapsed_seconds=max_seconds,
            game_over=False,
        )


def _checkpoint_state_paths(run_dir: Path) -> tuple[Path, ...]:
    return tuple(
        sorted((run_dir / "checkpoints" / "objects").glob("*/state.pt"))
    )
