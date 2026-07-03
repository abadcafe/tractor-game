"""Tests for actual self-play training loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from server.training.config import ModelConfig, TrainConfig
from server.training.loop import train_self_play
from server.training.metrics import read_metrics
from server.training.torch_checkpoints import (
    read_torch_checkpoint_metadata,
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
            dropout=0.0,
            max_tokens=512,
        ),
        train_config=TrainConfig(
            device="cpu",
            learning_rate=0.0003,
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
        dropout=0.0,
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
