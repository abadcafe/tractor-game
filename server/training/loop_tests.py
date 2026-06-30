"""Tests for actual self-play training loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from server.training.config import ModelConfig, TrainConfig
from server.training.loop import train_self_play
from server.training.metrics import read_metrics


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
            max_tokens=256,
        ),
        train_config=TrainConfig(
            device="cpu",
            learning_rate=0.0003,
            max_round_seconds=120.0,
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
