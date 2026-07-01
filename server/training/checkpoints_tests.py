"""Tests for portable training checkpoint persistence."""

from __future__ import annotations

from pathlib import Path

from server.training.checkpoints import (
    TrainingCheckpoint,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_round_trips_json_payload(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    checkpoint = TrainingCheckpoint(
        run_id="rk3588-run",
        total_games=12,
        total_updates=3,
        model_config={"d_model": 128, "layers": 3},
        train_config={"device": "cpu"},
        token_schema_version="card-context-v1",
        rules_progress_version="required-level-v1",
        model_state={"weights": [1.0, 2.0]},
        optimizer_state={"step": 3},
        rng_state={"seed": 9},
    )

    save_checkpoint(path, checkpoint)
    loaded = load_checkpoint(path)

    assert loaded == checkpoint
