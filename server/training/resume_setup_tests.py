"""Black-box tests for canonical resume timeline preparation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import torch

from server.foundation.result import Ok
from server.training.config import ModelConfig, TrainConfig
from server.training.resume_setup import canonicalize_resume_timeline
from server.training.run_setup import initialize_training_run
from server.training.runtime.config import ExecutionConfig
from server.training.torch_checkpoints.load import (
    load_torch_checkpoint,
    read_torch_checkpoint_metadata,
)
from server.training.torch_checkpoints.save import save_torch_checkpoint
from server.training_events.store import database_path


def test_canonicalize_resume_timeline_removes_future_state(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=2, layers=1, heads=1, max_tokens=512
    )
    train_config = TrainConfig()
    initialized = initialize_training_run(
        run_dir=tmp_path,
        model_config=model_config,
        train_config=train_config,
    )
    assert isinstance(initialized, Ok)
    loaded = load_torch_checkpoint(
        path=initialized.value.checkpoint_path,
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    assert isinstance(loaded, Ok)
    update_one = tmp_path / "checkpoints" / "update-1.json"
    update_two = tmp_path / "checkpoints" / "update-2.json"
    for update, manifest_path in ((1, update_one), (2, update_two)):
        saved = save_torch_checkpoint(
            manifest_paths=(
                manifest_path,
                tmp_path / "checkpoints" / "latest.json",
            ),
            model=loaded.value.model,
            trainer=loaded.value.trainer,
            model_config=model_config,
            train_config=train_config,
            total_rounds=update * 10,
            total_samples=update * 100,
            total_updates=update,
            retained_update_count=5,
        )
        assert isinstance(saved, Ok)
    original_events = _event_rows(tmp_path)

    result = canonicalize_resume_timeline(
        run_dir=tmp_path,
        selected_checkpoint=update_one,
    )

    assert isinstance(result, Ok)
    latest = read_torch_checkpoint_metadata(
        tmp_path / "checkpoints" / "latest.json"
    )
    assert isinstance(latest, Ok)
    assert latest.value.total_updates == 1
    assert update_one.exists()
    assert not update_two.exists()
    assert _event_rows(tmp_path) == original_events


def _event_rows(run_dir: Path) -> tuple[tuple[int, str], ...]:
    with sqlite3.connect(database_path(run_dir)) as connection:
        rows = connection.execute(
            "SELECT sequence, event_json FROM training_logs "
            "ORDER BY sequence"
        ).fetchall()
    events: list[tuple[int, str]] = []
    for sequence, event_json in rows:
        assert isinstance(sequence, int)
        assert isinstance(event_json, str)
        events.append((sequence, event_json))
    return tuple(events)
