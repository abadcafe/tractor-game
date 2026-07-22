"""Tests for runtime checkpoint-state persistence."""

from __future__ import annotations

from pathlib import Path

import torch

from server.foundation.result import Ok
from server.training.config import TrainConfig
from server.training.model import ModelConfig
from server.training.runtime.checkpoint_state import (
    create_initial_runtime_checkpoint_state,
    load_runtime_checkpoint_state,
    save_runtime_checkpoint_state,
)
from server.training.runtime.config import ExecutionConfig
from server.training.runtime.state import (
    select_canonical_runtime_training_state,
)


def test_runtime_checkpoint_state_round_trips_torch_checkpoint(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(d_model=8, layers=1, heads=1)
    train_config = TrainConfig(seed=17)
    execution_config = ExecutionConfig()
    created = create_initial_runtime_checkpoint_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
    )
    checkpoint_path = tmp_path / "checkpoints" / "latest.json"

    saved = save_runtime_checkpoint_state(
        manifest_paths=(checkpoint_path,),
        state=created.state,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
        total_rounds=7,
        total_samples=111,
        total_updates=3,
        retained_update_count=1,
    )
    loaded = load_runtime_checkpoint_state(
        path=checkpoint_path,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
    )

    assert isinstance(saved, Ok)
    assert isinstance(loaded, Ok)
    assert loaded.value.total_rounds == 7
    assert loaded.value.total_samples == 111
    assert loaded.value.total_updates == 3
    state_match = select_canonical_runtime_training_state(
        (created.state, loaded.value.state)
    )
    assert isinstance(state_match, Ok)


def test_save_runtime_checkpoint_state_preserves_cpu_rng_state(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(d_model=8, layers=1, heads=1)
    train_config = TrainConfig(seed=17)
    execution_config = ExecutionConfig()
    created = create_initial_runtime_checkpoint_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
    )
    before = torch.random.get_rng_state().clone()

    saved = save_runtime_checkpoint_state(
        manifest_paths=(tmp_path / "checkpoints" / "latest.json",),
        state=created.state,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
        total_rounds=0,
        total_samples=0,
        total_updates=0,
        retained_update_count=1,
    )

    assert isinstance(saved, Ok)
    assert bool(torch.equal(before, torch.random.get_rng_state()))
