"""Tests for process-synchronized runtime training state."""

from __future__ import annotations

import torch

from server.foundation.result import Ok, Rejected
from server.training.config import TrainConfig
from server.training.model import ModelConfig
from server.training.runtime.config import ExecutionConfig
from server.training.runtime.state import (
    RuntimeTrainingState,
    capture_runtime_training_state,
    load_runtime_training_state,
    select_canonical_runtime_training_state,
)
from server.training.training_state import create_training_state


def test_select_canonical_runtime_state_accepts_matching_ranks() -> (
    None
):
    first = _runtime_state(weight=2.0, step_count=3)
    second = _runtime_state(weight=2.0, step_count=3)

    selected = select_canonical_runtime_training_state((first, second))

    assert isinstance(selected, Ok)
    assert selected.value is first


def test_select_canonical_runtime_state_rejects_model_divergence() -> (
    None
):
    first = _runtime_state(weight=2.0, step_count=3)
    second = _runtime_state(weight=3.0, step_count=3)

    selected = select_canonical_runtime_training_state((first, second))

    assert isinstance(selected, Rejected)
    assert "model state weight value differs" in selected.reason


def test_select_canonical_state_rejects_optimizer_divergence() -> None:
    first = _runtime_state(weight=2.0, step_count=3)
    second = _runtime_state(weight=2.0, step_count=4)

    selected = select_canonical_runtime_training_state((first, second))

    assert isinstance(selected, Rejected)
    assert "optimizer state step_count differs" in selected.reason


def test_capture_and_load_runtime_training_state_round_trips() -> None:
    model_config = ModelConfig(d_model=8, layers=1, heads=1)
    train_config = TrainConfig()
    source = create_training_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
    )
    target = create_training_state(
        model_config=model_config,
        train_config=TrainConfig(seed=7),
        execution_config=ExecutionConfig(),
    )
    snapshot = capture_runtime_training_state(
        model=source.model,
        trainer=source.trainer,
    )

    load_runtime_training_state(state=target, snapshot=snapshot)
    target_snapshot = capture_runtime_training_state(
        model=target.model,
        trainer=target.trainer,
    )

    assert (
        snapshot.model_state.keys()
        == target_snapshot.model_state.keys()
    )
    for key, value in snapshot.model_state.items():
        assert torch.equal(value, target_snapshot.model_state[key])


def _runtime_state(
    *,
    weight: float,
    step_count: int,
) -> RuntimeTrainingState:
    return RuntimeTrainingState(
        model_state={
            "weight": torch.tensor([weight], dtype=torch.float32),
            "counter": torch.tensor([5], dtype=torch.int64),
        },
        optimizer_state={
            "kind": "ppo_adamw",
            "step_count": step_count,
            "exp_avgs": [
                torch.tensor([weight], dtype=torch.float32),
                None,
            ],
            "exp_avg_sqs": [
                torch.tensor([weight + 1.0], dtype=torch.float32),
                None,
            ],
        },
    )
