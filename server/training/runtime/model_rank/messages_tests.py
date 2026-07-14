"""Tests for model-rank process message protocol."""

from __future__ import annotations

import torch

from server.training.runtime.model_rank.messages import (
    ModelRankLoadStateCommand,
    ModelRankSnapshotCommand,
    ModelRankUpdateCommand,
)
from server.training.runtime.state import RuntimeTrainingState


def test_model_rank_load_state_command_carries_state_snapshot() -> None:
    state = _runtime_state()

    command = ModelRankLoadStateCommand(state=state, policy_version=3)

    assert command.state is state
    assert command.policy_version == 3


def test_model_rank_update_command_carries_no_state_snapshot() -> None:
    command = ModelRankUpdateCommand(
        policy_version=4, rollout_id="rollout-4"
    )

    assert command.policy_version == 4
    assert not hasattr(command, "state")


def test_model_rank_snapshot_command_carries_no_state_snapshot() -> (
    None
):
    command = ModelRankSnapshotCommand(policy_version=5)

    assert command.policy_version == 5
    assert not hasattr(command, "state")


def _runtime_state() -> RuntimeTrainingState:
    return RuntimeTrainingState(
        model_state={"weight": torch.tensor([1.0])},
        optimizer_state={
            "kind": "adamw",
            "step_count": 0,
            "exp_avgs": [],
            "exp_avg_sqs": [],
        },
    )
