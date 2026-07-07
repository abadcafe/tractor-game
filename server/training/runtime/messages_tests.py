"""Tests for worker process message protocol."""

from __future__ import annotations

import torch

from server.training.runtime.messages import (
    WorkerLoadStateCommand,
    WorkerStartSamplingCommand,
)
from server.training.runtime.state import RuntimeTrainingState


def test_worker_start_sampling_command_carries_no_state_snapshot() -> (
    None
):
    command = WorkerStartSamplingCommand(
        policy_version=2, game_env_count=3
    )

    assert command.policy_version == 2
    assert command.game_env_count == 3
    assert not hasattr(command, "state")


def test_worker_load_state_command_carries_state_snapshot() -> None:
    state = RuntimeTrainingState(
        model_state={"weight": torch.tensor([1.0])},
        optimizer_state={
            "kind": "adamw",
            "step_count": 0,
            "exp_avgs": [],
            "exp_avg_sqs": [],
        },
    )

    command = WorkerLoadStateCommand(state=state, policy_version=3)

    assert command.state is state
    assert command.policy_version == 3
