"""Tests for worker process message protocol."""

from __future__ import annotations

import torch

from server.training.runtime.messages import (
    WorkerCommandRejected,
    WorkerLoadStateCommand,
    WorkerSamplingAlreadyStopped,
    WorkerSamplingStarted,
    WorkerSamplingStopped,
    WorkerSnapshotCommand,
    WorkerStartSamplingCommand,
    WorkerStopSamplingCommand,
    WorkerUpdateCommand,
    decode_worker_command,
    decode_worker_response,
)
from server.training.runtime.state import RuntimeTrainingState


def test_worker_start_sampling_command_carries_no_state_snapshot() -> (
    None
):
    command = WorkerStartSamplingCommand(
        policy_version=2, rollout_id="rollout-2", game_env_count=3
    )

    assert command.policy_version == 2
    assert command.game_env_count == 3
    assert not hasattr(command, "state")


def test_decode_worker_stop_sampling_command_accepts_stop() -> None:
    command = WorkerStopSamplingCommand(
        policy_version=2, rollout_id="rollout-2"
    )

    decoded = decode_worker_command(command)

    assert decoded is command


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


def test_worker_update_command_carries_no_state_snapshot() -> None:
    command = WorkerUpdateCommand(
        policy_version=4, rollout_id="rollout-4"
    )

    assert command.policy_version == 4
    assert not hasattr(command, "state")


def test_worker_snapshot_command_carries_no_state_snapshot() -> None:
    command = WorkerSnapshotCommand(policy_version=5)

    assert command.policy_version == 5
    assert not hasattr(command, "state")


def test_decode_worker_sampling_started_response_accepts_start() -> (
    None
):
    response = WorkerSamplingStarted(worker_index=1, policy_version=6)

    decoded = decode_worker_response(response)

    assert decoded is response


def test_decode_worker_sampling_stopped_response_accepts_stop() -> None:
    response = WorkerSamplingStopped(
        worker_index=1,
        policy_version=6,
        cancelled_env_count=2,
    )

    decoded = decode_worker_response(response)

    assert decoded is response


def test_decode_worker_already_stopped_accepts_cleanup() -> None:
    response = WorkerSamplingAlreadyStopped(
        worker_index=1,
        policy_version=6,
    )

    decoded = decode_worker_response(response)

    assert decoded is response


def test_decode_worker_command_rejected_response_accepts_context() -> (
    None
):
    response = WorkerCommandRejected(
        worker_index=1,
        command="start_sampling",
        policy_version=6,
        reason="start failed",
    )

    decoded = decode_worker_response(response)

    assert decoded is response


def test_decode_worker_rejection_accepts_setup_without_policy() -> None:
    response = WorkerCommandRejected(
        worker_index=1,
        command="setup",
        policy_version=None,
        reason="setup failed",
    )

    decoded = decode_worker_response(response)

    assert response.command == "setup"
    assert response.policy_version is None
    assert decoded is response
