"""Tests for training runtime worker sampling lifecycle."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from server.result import Ok, Rejected
from server.training.runtime.async_ipc import (
    AsyncChildControlEndpoint,
    AsyncCoordinatorControlEndpoint,
    ProcessControlProtocol,
    create_async_process_control_link,
)
from server.training.runtime.config import ExecutionConfig
from server.training.runtime.messages import (
    WorkerCommand,
    WorkerCommandRejected,
    WorkerResponse,
    WorkerSamplingAlreadyStopped,
    WorkerSamplingStarted,
    WorkerSamplingStopped,
    WorkerStartSamplingCommand,
    WorkerStopSamplingCommand,
    decode_worker_command,
    decode_worker_response,
)
from server.training.runtime.worker_sampling_lifecycle import (
    start_worker_sampling_session,
)

_WORKER_TEST_PROTOCOL: ProcessControlProtocol[
    WorkerCommand, WorkerResponse
] = ProcessControlProtocol(
    name="worker",
    decode_command=decode_worker_command,
    decode_response=decode_worker_response,
)


@dataclass(frozen=True, slots=True)
class _FakeWorkerHandle:
    index: int
    control: AsyncCoordinatorControlEndpoint[
        WorkerCommand, WorkerResponse
    ]


@dataclass(frozen=True, slots=True)
class _FakeWorker:
    handle: _FakeWorkerHandle
    child: AsyncChildControlEndpoint[WorkerCommand, WorkerResponse]

    def close(self) -> None:
        self.handle.control.close()
        self.child.close()


def _fake_worker(worker_index: int) -> _FakeWorker:
    link = create_async_process_control_link(
        protocol=_WORKER_TEST_PROTOCOL,
    )
    return _FakeWorker(
        handle=_FakeWorkerHandle(
            index=worker_index,
            control=link.coordinator,
        ),
        child=link.child,
    )


async def _receive_start_command(
    worker: _FakeWorker,
) -> WorkerStartSamplingCommand:
    command = await worker.child.recv_command()
    assert isinstance(command, Ok)
    assert isinstance(command.value, WorkerStartSamplingCommand)
    return command.value


async def _receive_stop_command(
    worker: _FakeWorker,
) -> WorkerStopSamplingCommand:
    command = await worker.child.recv_command()
    assert isinstance(command, Ok)
    assert isinstance(command.value, WorkerStopSamplingCommand)
    return command.value


@pytest.mark.asyncio
async def test_start_sampling_session_stops_commanded_workers() -> None:
    first = _fake_worker(0)
    second = _fake_worker(1)
    try:
        task = asyncio.create_task(
            start_worker_sampling_session(
                handles=(first.handle, second.handle),
                execution_config=ExecutionConfig(
                    game_envs_per_worker=2,
                ),
                policy_version=7,
            )
        )

        first_start = await _receive_start_command(first)
        second_start = await _receive_start_command(second)
        assert first_start.policy_version == 7
        assert second_start.policy_version == 7
        assert first_start.game_env_count == 2
        assert second_start.game_env_count == 2

        start_rejection = await second.child.send_response(
            WorkerCommandRejected(
                worker_index=1,
                command="start_sampling",
                policy_version=7,
                reason="bad start",
            )
        )
        assert isinstance(start_rejection, Ok)

        first_stop = await _receive_stop_command(first)
        second_stop = await _receive_stop_command(second)
        assert first_stop.policy_version == 7
        assert second_stop.policy_version == 7

        late_start = await first.child.send_response(
            WorkerSamplingStarted(worker_index=0, policy_version=7)
        )
        stopped = await first.child.send_response(
            WorkerSamplingStopped(
                worker_index=0,
                policy_version=7,
                cancelled_env_count=2,
            )
        )
        already_stopped = await second.child.send_response(
            WorkerSamplingAlreadyStopped(
                worker_index=1,
                policy_version=7,
            )
        )
        assert isinstance(late_start, Ok)
        assert isinstance(stopped, Ok)
        assert isinstance(already_stopped, Ok)

        result = await task

        assert isinstance(result, Rejected)
        assert result.reason == "worker-1: bad start"
    finally:
        first.close()
        second.close()
