"""Tests for training runtime worker sampling lifecycle."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from pathlib import Path

import pytest

from server.result import Ok, Rejected
from server.training.config import ModelConfig, TrainConfig
from server.training.runtime import training_runtime
from server.training.runtime.async_ipc import (
    AsyncChildControlEndpoint,
    AsyncCoordinatorControlEndpoint,
    ProcessControlProtocol,
    create_async_process_control_link,
)
from server.training.runtime.config import (
    ExecutionConfig,
    ExecutionTimeouts,
    ModelRankPlacement,
)
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
from server.training.runtime.shared_rollout_arena import (
    RolloutArenaSnapshot,
    SharedRolloutArenaGroup,
    close_shared_rollout_arenas,
    create_shared_rollout_arena_group,
)
from server.training.runtime.worker_sampling_lifecycle import (
    WorkerControlHandle,
    WorkerSamplingSession,
    start_worker_sampling_session,
)

_WORKER_TEST_PROTOCOL: ProcessControlProtocol[
    WorkerCommand, WorkerResponse
] = ProcessControlProtocol(
    name="worker",
    decode_command=decode_worker_command,
    decode_response=decode_worker_response,
)


@dataclass(slots=True)
class _InterruptingStarter:
    original: Callable[[BaseProcess], None]
    started_processes: list[BaseProcess]
    interrupt_after_start_count: int
    start_count: int = 0

    def __call__(self, process: BaseProcess) -> None:
        self.original(process)
        self.started_processes.append(process)
        self.start_count += 1
        if self.start_count == self.interrupt_after_start_count:
            raise KeyboardInterrupt


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


@dataclass(frozen=True, slots=True)
class _FakeWorkerPool:
    handles: tuple[WorkerControlHandle, ...]


@dataclass(frozen=True, slots=True)
class _FakeRuntimePools:
    worker_pool: _FakeWorkerPool
    model_rank_pool: None
    worker_inference_links: tuple[object, ...]
    rollout_arena_group: SharedRolloutArenaGroup


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


@pytest.mark.asyncio
async def test_start_sampling_cleans_sent_worker_on_failure() -> None:
    first = _fake_worker(0)
    second = _fake_worker(1)
    try:
        first.handle.control.close()
        task = asyncio.create_task(
            start_worker_sampling_session(
                handles=(first.handle, second.handle),
                execution_config=ExecutionConfig(
                    game_envs_per_worker=2,
                ),
                policy_version=7,
            )
        )

        second_start = await _receive_start_command(second)
        assert second_start.policy_version == 7

        second_stop = await _receive_stop_command(second)
        assert second_stop.policy_version == 7
        stopped = await second.child.send_response(
            WorkerSamplingAlreadyStopped(
                worker_index=1,
                policy_version=7,
            )
        )
        assert isinstance(stopped, Ok)

        result = await task

        assert isinstance(result, Rejected)
        assert "async IPC endpoint is closed" in result.reason
    finally:
        first.close()
        second.close()


def test_rollout_arena_capacity_per_worker_keeps_aggregate_target() -> (
    None
):
    execution_config = ExecutionConfig(
        worker_cpus=(0, 1, 2),
        game_envs_per_worker=2,
        samples_per_update=32,
    )

    capacity = training_runtime.rollout_arena_capacity_per_worker(
        execution_config
    )

    assert capacity == 32 + 512


@pytest.mark.asyncio
async def test_rollout_sample_wait_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = mp.get_context("spawn")
    group_result = create_shared_rollout_arena_group(
        context=context,
        worker_count=1,
        arena_capacity_per_worker=1,
        policy_version=3,
    )
    assert isinstance(group_result, Ok)
    group = group_result.value

    def blocking_wait_rollout_sample_target(
        *,
        group: SharedRolloutArenaGroup,
        policy_version: int,
        target_sample_count: int,
        timeout_seconds: float,
    ) -> Ok[RolloutArenaSnapshot] | Rejected:
        assert group.handles
        assert policy_version == 3
        assert target_sample_count == 1
        assert timeout_seconds == 1.0
        time.sleep(0.2)
        return Ok(value=_empty_snapshot(policy_version=policy_version))

    monkeypatch.setattr(
        training_runtime,
        "wait_rollout_sample_target",
        blocking_wait_rollout_sample_target,
    )

    try:
        start = time.perf_counter()
        wait_task = asyncio.create_task(
            training_runtime.wait_rollout_sample_target_async(
                group=group,
                policy_version=3,
                target_sample_count=1,
                timeout_seconds=1.0,
            )
        )
        await asyncio.sleep(0.02)
        elapsed = time.perf_counter() - start
        result = await wait_task
    finally:
        close_shared_rollout_arenas(group)

    assert elapsed < 0.1
    assert isinstance(result, Ok)


@pytest.mark.asyncio
async def test_runtime_poisoned_after_sampling_stop_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = mp.get_context("spawn")
    group_result = create_shared_rollout_arena_group(
        context=context,
        worker_count=1,
        arena_capacity_per_worker=1,
        policy_version=0,
    )
    assert isinstance(group_result, Ok)
    group = group_result.value
    fake_worker = _fake_worker(0)
    pools = _FakeRuntimePools(
        worker_pool=_FakeWorkerPool(handles=(fake_worker.handle,)),
        model_rank_pool=None,
        worker_inference_links=(),
        rollout_arena_group=group,
    )
    start_calls = 0
    force_calls = 0
    group_closed = False

    async def started_sampling_session(
        *,
        handles: tuple[WorkerControlHandle, ...],
        execution_config: ExecutionConfig,
        policy_version: int,
    ) -> Ok[WorkerSamplingSession] | Rejected:
        nonlocal start_calls
        assert execution_config.samples_per_update == 1
        start_calls += 1
        return Ok(
            value=WorkerSamplingSession(
                policy_version=policy_version,
                commanded_handles=handles,
                started_handles=handles,
            )
        )

    async def completed_rollout_wait(
        *,
        group: SharedRolloutArenaGroup,
        policy_version: int,
        target_sample_count: int,
        timeout_seconds: float,
    ) -> Ok[RolloutArenaSnapshot] | Rejected:
        assert group.handles
        assert target_sample_count == 1
        assert timeout_seconds == 2.0
        return Ok(value=_empty_snapshot(policy_version=policy_version))

    async def rejected_sampling_stop(
        *, session: WorkerSamplingSession, timeout_seconds: float
    ) -> Ok[tuple[WorkerSamplingStopped, ...]] | Rejected:
        assert session.policy_version == 3
        assert timeout_seconds == 3.0
        return Rejected(reason="sampling stop timed out")

    def force_stop_runtime_pools(_pools: object) -> None:
        nonlocal force_calls, group_closed
        force_calls += 1
        fake_worker.close()
        close_shared_rollout_arenas(group)
        group_closed = True

    def start_runtime_pools(
        *,
        run_dir: Path,
        run_id: str,
        model_config: ModelConfig,
        train_config: TrainConfig,
        execution_config: ExecutionConfig,
    ) -> Ok[_FakeRuntimePools] | Rejected:
        assert run_dir == Path("unused")
        assert run_id == "poisoned-runtime"
        assert model_config.d_model == 4
        assert train_config.ppo_epochs == 1
        assert execution_config.samples_per_update == 1
        return Ok(value=pools)

    monkeypatch.setattr(
        training_runtime,
        "_start_runtime_pools",
        start_runtime_pools,
    )
    monkeypatch.setattr(
        training_runtime,
        "start_worker_sampling_session",
        started_sampling_session,
    )
    monkeypatch.setattr(
        training_runtime,
        "wait_rollout_sample_target_async",
        completed_rollout_wait,
    )
    monkeypatch.setattr(
        training_runtime,
        "stop_worker_sampling_session",
        rejected_sampling_stop,
    )
    monkeypatch.setattr(
        training_runtime,
        "_force_stop_runtime_pools",
        force_stop_runtime_pools,
    )
    runtime_result = training_runtime.open_training_runtime(
        run_dir=Path("unused"),
        run_id="poisoned-runtime",
        model_config=ModelConfig(d_model=4, layers=1, heads=1),
        train_config=TrainConfig(ppo_epochs=1),
        execution_config=ExecutionConfig(
            samples_per_update=1,
            timeouts=ExecutionTimeouts(
                rollout_sample_seconds=2.0,
                sampling_stop_seconds=3.0,
            ),
        ),
    )
    assert isinstance(runtime_result, Ok)
    runtime = runtime_result.value
    try:
        first = await runtime.run_update(policy_version=3)
        second = await runtime.run_update(policy_version=4)
        await runtime.close()
    finally:
        if not group_closed:
            fake_worker.close()
            close_shared_rollout_arenas(group)

    assert isinstance(first, Rejected)
    assert first.reason == "sampling stop timed out"
    assert second is first
    assert start_calls == 1
    assert force_calls == 1


def test_start_runtime_pools_cleans_worker_started_before_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _speed_startup_process_cleanup(monkeypatch)
    started_processes: list[BaseProcess] = []
    starter = _InterruptingStarter(
        original=(
            training_runtime.start_child_process_ignoring_terminal_interrupt
        ),
        started_processes=started_processes,
        interrupt_after_start_count=1,
    )
    monkeypatch.setattr(
        training_runtime,
        "run_training_worker_process",
        _sleep_forever_training_process,
    )
    monkeypatch.setattr(
        training_runtime,
        "start_child_process_ignoring_terminal_interrupt",
        starter,
    )
    interrupted = False
    try:
        training_runtime.open_training_runtime(
            run_dir=tmp_path,
            run_id="interrupt-worker",
            model_config=ModelConfig(d_model=4, layers=1, heads=1),
            train_config=TrainConfig(),
            execution_config=ExecutionConfig(
                worker_cpus=(0, 1),
                samples_per_update=1,
            ),
        )
    except KeyboardInterrupt:
        interrupted = True
    finally:
        _kill_live_processes(started_processes)

    assert interrupted
    assert starter.start_count == 1
    assert all(not process.is_alive() for process in started_processes)


def test_start_runtime_pools_cleans_model_rank_started_before_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _speed_startup_process_cleanup(monkeypatch)
    started_processes: list[BaseProcess] = []
    starter = _InterruptingStarter(
        original=(
            training_runtime.start_child_process_ignoring_terminal_interrupt
        ),
        started_processes=started_processes,
        interrupt_after_start_count=1,
    )
    monkeypatch.setattr(
        training_runtime,
        "run_model_rank_process",
        _sleep_forever_training_process,
    )
    monkeypatch.setattr(
        training_runtime,
        "start_child_process_ignoring_terminal_interrupt",
        starter,
    )
    interrupted = False
    try:
        training_runtime.open_training_runtime(
            run_dir=tmp_path,
            run_id="interrupt-model-rank",
            model_config=ModelConfig(d_model=4, layers=1, heads=1),
            train_config=TrainConfig(),
            execution_config=ExecutionConfig(
                model_ranks=ModelRankPlacement(
                    kind="cuda",
                    devices=("cuda:0",),
                ),
                samples_per_update=1,
            ),
        )
    except KeyboardInterrupt:
        interrupted = True
    finally:
        _kill_live_processes(started_processes)

    assert interrupted
    assert starter.start_count == 1
    assert all(not process.is_alive() for process in started_processes)


def _sleep_forever_training_process(**_kwargs: object) -> None:
    while True:
        time.sleep(0.1)


def _empty_snapshot(*, policy_version: int) -> RolloutArenaSnapshot:
    return RolloutArenaSnapshot(
        policy_version=policy_version,
        capacity=1,
        round_count=0,
        sample_count=1,
        generated_action_count=0,
        accepted_action_count=0,
        action_choice_count=0,
        game_over_count=0,
        dropped_sample_count=0,
        cancelled_env_count=0,
        total_step_count=0,
        max_step_count=0,
        team0_reward_sum=0.0,
        team1_reward_sum=0.0,
        elapsed_seconds_max=0.0,
    )


def _speed_startup_process_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        training_runtime, "_GRACEFUL_PROCESS_STOP_SECONDS", 0.01
    )
    monkeypatch.setattr(
        training_runtime, "_TERMINATED_PROCESS_STOP_SECONDS", 0.1
    )


def _kill_live_processes(processes: list[BaseProcess]) -> None:
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)
