"""Worker process entry point for synchronized training."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import torch

from server.foundation import result as _result
from server.foundation.result import Ok, Rejected
from server.training.config import ModelConfig, TrainConfig
from server.training.policy import TrainingPolicy
from server.training.ppo.distributed import (
    PPOUpdatePartition,
    single_update_partition,
)
from server.training.runner import SelfPlaySession, TrainingRoundResult
from server.training.runtime.affinity import apply_cpu_affinity
from server.training.runtime.async_ipc import AsyncChildControlEndpoint
from server.training.runtime.config import CpuSet, ExecutionConfig
from server.training.runtime.distributed import (
    DistributedRankConfig,
    destroy_distributed_rank,
    initialize_distributed_rank,
)
from server.training.runtime.messages import (
    StopWorkerCommand,
    WorkerCommand,
    WorkerCommandKind,
    WorkerCommandRejected,
    WorkerLoadStateCommand,
    WorkerResponse,
    WorkerSamplingAlreadyStopped,
    WorkerSamplingStarted,
    WorkerSamplingStopped,
    WorkerSnapshotCommand,
    WorkerSnapshotCompleted,
    WorkerStartSamplingCommand,
    WorkerStateLoaded,
    WorkerStopSamplingCommand,
    WorkerUpdateCommand,
    WorkerUpdateCompleted,
)
from server.training.runtime.model_rank import (
    AsyncRemotePolicyBatchTransport,
    BatchedPolicyClient,
    LocalModelRank,
    LocalPolicyBatchTransport,
    create_model_replica,
)
from server.training.runtime.model_rank.inference_transport import (
    AsyncPolicyPeer,
)
from server.training.runtime.process_signals import (
    ignore_terminal_interrupt_in_child_process,
)
from server.training.runtime.shared_rollout_arena import (
    RolloutArenaHandle,
    RolloutRoundMetrics,
    SharedRolloutArenaReader,
    SharedRolloutArenaWriter,
    attach_rollout_arena_reader,
    attach_rollout_arena_writer,
)
from server.training.runtime.threads import (
    apply_worker_torch_thread_config,
)
from server.training_events import EventContext, StructuredEventSink


@dataclass(slots=True)
class _WorkerRuntime:
    policy: BatchedPolicyClient
    local_model_rank: LocalModelRank | None
    sessions: tuple[SelfPlaySession, ...]
    arena_writer: SharedRolloutArenaWriter
    arena_reader: SharedRolloutArenaReader
    next_episode_id: int = 0
    completed_round_count: int = 0
    sampling_task: asyncio.Task[_SamplingTaskResult] | None = None
    sampling_policy_version: int | None = None
    sampling_rollout_id: str | None = None


@dataclass(frozen=True, slots=True)
class _EnvRoundResult:
    game_env_index: int
    episode_id: int
    round_data: TrainingRoundResult


type _EnvRoundTaskResult = (
    _result.Ok[_EnvRoundResult] | _result.Rejected
)


@dataclass(frozen=True, slots=True)
class _SamplingSummary:
    active_game_envs: int
    completed_rounds: int
    round_seconds: float
    append_seconds: float
    cancelled_envs: int

    def __post_init__(self) -> None:
        assert self.active_game_envs > 0
        assert self.completed_rounds >= 0
        assert self.round_seconds >= 0.0
        assert self.append_seconds >= 0.0
        assert self.cancelled_envs >= 0


type _SamplingTaskResult = (
    _result.Ok[_SamplingSummary] | _result.Rejected
)


def run_training_worker_process(
    *,
    worker_index: int,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    worker_cpus: CpuSet,
    control: AsyncChildControlEndpoint[WorkerCommand, WorkerResponse],
    event_sink: StructuredEventSink,
    inference_peer: AsyncPolicyPeer | None,
    rollout_arena_handle: RolloutArenaHandle,
    distributed_rank_config: DistributedRankConfig | None,
) -> None:
    """Worker process main loop."""
    ignore_terminal_interrupt_in_child_process()
    asyncio.run(
        _run_training_worker_process_async(
            worker_index=worker_index,
            run_id=run_id,
            model_config=model_config,
            train_config=train_config,
            execution_config=execution_config,
            worker_cpus=worker_cpus,
            control=control,
            event_sink=event_sink,
            inference_peer=inference_peer,
            rollout_arena_handle=rollout_arena_handle,
            distributed_rank_config=distributed_rank_config,
        )
    )


async def _run_training_worker_process_async(
    *,
    worker_index: int,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    worker_cpus: CpuSet,
    control: AsyncChildControlEndpoint[WorkerCommand, WorkerResponse],
    event_sink: StructuredEventSink,
    inference_peer: AsyncPolicyPeer | None,
    rollout_arena_handle: RolloutArenaHandle,
    distributed_rank_config: DistributedRankConfig | None,
) -> None:
    """Async worker process main loop."""
    assert worker_index >= 0
    assert run_id
    setup_result = _setup_worker_runtime(
        worker_index=worker_index,
        worker_cpus=worker_cpus,
    )
    if isinstance(setup_result, Rejected):
        event_sink.emit(
            "process.start",
            error=setup_result.reason,
        )
        event_sink.close()
        await control.send_response(
            _worker_rejection(
                worker_index=worker_index,
                command="setup",
                policy_version=None,
                reason=setup_result.reason,
            )
        )
        return
    sync_result = initialize_distributed_rank(distributed_rank_config)
    if isinstance(sync_result, Rejected):
        event_sink.emit(
            "process.start",
            error=sync_result.reason,
        )
        event_sink.close()
        await control.send_response(
            _worker_rejection(
                worker_index=worker_index,
                command="setup",
                policy_version=None,
                reason=sync_result.reason,
            )
        )
        return
    runtime_result = _create_worker_runtime(
        worker_index=worker_index,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
        device=setup_result.value,
        inference_peer=inference_peer,
        rollout_arena_handle=rollout_arena_handle,
        distributed_rank_config=distributed_rank_config,
        event_sink=event_sink,
    )
    if isinstance(runtime_result, Rejected):
        event_sink.emit(
            "process.start",
            error=runtime_result.reason,
        )
        event_sink.close()
        await control.send_response(
            _worker_rejection(
                worker_index=worker_index,
                command="setup",
                policy_version=None,
                reason=runtime_result.reason,
            )
        )
        destroy_distributed_rank()
        return
    runtime = runtime_result.value
    event_sink.emit(
        "process.start",
        fields={"worker_index": worker_index},
    )
    try:
        while True:
            command_result = await control.recv_command()
            if isinstance(command_result, Rejected):
                return
            command = command_result.value
            response = await _handle_worker_command(
                worker_index=worker_index,
                train_config=train_config,
                execution_config=execution_config,
                runtime=runtime,
                command=command,
                event_sink=event_sink,
            )
            if response is None:
                return
            send_result = await control.send_response(response)
            if isinstance(send_result, Rejected):
                return
    finally:
        await _cancel_active_sampling(runtime=runtime)
        runtime.arena_writer.close()
        runtime.arena_reader.close()
        destroy_distributed_rank()
        event_sink.emit("process.stop")
        event_sink.close()


def _setup_worker_runtime(
    *,
    worker_index: int,
    worker_cpus: CpuSet,
) -> _result.Ok[torch.device] | _result.Rejected:
    affinity_result = apply_cpu_affinity(
        label=f"worker-{worker_index}",
        cpus=worker_cpus,
    )
    if isinstance(affinity_result, Rejected):
        return affinity_result
    thread_result = apply_worker_torch_thread_config()
    if isinstance(thread_result, Rejected):
        return thread_result
    return Ok(value=torch.device("cpu"))


def _create_worker_runtime(
    *,
    worker_index: int,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    device: torch.device,
    inference_peer: AsyncPolicyPeer | None,
    rollout_arena_handle: RolloutArenaHandle,
    distributed_rank_config: DistributedRankConfig | None,
    event_sink: StructuredEventSink,
) -> _result.Ok[_WorkerRuntime] | _result.Rejected:
    if rollout_arena_handle.worker_index != worker_index:
        return Rejected(reason="worker rollout arena handle mismatch")
    if execution_config.uses_model_rank_processes():
        if inference_peer is None:
            return Rejected(
                reason="model-rank worker is missing inference peer"
            )
    arena_writer = attach_rollout_arena_writer(rollout_arena_handle)
    arena_reader = attach_rollout_arena_reader((rollout_arena_handle,))
    if execution_config.uses_model_rank_processes():
        assert inference_peer is not None
        policy = BatchedPolicyClient(
            worker_index=worker_index,
            max_observation_tokens=model_config.max_tokens,
            transport=AsyncRemotePolicyBatchTransport(
                peer=inference_peer,
            ),
            timeout_seconds=(execution_config.timeouts.round_seconds),
            batch_size=execution_config.model_inference_batch_size,
            event_sink=event_sink,
        )
        return Ok(
            value=_WorkerRuntime(
                policy=policy,
                local_model_rank=None,
                sessions=_create_game_envs(
                    policy=policy,
                    count=execution_config.game_envs_per_worker,
                ),
                arena_writer=arena_writer,
                arena_reader=arena_reader,
            )
        )
    core = create_model_replica(
        model_rank_index=worker_index,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
        device=device,
        update_partition=_worker_update_partition(
            distributed_rank_config
        ),
    )
    local_model_rank = LocalModelRank(replica=core)
    policy = BatchedPolicyClient(
        worker_index=worker_index,
        max_observation_tokens=model_config.max_tokens,
        transport=LocalPolicyBatchTransport(replica=core),
        timeout_seconds=(execution_config.timeouts.round_seconds),
        batch_size=execution_config.model_inference_batch_size,
        event_sink=event_sink,
    )
    return Ok(
        value=_WorkerRuntime(
            policy=policy,
            local_model_rank=local_model_rank,
            sessions=_create_game_envs(
                policy=policy,
                count=execution_config.game_envs_per_worker,
            ),
            arena_writer=arena_writer,
            arena_reader=arena_reader,
        )
    )


def _create_game_envs(
    *, policy: TrainingPolicy, count: int
) -> tuple[SelfPlaySession, ...]:
    assert count > 0
    return tuple(SelfPlaySession(policy=policy) for _ in range(count))


def _worker_update_partition(
    config: DistributedRankConfig | None,
) -> PPOUpdatePartition:
    if config is None:
        return single_update_partition()
    return PPOUpdatePartition(
        rank=config.rank,
        world_size=config.world_size,
    )


async def _handle_worker_command(
    *,
    worker_index: int,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    runtime: _WorkerRuntime,
    command: WorkerCommand,
    event_sink: StructuredEventSink,
) -> WorkerResponse | None:
    if isinstance(command, StopWorkerCommand):
        return None
    if isinstance(command, WorkerLoadStateCommand):
        return _load_worker_state(
            worker_index=worker_index,
            runtime=runtime,
            command=command,
        )
    if isinstance(command, WorkerStartSamplingCommand):
        return _start_worker_sampling(
            worker_index=worker_index,
            train_config=train_config,
            execution_config=execution_config,
            runtime=runtime,
            command=command,
            event_sink=event_sink,
        )
    if isinstance(command, WorkerStopSamplingCommand):
        return await _stop_worker_sampling(
            worker_index=worker_index,
            runtime=runtime,
            command=command,
            event_sink=event_sink,
        )
    if isinstance(command, WorkerSnapshotCommand):
        return _snapshot_worker_state(
            worker_index=worker_index,
            runtime=runtime,
            command=command,
        )
    return _run_worker_update(
        worker_index=worker_index,
        train_config=train_config,
        execution_config=execution_config,
        runtime=runtime,
        command=command,
        event_sink=event_sink,
    )


def _load_worker_state(
    *,
    worker_index: int,
    runtime: _WorkerRuntime,
    command: WorkerLoadStateCommand,
) -> WorkerResponse:
    if _sampling_is_active(runtime):
        return _worker_rejection(
            worker_index=worker_index,
            command="load_state",
            policy_version=command.policy_version,
            reason="worker cannot load state while sampling is active",
        )
    if runtime.local_model_rank is None:
        return _worker_rejection(
            worker_index=worker_index,
            command="load_state",
            policy_version=command.policy_version,
            reason="worker does not own a local model rank",
        )
    load_result = runtime.local_model_rank.load_state(
        state=command.state,
        policy_version=command.policy_version,
    )
    if isinstance(load_result, Rejected):
        return _worker_rejection(
            worker_index=worker_index,
            command="load_state",
            policy_version=command.policy_version,
            reason=load_result.reason,
        )
    return WorkerStateLoaded(
        worker_index=worker_index,
        policy_version=command.policy_version,
    )


def _run_worker_update(
    *,
    worker_index: int,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    runtime: _WorkerRuntime,
    command: WorkerUpdateCommand,
    event_sink: StructuredEventSink,
) -> WorkerResponse:
    if _sampling_is_active(runtime):
        event_sink.emit(
            "update.rank",
            context=EventContext(
                policy_version=command.policy_version,
                rollout_id=command.rollout_id,
                worker_index=worker_index,
            ),
            error="worker cannot update while sampling is active",
        )
        return _worker_rejection(
            worker_index=worker_index,
            command="update",
            policy_version=command.policy_version,
            reason="worker cannot update while sampling is active",
        )
    if runtime.local_model_rank is None:
        event_sink.emit(
            "update.rank",
            context=EventContext(
                policy_version=command.policy_version,
                rollout_id=command.rollout_id,
                worker_index=worker_index,
            ),
            error="worker does not own a local model rank",
        )
        return _worker_rejection(
            worker_index=worker_index,
            command="update",
            policy_version=command.policy_version,
            reason="worker does not own a local model rank",
        )
    context = EventContext(
        policy_version=command.policy_version,
        rollout_id=command.rollout_id,
        worker_index=worker_index,
    )
    update_started = time.perf_counter()
    returns_result = runtime.arena_reader.read_rank_batch(
        policy_version=command.policy_version,
        model_rank_index=worker_index,
        device=torch.device("cpu"),
    )
    if isinstance(returns_result, Rejected):
        event_sink.emit(
            "update.rank",
            context=context,
            fields={
                "duration_seconds": max(
                    time.perf_counter() - update_started, 0.0
                )
            },
            error=returns_result.reason,
        )
        return _worker_rejection(
            worker_index=worker_index,
            command="update",
            policy_version=command.policy_version,
            reason=returns_result.reason,
        )
    update_result = runtime.local_model_rank.update(
        returns=returns_result.value,
        policy_version=command.policy_version,
    )
    if isinstance(update_result, Rejected):
        event_sink.emit(
            "update.rank",
            context=context,
            fields={
                "duration_seconds": max(
                    time.perf_counter() - update_started, 0.0
                )
            },
            error=update_result.reason,
        )
        return _worker_rejection(
            worker_index=worker_index,
            command="update",
            policy_version=command.policy_version,
            reason=update_result.reason,
        )
    event_sink.emit(
        "update.rank",
        context=context,
        fields={
            "duration_seconds": max(
                time.perf_counter() - update_started, 0.0
            ),
            "sample_count": int(
                returns_result.value.row_indices.shape[0]
            ),
            "step_count": returns_result.value.total_step_count,
            "round_count": returns_result.value.round_count,
        },
    )
    return WorkerUpdateCompleted(
        worker_index=worker_index,
        policy_version=command.policy_version,
        update_stats=update_result.value,
    )


def _snapshot_worker_state(
    *,
    worker_index: int,
    runtime: _WorkerRuntime,
    command: WorkerSnapshotCommand,
) -> WorkerResponse:
    if _sampling_is_active(runtime):
        return _worker_rejection(
            worker_index=worker_index,
            command="snapshot",
            policy_version=command.policy_version,
            reason="worker cannot snapshot while sampling is active",
        )
    if runtime.local_model_rank is None:
        return _worker_rejection(
            worker_index=worker_index,
            command="snapshot",
            policy_version=command.policy_version,
            reason="worker does not own a local model rank",
        )
    snapshot_result = runtime.local_model_rank.snapshot()
    if isinstance(snapshot_result, Rejected):
        return _worker_rejection(
            worker_index=worker_index,
            command="snapshot",
            policy_version=command.policy_version,
            reason=snapshot_result.reason,
        )
    return WorkerSnapshotCompleted(
        worker_index=worker_index,
        policy_version=command.policy_version,
        state=snapshot_result.value,
    )


def _sampling_summary(
    *,
    active_game_envs: int,
    completed_rounds: int,
    round_seconds: float,
    append_seconds: float,
    cancelled_envs: int,
) -> _SamplingSummary:
    return _SamplingSummary(
        active_game_envs=active_game_envs,
        completed_rounds=completed_rounds,
        round_seconds=round_seconds,
        append_seconds=append_seconds,
        cancelled_envs=cancelled_envs,
    )


def _start_worker_sampling(
    *,
    worker_index: int,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    runtime: _WorkerRuntime,
    command: WorkerStartSamplingCommand,
    event_sink: StructuredEventSink,
) -> WorkerResponse:
    if runtime.sampling_task is not None:
        event_sink.emit(
            "sampling",
            context=EventContext(
                policy_version=command.policy_version,
                rollout_id=command.rollout_id,
                worker_index=worker_index,
            ),
            error="worker sampling is already active",
        )
        return _worker_rejection(
            worker_index=worker_index,
            command="start_sampling",
            policy_version=command.policy_version,
            reason="worker sampling is already active",
        )
    if command.game_env_count > len(runtime.sessions):
        event_sink.emit(
            "sampling",
            context=EventContext(
                policy_version=command.policy_version,
                rollout_id=command.rollout_id,
                worker_index=worker_index,
            ),
            error="worker game env count exceeds configured capacity",
        )
        return _worker_rejection(
            worker_index=worker_index,
            command="start_sampling",
            policy_version=command.policy_version,
            reason="worker game env count exceeds configured capacity",
        )
    runtime.sampling_policy_version = command.policy_version
    runtime.sampling_rollout_id = command.rollout_id
    runtime.sampling_task = asyncio.create_task(
        _run_sampling_until_stopped(
            worker_index=worker_index,
            train_config=train_config,
            execution_config=execution_config,
            runtime=runtime,
            command=command,
            event_sink=event_sink,
        )
    )
    return WorkerSamplingStarted(
        worker_index=worker_index,
        policy_version=command.policy_version,
    )


def _sampling_is_active(runtime: _WorkerRuntime) -> bool:
    return runtime.sampling_task is not None


def _worker_rejection(
    *,
    worker_index: int,
    command: WorkerCommandKind,
    policy_version: int | None,
    reason: str,
) -> WorkerCommandRejected:
    return WorkerCommandRejected(
        worker_index=worker_index,
        command=command,
        policy_version=policy_version,
        reason=reason,
    )


async def _stop_worker_sampling(
    *,
    worker_index: int,
    runtime: _WorkerRuntime,
    command: WorkerStopSamplingCommand,
    event_sink: StructuredEventSink,
) -> WorkerResponse:
    task = runtime.sampling_task
    if task is None:
        return WorkerSamplingAlreadyStopped(
            worker_index=worker_index,
            policy_version=command.policy_version,
        )
    if runtime.sampling_policy_version != command.policy_version:
        event_sink.emit(
            "sampling",
            context=EventContext(
                policy_version=command.policy_version,
                rollout_id=command.rollout_id,
                worker_index=worker_index,
            ),
            error="worker sampling policy version mismatch",
        )
        return _worker_rejection(
            worker_index=worker_index,
            command="stop_sampling",
            policy_version=command.policy_version,
            reason="worker sampling policy version mismatch",
        )
    if runtime.sampling_rollout_id != command.rollout_id:
        event_sink.emit(
            "sampling",
            context=EventContext(
                policy_version=command.policy_version,
                rollout_id=command.rollout_id,
                worker_index=worker_index,
            ),
            error="worker sampling rollout id mismatch",
        )
        return _worker_rejection(
            worker_index=worker_index,
            command="stop_sampling",
            policy_version=command.policy_version,
            reason="worker sampling rollout id mismatch",
        )
    if not task.done():
        task.cancel()
    sampling_result = await task
    runtime.sampling_task = None
    runtime.sampling_policy_version = None
    runtime.sampling_rollout_id = None
    if isinstance(sampling_result, Rejected):
        event_sink.emit(
            "sampling",
            context=EventContext(
                policy_version=command.policy_version,
                rollout_id=command.rollout_id,
                worker_index=worker_index,
            ),
            error=sampling_result.reason,
        )
        return _worker_rejection(
            worker_index=worker_index,
            command="stop_sampling",
            policy_version=command.policy_version,
            reason=sampling_result.reason,
        )
    policy_stats = runtime.policy.drain_stats()
    event_sink.emit(
        "sampling",
        context=EventContext(
            policy_version=command.policy_version,
            rollout_id=command.rollout_id,
            worker_index=worker_index,
        ),
        fields={
            "active_game_envs": 0,
            "completed_rounds": sampling_result.value.completed_rounds,
            "round_seconds": sampling_result.value.round_seconds,
            "policy_wait_seconds": policy_stats.wait_seconds,
            "decision_count": policy_stats.decision_count,
            "arena_append_seconds": (
                sampling_result.value.append_seconds
            ),
            "cancelled_game_envs": sampling_result.value.cancelled_envs,
        },
    )
    return WorkerSamplingStopped(
        worker_index=worker_index,
        policy_version=command.policy_version,
        cancelled_env_count=sampling_result.value.cancelled_envs,
    )


async def _cancel_active_sampling(*, runtime: _WorkerRuntime) -> None:
    task = runtime.sampling_task
    if task is None:
        return
    if not task.done():
        task.cancel()
    await task
    runtime.sampling_task = None
    runtime.sampling_policy_version = None
    runtime.sampling_rollout_id = None


async def _run_sampling_until_stopped(
    *,
    worker_index: int,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    runtime: _WorkerRuntime,
    command: WorkerStartSamplingCommand,
    event_sink: StructuredEventSink,
) -> _result.Ok[_SamplingSummary] | _result.Rejected:
    assert command.game_env_count <= len(runtime.sessions)
    sessions = list(runtime.sessions)
    completed_rounds = 0
    round_seconds = 0.0
    append_seconds = 0.0
    cancelled_envs = 0
    pending: dict[asyncio.Task[_EnvRoundTaskResult], int] = {}
    for game_env_index in range(command.game_env_count):
        task = _schedule_round(
            session=sessions[game_env_index],
            game_env_index=game_env_index,
            episode_id=_next_worker_episode_id(
                worker_index=worker_index,
                local_episode_id=runtime.next_episode_id,
            ),
            train_config=train_config,
            execution_config=execution_config,
            policy_version=command.policy_version,
            rollout_id=command.rollout_id,
            worker_index=worker_index,
            event_sink=event_sink,
        )
        runtime.next_episode_id += 1
        pending[task] = game_env_index
    try:
        while pending:
            done, _pending_tasks = await asyncio.wait(
                pending.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                game_env_index = pending.pop(task)
                round_result = task.result()
                if isinstance(round_result, Rejected):
                    cancelled_count = len(pending)
                    await _cancel_pending_rounds(
                        pending=pending,
                        sessions=sessions,
                        policy=runtime.policy,
                    )
                    cancelled_envs += cancelled_count
                    runtime.arena_writer.record_cancelled_envs(
                        cancelled_count
                    )
                    runtime.sessions = tuple(sessions)
                    return round_result
                env_round = round_result.value
                round_data = env_round.round_data
                completed_rounds += 1
                round_seconds += round_data.elapsed_seconds
                if round_data.game_over:
                    sessions[game_env_index] = SelfPlaySession(
                        policy=runtime.policy
                    )
                append_start = time.perf_counter()
                append_result = runtime.arena_writer.append_round(
                    policy_version=command.policy_version,
                    metrics=_round_metrics(round_data),
                    commit=round_data.returns,
                )
                append_seconds += max(
                    time.perf_counter() - append_start, 0.0
                )
                if isinstance(append_result, Rejected):
                    cancelled_count = len(pending)
                    await _cancel_pending_rounds(
                        pending=pending,
                        sessions=sessions,
                        policy=runtime.policy,
                    )
                    cancelled_envs += cancelled_count
                    runtime.arena_writer.record_cancelled_envs(
                        cancelled_count
                    )
                    runtime.sessions = tuple(sessions)
                    return append_result
                runtime.completed_round_count += 1
                if append_result.value.capacity_reached:
                    cancelled_count = len(pending)
                    await _cancel_pending_rounds(
                        pending=pending,
                        sessions=sessions,
                        policy=runtime.policy,
                    )
                    cancelled_envs += cancelled_count
                    runtime.arena_writer.record_cancelled_envs(
                        cancelled_count
                    )
                    runtime.sessions = tuple(sessions)
                    return Ok(
                        value=_sampling_summary(
                            active_game_envs=command.game_env_count,
                            completed_rounds=completed_rounds,
                            round_seconds=round_seconds,
                            append_seconds=append_seconds,
                            cancelled_envs=cancelled_envs,
                        )
                    )
                task = _schedule_round(
                    session=sessions[game_env_index],
                    game_env_index=game_env_index,
                    episode_id=_next_worker_episode_id(
                        worker_index=worker_index,
                        local_episode_id=runtime.next_episode_id,
                    ),
                    train_config=train_config,
                    execution_config=execution_config,
                    policy_version=command.policy_version,
                    rollout_id=command.rollout_id,
                    worker_index=worker_index,
                    event_sink=event_sink,
                )
                runtime.next_episode_id += 1
                pending[task] = game_env_index
    except asyncio.CancelledError:
        cancelled_count = len(pending)
        await _cancel_pending_rounds(
            pending=pending,
            sessions=sessions,
            policy=runtime.policy,
        )
        cancelled_envs += cancelled_count
        runtime.arena_writer.record_cancelled_envs(cancelled_count)
        runtime.sessions = tuple(sessions)
        return Ok(
            value=_sampling_summary(
                active_game_envs=command.game_env_count,
                completed_rounds=completed_rounds,
                round_seconds=round_seconds,
                append_seconds=append_seconds,
                cancelled_envs=cancelled_envs,
            )
        )
    runtime.sessions = tuple(sessions)
    return Ok(
        value=_sampling_summary(
            active_game_envs=command.game_env_count,
            completed_rounds=completed_rounds,
            round_seconds=round_seconds,
            append_seconds=append_seconds,
            cancelled_envs=cancelled_envs,
        )
    )


def _schedule_round(
    *,
    session: SelfPlaySession,
    game_env_index: int,
    episode_id: int,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    policy_version: int,
    rollout_id: str,
    worker_index: int,
    event_sink: StructuredEventSink,
) -> asyncio.Task[_EnvRoundTaskResult]:
    return asyncio.create_task(
        _play_worker_round(
            session=session,
            game_env_index=game_env_index,
            episode_id=episode_id,
            base_seed=train_config.seed,
            policy_version=policy_version,
            rollout_id=rollout_id,
            max_seconds=execution_config.timeouts.round_seconds,
            worker_index=worker_index,
            event_sink=event_sink,
        )
    )


async def _play_worker_round(
    *,
    session: SelfPlaySession,
    game_env_index: int,
    episode_id: int,
    base_seed: int,
    policy_version: int,
    rollout_id: str,
    max_seconds: float,
    worker_index: int,
    event_sink: StructuredEventSink,
) -> _EnvRoundTaskResult:
    started = time.perf_counter()
    context = EventContext(
        policy_version=policy_version,
        rollout_id=rollout_id,
        worker_index=worker_index,
        game_env_index=game_env_index,
        episode_id=episode_id,
    )
    result = await session.play_round(
        base_seed=base_seed,
        policy_version=policy_version,
        rollout_id=rollout_id,
        episode_id=episode_id,
        max_seconds=max_seconds,
    )
    if isinstance(result, Rejected):
        event_sink.emit(
            "round",
            context=context,
            fields={
                "duration_seconds": max(
                    time.perf_counter() - started, 0.0
                ),
            },
            error=result.reason,
        )
        return result
    round_data = result.value
    event_sink.emit(
        "round",
        context=context,
        fields={
            "duration_seconds": round_data.elapsed_seconds,
            "team0_reward": round_data.team0_reward,
            "team1_reward": round_data.team1_reward,
            "generated_action_count": round_data.generated_action_count,
            "accepted_action_count": round_data.accepted_action_count,
            "action_choice_count": round_data.action_choice_count,
            "decision_count": round_data.returns.sample_count(),
            "game_over": round_data.game_over,
        },
    )
    return Ok(
        value=_EnvRoundResult(
            game_env_index=game_env_index,
            episode_id=episode_id,
            round_data=round_data,
        )
    )


async def _cancel_pending_rounds(
    *,
    pending: dict[asyncio.Task[_EnvRoundTaskResult], int],
    sessions: list[SelfPlaySession],
    policy: TrainingPolicy,
) -> None:
    for task in pending:
        task.cancel()
    for task, game_env_index in tuple(pending.items()):
        try:
            await task
        except asyncio.CancelledError:
            pass
        await sessions[game_env_index].close()
        sessions[game_env_index] = SelfPlaySession(policy=policy)
    pending.clear()


def _next_worker_episode_id(
    *, worker_index: int, local_episode_id: int
) -> int:
    assert worker_index >= 0
    assert local_episode_id >= 0
    return worker_index * 1_000_000_000 + local_episode_id


def _round_metrics(
    round_data: TrainingRoundResult,
) -> RolloutRoundMetrics:
    return RolloutRoundMetrics(
        team0_reward=round_data.team0_reward,
        team1_reward=round_data.team1_reward,
        generated_action_count=round_data.generated_action_count,
        accepted_action_count=round_data.accepted_action_count,
        action_choice_count=round_data.action_choice_count,
        decision_count=round_data.returns.sample_count(),
        elapsed_seconds=round_data.elapsed_seconds,
        game_over=round_data.game_over,
    )
