"""Worker process entry point for synchronized training."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import torch

from server import result as _result
from server.result import Ok, Rejected
from server.training.config import ModelConfig, TrainConfig
from server.training.policy import TrainingPolicy
from server.training.ppo.distributed import (
    PPOUpdatePartition,
    single_update_partition,
)
from server.training.runner import SelfPlaySession, TrainingRoundResult
from server.training.runtime.affinity import apply_cpu_affinity
from server.training.runtime.config import CpuSet, ExecutionConfig
from server.training.runtime.distributed import (
    DistributedRankConfig,
    destroy_distributed_rank,
    initialize_distributed_rank,
)
from server.training.runtime.messages import (
    StopWorkerCommand,
    WorkerCommand,
    WorkerCommandReceiver,
    WorkerLoadStateCommand,
    WorkerRejected,
    WorkerResponse,
    WorkerResponseSender,
    WorkerSamplingStopped,
    WorkerStartSamplingCommand,
    WorkerStateLoaded,
    WorkerUpdateCommand,
    WorkerUpdateCompleted,
)
from server.training.runtime.model_rank import (
    DirectPolicyClient,
    FramedPolicyClient,
    LocalModelRank,
    create_model_replica,
)
from server.training.runtime.model_rank.inference_transport import (
    ConnectionPolicyRequestSender,
    ConnectionPolicyResponseReceiver,
)
from server.training.runtime.shared_rollout_arena import (
    RolloutArenaHandle,
    RolloutRoundMetrics,
    SharedRolloutArenaReader,
    SharedRolloutArenaWriter,
    attach_rollout_arena_reader,
    attach_rollout_arena_writer,
)
from server.training.runtime.telemetry import (
    TelemetryEvent,
    TelemetrySink,
)
from server.training.runtime.threads import apply_torch_thread_config


@dataclass(slots=True)
class _WorkerRuntime:
    policy: TrainingPolicy
    local_model_rank: LocalModelRank | None
    sessions: tuple[SelfPlaySession, ...]
    arena_writer: SharedRolloutArenaWriter
    arena_reader: SharedRolloutArenaReader
    next_episode_id: int = 0


@dataclass(frozen=True, slots=True)
class _EnvRoundResult:
    game_env_index: int
    episode_id: int
    round_data: TrainingRoundResult


type _EnvRoundTaskResult = (
    _result.Ok[_EnvRoundResult] | _result.Rejected
)


def run_training_worker_process(
    *,
    worker_index: int,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    worker_cpus: CpuSet,
    command_receiver: WorkerCommandReceiver,
    response_sender: WorkerResponseSender,
    telemetry_sink: TelemetrySink,
    inference_request_sender: ConnectionPolicyRequestSender | None,
    inference_response_receiver: (
        ConnectionPolicyResponseReceiver | None
    ),
    rollout_arena_handles: tuple[RolloutArenaHandle, ...],
    distributed_rank_config: DistributedRankConfig | None,
) -> None:
    """Worker process main loop."""
    assert worker_index >= 0
    setup_result = _setup_worker_runtime(
        worker_index=worker_index,
        execution_config=execution_config,
        worker_cpus=worker_cpus,
    )
    if isinstance(setup_result, Rejected):
        response_sender.put(
            WorkerRejected(
                worker_index=worker_index,
                reason=setup_result.reason,
            )
        )
        return
    sync_result = initialize_distributed_rank(distributed_rank_config)
    if isinstance(sync_result, Rejected):
        response_sender.put(
            WorkerRejected(
                worker_index=worker_index,
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
        inference_request_sender=inference_request_sender,
        inference_response_receiver=inference_response_receiver,
        rollout_arena_handles=rollout_arena_handles,
        distributed_rank_config=distributed_rank_config,
    )
    if isinstance(runtime_result, Rejected):
        response_sender.put(
            WorkerRejected(
                worker_index=worker_index,
                reason=runtime_result.reason,
            )
        )
        destroy_distributed_rank()
        return
    runtime = runtime_result.value
    try:
        while True:
            command = command_receiver.get()
            response = _handle_worker_command(
                worker_index=worker_index,
                run_id=run_id,
                train_config=train_config,
                execution_config=execution_config,
                runtime=runtime,
                command=command,
                telemetry_sink=telemetry_sink,
            )
            if response is None:
                return
            response_sender.put(response)
    finally:
        runtime.arena_writer.close()
        runtime.arena_reader.close()
        destroy_distributed_rank()


def _setup_worker_runtime(
    *,
    worker_index: int,
    execution_config: ExecutionConfig,
    worker_cpus: CpuSet,
) -> _result.Ok[torch.device] | _result.Rejected:
    affinity_result = apply_cpu_affinity(
        label=f"worker-{worker_index}",
        cpus=worker_cpus,
    )
    if isinstance(affinity_result, Rejected):
        return affinity_result
    thread_result = apply_torch_thread_config(
        num_threads=1 if worker_cpus else None,
        num_interop_threads=1 if worker_cpus else None,
    )
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
    inference_request_sender: ConnectionPolicyRequestSender | None,
    inference_response_receiver: (
        ConnectionPolicyResponseReceiver | None
    ),
    rollout_arena_handles: tuple[RolloutArenaHandle, ...],
    distributed_rank_config: DistributedRankConfig | None,
) -> _result.Ok[_WorkerRuntime] | _result.Rejected:
    if worker_index >= len(rollout_arena_handles):
        return Rejected(reason="worker rollout arena handle is missing")
    if execution_config.uses_model_rank_processes():
        if (
            inference_request_sender is None
            or inference_response_receiver is None
        ):
            return Rejected(
                reason="model-rank worker is missing inference queues"
            )
    arena_writer = attach_rollout_arena_writer(
        rollout_arena_handles[worker_index]
    )
    arena_reader = attach_rollout_arena_reader(rollout_arena_handles)
    if execution_config.uses_model_rank_processes():
        assert inference_request_sender is not None
        assert inference_response_receiver is not None
        policy = FramedPolicyClient(
            worker_index=worker_index,
            request_sender=inference_request_sender,
            response_receiver=inference_response_receiver,
            timeout_seconds=(execution_config.timeouts.round_seconds),
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
    policy = DirectPolicyClient(replica=core)
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


def _handle_worker_command(
    *,
    worker_index: int,
    run_id: str,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    runtime: _WorkerRuntime,
    command: WorkerCommand,
    telemetry_sink: TelemetrySink,
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
        return _run_worker_sampling(
            worker_index=worker_index,
            run_id=run_id,
            train_config=train_config,
            execution_config=execution_config,
            runtime=runtime,
            command=command,
            telemetry_sink=telemetry_sink,
        )
    return _run_worker_update(
        worker_index=worker_index,
        run_id=run_id,
        train_config=train_config,
        execution_config=execution_config,
        runtime=runtime,
        command=command,
        telemetry_sink=telemetry_sink,
    )


def _load_worker_state(
    *,
    worker_index: int,
    runtime: _WorkerRuntime,
    command: WorkerLoadStateCommand,
) -> WorkerResponse:
    if runtime.local_model_rank is None:
        return WorkerRejected(
            worker_index=worker_index,
            reason="worker does not own a local model rank",
        )
    load_result = runtime.local_model_rank.load_state(
        state=command.state,
        policy_version=command.policy_version,
    )
    if isinstance(load_result, Rejected):
        return WorkerRejected(
            worker_index=worker_index,
            reason=load_result.reason,
        )
    return WorkerStateLoaded(
        worker_index=worker_index,
        policy_version=command.policy_version,
    )


def _run_worker_update(
    *,
    worker_index: int,
    run_id: str,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    runtime: _WorkerRuntime,
    command: WorkerUpdateCommand,
    telemetry_sink: TelemetrySink,
) -> WorkerResponse:
    if runtime.local_model_rank is None:
        return WorkerRejected(
            worker_index=worker_index,
            reason="worker does not own a local model rank",
        )
    update_telemetry = _record_worker_stage(
        telemetry_sink=telemetry_sink,
        run_id=run_id,
        worker_index=worker_index,
        stage="update",
        total_rounds=0,
        total_updates=command.policy_version,
    )
    if isinstance(update_telemetry, Rejected):
        return WorkerRejected(
            worker_index=worker_index,
            reason=update_telemetry.reason,
        )
    commit_result = runtime.arena_reader.read_commit_for_rank(
        policy_version=command.policy_version,
        model_rank_index=worker_index,
    )
    if isinstance(commit_result, Rejected):
        return WorkerRejected(
            worker_index=worker_index,
            reason=commit_result.reason,
        )
    update_result = runtime.local_model_rank.update(
        commit=commit_result.value,
        policy_version=command.policy_version,
    )
    if isinstance(update_result, Rejected):
        return WorkerRejected(
            worker_index=worker_index,
            reason=update_result.reason,
        )
    return WorkerUpdateCompleted(
        worker_index=worker_index,
        update_stats=update_result.value.update_stats,
        state=update_result.value.state,
    )


def _run_worker_sampling(
    *,
    worker_index: int,
    run_id: str,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    runtime: _WorkerRuntime,
    command: WorkerStartSamplingCommand,
    telemetry_sink: TelemetrySink,
) -> WorkerResponse:
    telemetry_result = _record_worker_stage(
        telemetry_sink=telemetry_sink,
        run_id=run_id,
        worker_index=worker_index,
        stage="rollout",
        total_rounds=runtime.next_episode_id,
        total_updates=command.policy_version,
    )
    if isinstance(telemetry_result, Rejected):
        return WorkerRejected(
            worker_index=worker_index,
            reason=telemetry_result.reason,
        )
    sampling_result = asyncio.run(
        _run_sampling_until_full(
            worker_index=worker_index,
            train_config=train_config,
            execution_config=execution_config,
            runtime=runtime,
            command=command,
        )
    )
    if isinstance(sampling_result, Rejected):
        return WorkerRejected(
            worker_index=worker_index,
            reason=sampling_result.reason,
        )
    return WorkerSamplingStopped(
        worker_index=worker_index,
        policy_version=command.policy_version,
    )


async def _run_sampling_until_full(
    *,
    worker_index: int,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    runtime: _WorkerRuntime,
    command: WorkerStartSamplingCommand,
) -> _result.Ok[None] | _result.Rejected:
    assert command.game_env_count <= len(runtime.sessions)
    sessions = list(runtime.sessions)
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
        )
        runtime.next_episode_id += 1
        pending[task] = game_env_index
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
                runtime.arena_writer.record_cancelled_envs(
                    cancelled_count
                )
                return round_result
            env_round = round_result.value
            round_data = env_round.round_data
            if round_data.game_over:
                sessions[game_env_index] = SelfPlaySession(
                    policy=runtime.policy
                )
            append_result = runtime.arena_writer.append_round(
                policy_version=command.policy_version,
                metrics=_round_metrics(round_data),
                commit=round_data.returns,
            )
            if isinstance(append_result, Rejected):
                cancelled_count = len(pending)
                await _cancel_pending_rounds(
                    pending=pending,
                    sessions=sessions,
                    policy=runtime.policy,
                )
                runtime.arena_writer.record_cancelled_envs(
                    cancelled_count
                )
                return append_result
            if append_result.value.arena_full:
                cancelled_count = len(pending)
                await _cancel_pending_rounds(
                    pending=pending,
                    sessions=sessions,
                    policy=runtime.policy,
                )
                runtime.arena_writer.record_cancelled_envs(
                    cancelled_count
                )
                runtime.sessions = tuple(sessions)
                return Ok(value=None)
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
            )
            runtime.next_episode_id += 1
            pending[task] = game_env_index
    runtime.sessions = tuple(sessions)
    return Ok(value=None)


def _schedule_round(
    *,
    session: SelfPlaySession,
    game_env_index: int,
    episode_id: int,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    policy_version: int,
) -> asyncio.Task[_EnvRoundTaskResult]:
    return asyncio.create_task(
        _play_worker_round(
            session=session,
            game_env_index=game_env_index,
            episode_id=episode_id,
            base_seed=train_config.seed,
            policy_version=policy_version,
            max_seconds=execution_config.timeouts.round_seconds,
        )
    )


async def _play_worker_round(
    *,
    session: SelfPlaySession,
    game_env_index: int,
    episode_id: int,
    base_seed: int,
    policy_version: int,
    max_seconds: float,
) -> _EnvRoundTaskResult:
    result = await session.play_round(
        base_seed=base_seed,
        policy_version=policy_version,
        episode_id=episode_id,
        max_seconds=max_seconds,
    )
    if isinstance(result, Rejected):
        return result
    return Ok(
        value=_EnvRoundResult(
            game_env_index=game_env_index,
            episode_id=episode_id,
            round_data=result.value,
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


def _record_worker_stage(
    *,
    telemetry_sink: TelemetrySink,
    run_id: str,
    worker_index: int,
    stage: str,
    total_rounds: int,
    total_updates: int,
) -> _result.Ok[None] | _result.Rejected:
    assert stage in ("rollout", "update")
    return telemetry_sink.append(
        TelemetryEvent(
            run_id=run_id,
            process_label=f"worker-{worker_index}",
            stage=stage,
            total_rounds=total_rounds,
            total_updates=total_updates,
            progress_numerator=0,
            progress_denominator=1,
            unix_seconds=time.time(),
        )
    )


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
