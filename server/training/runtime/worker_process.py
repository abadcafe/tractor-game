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
    WorkerRolloutCommand,
    WorkerRolloutCompleted,
    WorkerRoundSummary,
    WorkerStateLoaded,
    WorkerUpdateCommand,
    WorkerUpdateCompleted,
)
from server.training.runtime.model_rank import (
    DirectPolicyClient,
    FramedPolicyClient,
    InlineModelRank,
    create_model_replica,
)
from server.training.runtime.model_rank.inference_transport import (
    ConnectionPolicyRequestSender,
    ConnectionPolicyResponseReceiver,
)
from server.training.runtime.telemetry import (
    TelemetryEvent,
    TelemetrySink,
)
from server.training.runtime.threads import apply_torch_thread_config


@dataclass(slots=True)
class _WorkerRuntime:
    policy: TrainingPolicy
    inline_model_rank: InlineModelRank | None
    session: SelfPlaySession


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
    distributed_rank_config: DistributedRankConfig | None,
) -> _result.Ok[_WorkerRuntime] | _result.Rejected:
    if execution_config.uses_model_rank_processes():
        if (
            inference_request_sender is None
            or inference_response_receiver is None
        ):
            return Rejected(
                reason="model-rank worker is missing inference queues"
            )
        policy = FramedPolicyClient(
            worker_index=worker_index,
            request_sender=inference_request_sender,
            response_receiver=inference_response_receiver,
            timeout_seconds=(execution_config.timeouts.round_seconds),
        )
        return Ok(
            value=_WorkerRuntime(
                policy=policy,
                inline_model_rank=None,
                session=SelfPlaySession(policy=policy),
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
    inline_model_rank = InlineModelRank(replica=core)
    policy = DirectPolicyClient(replica=core)
    return Ok(
        value=_WorkerRuntime(
            policy=policy,
            inline_model_rank=inline_model_rank,
            session=SelfPlaySession(policy=policy),
        )
    )


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
    if isinstance(command, WorkerRolloutCommand):
        return _run_worker_rollout(
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
    if runtime.inline_model_rank is None:
        return WorkerRejected(
            worker_index=worker_index,
            reason="worker does not own an inline model rank",
        )
    load_result = runtime.inline_model_rank.load_state(
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
    if runtime.inline_model_rank is None:
        return WorkerRejected(
            worker_index=worker_index,
            reason="worker does not own an inline model rank",
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
    update_result = runtime.inline_model_rank.update(
        shard=command.shard,
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


def _run_worker_rollout(
    *,
    worker_index: int,
    run_id: str,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    runtime: _WorkerRuntime,
    command: WorkerRolloutCommand,
    telemetry_sink: TelemetrySink,
) -> WorkerResponse:
    telemetry_result = _record_worker_stage(
        telemetry_sink=telemetry_sink,
        run_id=run_id,
        worker_index=worker_index,
        stage="rollout",
        total_rounds=command.episode_id,
        total_updates=command.policy_version,
    )
    if isinstance(telemetry_result, Rejected):
        return WorkerRejected(
            worker_index=worker_index,
            reason=telemetry_result.reason,
        )
    round_result = _play_worker_round(
        session=runtime.session,
        train_config=train_config,
        execution_config=execution_config,
        policy_version=command.policy_version,
        episode_id=command.episode_id,
    )
    if isinstance(round_result, Rejected):
        return WorkerRejected(
            worker_index=worker_index,
            reason=round_result.reason,
        )
    round_data = round_result.value
    if round_data.game_over:
        runtime.session = SelfPlaySession(policy=runtime.policy)
    return WorkerRolloutCompleted(
        worker_index=worker_index,
        episode_id=command.episode_id,
        summary=_round_summary(round_data),
        rollout_commit=round_data.rollout,
    )


def _play_worker_round(
    *,
    session: SelfPlaySession,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    policy_version: int,
    episode_id: int,
) -> _result.Ok[TrainingRoundResult] | _result.Rejected:
    return asyncio.run(
        session.play_round(
            base_seed=train_config.seed,
            policy_version=policy_version,
            episode_id=episode_id,
            max_seconds=execution_config.timeouts.round_seconds,
        )
    )


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


def _round_summary(
    round_data: TrainingRoundResult,
) -> WorkerRoundSummary:
    return WorkerRoundSummary(
        team0_reward=round_data.team0_reward,
        team1_reward=round_data.team1_reward,
        generated_action_count=round_data.generated_action_count,
        accepted_action_count=round_data.accepted_action_count,
        action_choice_count=round_data.action_choice_count,
        decision_count=round_data.rollout.transition_count(),
        elapsed_seconds=round_data.elapsed_seconds,
        game_over=round_data.game_over,
    )
