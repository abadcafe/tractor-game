"""Training runtime topology hidden behind an update-cycle interface."""

from __future__ import annotations

import multiprocessing as mp
import queue
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Protocol, cast

from server import result as _result
from server.result import Ok, Rejected
from server.training.config import ModelConfig, TrainConfig
from server.training.ppo import PPOUpdateProfile, PPOUpdateStats
from server.training.runtime.config import ExecutionConfig
from server.training.runtime.distributed import (
    DistributedBackend,
    DistributedRankConfig,
)
from server.training.runtime.messages import (
    StopWorkerCommand,
    WorkerCommandReceiver,
    WorkerCommandSender,
    WorkerLoadStateCommand,
    WorkerRejected,
    WorkerResponseReceiver,
    WorkerResponseSender,
    WorkerSamplingStopped,
    WorkerStartSamplingCommand,
    WorkerStateLoaded,
    WorkerUpdateCommand,
    WorkerUpdateCompleted,
)
from server.training.runtime.model_rank import run_model_rank_process
from server.training.runtime.model_rank.inference_transport import (
    ConnectionPolicyRequestReceiver,
    ConnectionPolicyRequestSender,
    ConnectionPolicyResponseReceiver,
)
from server.training.runtime.model_rank.messages import (
    ModelRankCommandReceiver,
    ModelRankCommandSender,
    ModelRankLoadStateCommand,
    ModelRankRejected,
    ModelRankResponseReceiver,
    ModelRankResponseSender,
    ModelRankStateLoaded,
    ModelRankStopCommand,
    ModelRankUpdateCommand,
    ModelRankUpdateCompleted,
)
from server.training.runtime.rendezvous import create_file_rendezvous
from server.training.runtime.shared_rollout_arena import (
    RolloutArenaHandle,
    RolloutArenaSnapshot,
    SharedRolloutArenaGroup,
    close_shared_rollout_arenas,
    create_shared_rollout_arena_group,
    reset_rollout_arenas,
    wait_all_rollout_arenas_full,
)
from server.training.runtime.state import RuntimeTrainingState
from server.training.runtime.telemetry import (
    IntervalTelemetrySink,
    JsonlTelemetrySink,
    TelemetrySink,
)
from server.training.runtime.worker_process import (
    run_training_worker_process,
)

_GRACEFUL_PROCESS_STOP_SECONDS = 1.0
_TERMINATED_PROCESS_STOP_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class TrainingUpdateRequest:
    """One synchronized arena-backed update requested by coordinator."""

    state: RuntimeTrainingState
    policy_version: int

    def __post_init__(self) -> None:
        assert self.policy_version >= 0


@dataclass(frozen=True, slots=True)
class TrainingUpdateResult:
    """Result produced by one arena-backed training update."""

    snapshot: RolloutArenaSnapshot
    states: tuple[RuntimeTrainingState, ...]
    update_stats: PPOUpdateStats

    def __post_init__(self) -> None:
        assert self.states


class TrainingRuntime(Protocol):
    """Coordinator-facing training runtime interface."""

    def run_update(
        self, request: TrainingUpdateRequest
    ) -> _result.Ok[TrainingUpdateResult] | _result.Rejected: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _WorkerHandle:
    index: int
    command_sender: WorkerCommandSender
    process: BaseProcess


@dataclass(frozen=True, slots=True)
class _WorkerPool:
    handles: tuple[_WorkerHandle, ...]
    response_receiver: WorkerResponseReceiver


@dataclass(frozen=True, slots=True)
class _ModelRankHandle:
    index: int
    command_sender: ModelRankCommandSender
    process: BaseProcess


@dataclass(frozen=True, slots=True)
class _ModelRankPool:
    handles: tuple[_ModelRankHandle, ...]
    response_receiver: ModelRankResponseReceiver


@dataclass(frozen=True, slots=True)
class _RuntimePools:
    worker_pool: _WorkerPool
    model_rank_pool: _ModelRankPool | None
    worker_inference_links: tuple[_WorkerInferenceLink, ...]
    rollout_arena_group: SharedRolloutArenaGroup


@dataclass(frozen=True, slots=True)
class _WorkerInferenceLink:
    request_sender: Connection
    request_receiver: Connection
    response_sender: Connection
    response_receiver: Connection


@dataclass(frozen=True, slots=True)
class _DistributedUpdateGroup:
    backend: DistributedBackend
    init_method: str
    world_size: int
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class _UpdateResult:
    states: tuple[RuntimeTrainingState, ...]
    update_stats: PPOUpdateStats

    def __post_init__(self) -> None:
        assert self.states


@dataclass(slots=True)
class _ProcessTrainingRuntime:
    execution_config: ExecutionConfig
    pools: _RuntimePools

    def run_update(
        self, request: TrainingUpdateRequest
    ) -> _result.Ok[TrainingUpdateResult] | _result.Rejected:
        return _run_training_update(
            pools=self.pools,
            execution_config=self.execution_config,
            state=request.state,
            policy_version=request.policy_version,
        )

    def close(self) -> None:
        _stop_runtime_pools(self.pools)


def open_training_runtime(
    *,
    run_dir: Path,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
) -> _result.Ok[TrainingRuntime] | _result.Rejected:
    """Start worker/model-rank processes for synchronized training."""
    pools_result = _start_runtime_pools(
        run_dir=run_dir,
        run_id=run_id,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
    )
    if isinstance(pools_result, Rejected):
        return pools_result
    return Ok(
        value=_ProcessTrainingRuntime(
            execution_config=execution_config,
            pools=pools_result.value,
        )
    )


def _start_runtime_pools(
    *,
    run_dir: Path,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
) -> _result.Ok[_RuntimePools] | _result.Rejected:
    context: SpawnContext = mp.get_context("spawn")
    distributed_group_result = _distributed_update_group(
        run_dir=run_dir,
        execution_config=execution_config,
    )
    if isinstance(distributed_group_result, Rejected):
        return distributed_group_result
    distributed_group = distributed_group_result.value
    arena_group_result = create_shared_rollout_arena_group(
        context=context,
        worker_count=execution_config.worker_process_count(),
        samples_per_worker_update=(
            execution_config.samples_per_worker_update
        ),
    )
    if isinstance(arena_group_result, Rejected):
        return arena_group_result
    arena_group = arena_group_result.value
    worker_response_queue = context.Queue()
    worker_response_receiver = cast(
        WorkerResponseReceiver, worker_response_queue
    )
    worker_response_sender = cast(
        WorkerResponseSender, worker_response_queue
    )
    worker_inference_links = _worker_inference_links(
        context=context,
        worker_count=execution_config.worker_process_count(),
    )
    model_rank_pool_result = _start_model_rank_pool(
        context=context,
        run_dir=run_dir,
        run_id=run_id,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
        distributed_group=distributed_group,
        rank_inference_request_receivers=(
            _rank_inference_request_receivers(
                execution_config=execution_config,
                worker_inference_links=worker_inference_links,
            )
        ),
        worker_inference_response_senders=tuple(
            link.response_sender for link in worker_inference_links
        ),
        rollout_arena_handles=arena_group.handles,
    )
    if isinstance(model_rank_pool_result, Rejected):
        _close_worker_inference_links(worker_inference_links)
        close_shared_rollout_arenas(arena_group)
        return model_rank_pool_result
    model_rank_pool = model_rank_pool_result.value
    worker_handles: list[_WorkerHandle] = []
    for index in range(execution_config.worker_process_count()):
        command_queue = context.Queue()
        command_sender = cast(WorkerCommandSender, command_queue)
        command_receiver = cast(WorkerCommandReceiver, command_queue)
        inference_request_sender = (
            None
            if model_rank_pool is None
            else ConnectionPolicyRequestSender(
                connection=worker_inference_links[index].request_sender,
            )
        )
        inference_response_receiver = (
            None
            if model_rank_pool is None
            else ConnectionPolicyResponseReceiver(
                worker_inference_links[index].response_receiver
            )
        )
        process = context.Process(
            target=run_training_worker_process,
            kwargs={
                "worker_index": index,
                "run_id": run_id,
                "model_config": model_config,
                "train_config": train_config,
                "execution_config": execution_config,
                "worker_cpus": execution_config.worker_cpu_set(index),
                "command_receiver": command_receiver,
                "response_sender": worker_response_sender,
                "telemetry_sink": _telemetry_sink(
                    run_dir=run_dir,
                    execution_config=execution_config,
                ),
                "inference_request_sender": inference_request_sender,
                "inference_response_receiver": (
                    inference_response_receiver
                ),
                "rollout_arena_handles": arena_group.handles,
                "distributed_rank_config": (
                    _worker_distributed_rank_config(
                        execution_config=execution_config,
                        group=distributed_group,
                        worker_index=index,
                    )
                ),
            },
        )
        process.start()
        worker_handles.append(
            _WorkerHandle(
                index=index,
                command_sender=command_sender,
                process=process,
            )
        )
    return Ok(
        value=_RuntimePools(
            worker_pool=_WorkerPool(
                handles=tuple(worker_handles),
                response_receiver=worker_response_receiver,
            ),
            model_rank_pool=model_rank_pool,
            worker_inference_links=worker_inference_links,
            rollout_arena_group=arena_group,
        )
    )


def _start_model_rank_pool(
    *,
    context: SpawnContext,
    run_dir: Path,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    distributed_group: _DistributedUpdateGroup | None,
    rank_inference_request_receivers: tuple[
        tuple[ConnectionPolicyRequestReceiver, ...], ...
    ],
    worker_inference_response_senders: tuple[Connection, ...],
    rollout_arena_handles: tuple[RolloutArenaHandle, ...],
) -> _result.Ok[_ModelRankPool | None] | _result.Rejected:
    if not execution_config.uses_model_rank_processes():
        return Ok(value=None)
    response_queue = context.Queue()
    response_receiver = cast(ModelRankResponseReceiver, response_queue)
    response_sender = cast(ModelRankResponseSender, response_queue)
    handles: list[_ModelRankHandle] = []
    for index, model_rank_device in enumerate(
        execution_config.model_ranks.devices
    ):
        command_receiver_raw, command_sender_raw = context.Pipe(
            duplex=False
        )
        command_sender = cast(
            ModelRankCommandSender, command_sender_raw
        )
        command_receiver = cast(
            ModelRankCommandReceiver, command_receiver_raw
        )
        process = context.Process(
            target=run_model_rank_process,
            kwargs={
                "model_rank_index": index,
                "model_rank_device": model_rank_device,
                "run_id": run_id,
                "model_config": model_config,
                "train_config": train_config,
                "execution_config": execution_config,
                "command_receiver": command_receiver,
                "response_sender": response_sender,
                "inference_request_receivers": (
                    rank_inference_request_receivers[index]
                ),
                "inference_response_senders": (
                    worker_inference_response_senders
                ),
                "rollout_arena_handles": rollout_arena_handles,
                "telemetry_sink": _telemetry_sink(
                    run_dir=run_dir,
                    execution_config=execution_config,
                ),
                "distributed_rank_config": (
                    _model_rank_distributed_rank_config(
                        group=distributed_group,
                        model_rank_index=index,
                    )
                ),
            },
        )
        process.start()
        handles.append(
            _ModelRankHandle(
                index=index,
                command_sender=command_sender,
                process=process,
            )
        )
    return Ok(
        value=_ModelRankPool(
            handles=tuple(handles),
            response_receiver=response_receiver,
        )
    )


def _worker_inference_links(
    *,
    context: SpawnContext,
    worker_count: int,
) -> tuple[_WorkerInferenceLink, ...]:
    assert worker_count > 0
    links: list[_WorkerInferenceLink] = []
    for _ in range(worker_count):
        request_receiver, request_sender = context.Pipe(duplex=False)
        response_receiver, response_sender = context.Pipe(duplex=False)
        links.append(
            _WorkerInferenceLink(
                request_sender=request_sender,
                request_receiver=request_receiver,
                response_sender=response_sender,
                response_receiver=response_receiver,
            )
        )
    return tuple(links)


def _rank_inference_request_receivers(
    *,
    execution_config: ExecutionConfig,
    worker_inference_links: tuple[_WorkerInferenceLink, ...],
) -> tuple[tuple[ConnectionPolicyRequestReceiver, ...], ...]:
    if not execution_config.uses_model_rank_processes():
        return ()
    groups: list[list[ConnectionPolicyRequestReceiver]] = [
        [] for _ in range(execution_config.model_rank_process_count())
    ]
    for worker_index, link in enumerate(worker_inference_links):
        model_rank_index = execution_config.model_rank_index_for_worker(
            worker_index
        )
        groups[model_rank_index].append(
            ConnectionPolicyRequestReceiver(
                connection=link.request_receiver
            )
        )
    return tuple(tuple(group) for group in groups)


def _distributed_update_group(
    *,
    run_dir: Path,
    execution_config: ExecutionConfig,
) -> _result.Ok[_DistributedUpdateGroup | None] | _result.Rejected:
    if execution_config.uses_model_rank_processes():
        world_size = execution_config.model_rank_process_count()
        if world_size <= 1:
            return Ok(value=None)
        if execution_config.model_ranks.kind != "cuda":
            return Rejected(
                reason="multi-model-rank training requires CUDA NCCL"
            )
        init_result = create_file_rendezvous(run_dir)
        if isinstance(init_result, Rejected):
            return init_result
        return Ok(
            value=_DistributedUpdateGroup(
                backend="nccl",
                init_method=init_result.value.init_method,
                world_size=world_size,
                timeout_seconds=(
                    execution_config.timeouts.update_seconds
                ),
            )
        )
    world_size = execution_config.worker_process_count()
    if world_size <= 1:
        return Ok(value=None)
    init_result = create_file_rendezvous(run_dir)
    if isinstance(init_result, Rejected):
        return init_result
    return Ok(
        value=_DistributedUpdateGroup(
            backend="gloo",
            init_method=init_result.value.init_method,
            world_size=world_size,
            timeout_seconds=execution_config.timeouts.update_seconds,
        )
    )


def _telemetry_sink(
    *, run_dir: Path, execution_config: ExecutionConfig
) -> TelemetrySink:
    return IntervalTelemetrySink(
        sink=JsonlTelemetrySink(run_dir),
        min_interval_seconds=(
            execution_config.telemetry_interval_seconds
        ),
    )


def _worker_distributed_rank_config(
    *,
    execution_config: ExecutionConfig,
    group: _DistributedUpdateGroup | None,
    worker_index: int,
) -> DistributedRankConfig | None:
    if group is None or execution_config.uses_model_rank_processes():
        return None
    return _rank_config(group=group, rank=worker_index)


def _model_rank_distributed_rank_config(
    *,
    group: _DistributedUpdateGroup | None,
    model_rank_index: int,
) -> DistributedRankConfig | None:
    if group is None:
        return None
    return _rank_config(group=group, rank=model_rank_index)


def _rank_config(
    *,
    group: _DistributedUpdateGroup,
    rank: int,
) -> DistributedRankConfig:
    return DistributedRankConfig(
        backend=group.backend,
        init_method=group.init_method,
        rank=rank,
        world_size=group.world_size,
        timeout_seconds=group.timeout_seconds,
    )


def _run_training_update(
    *,
    pools: _RuntimePools,
    execution_config: ExecutionConfig,
    state: RuntimeTrainingState,
    policy_version: int,
) -> _result.Ok[TrainingUpdateResult] | _result.Rejected:
    prepare_result = _sync_compute_rank_states(
        worker_pool=pools.worker_pool,
        model_rank_pool=pools.model_rank_pool,
        state=state,
        policy_version=policy_version,
        state_sync_timeout_seconds=(
            execution_config.timeouts.state_sync_seconds
        ),
    )
    if isinstance(prepare_result, Rejected):
        return prepare_result
    reset_result = reset_rollout_arenas(
        group=pools.rollout_arena_group,
        policy_version=policy_version,
    )
    if isinstance(reset_result, Rejected):
        return reset_result
    start_result = _start_worker_sampling(
        worker_pool=pools.worker_pool,
        execution_config=execution_config,
        policy_version=policy_version,
    )
    if isinstance(start_result, Rejected):
        return start_result
    snapshot_result = wait_all_rollout_arenas_full(
        group=pools.rollout_arena_group,
        policy_version=policy_version,
        timeout_seconds=execution_config.timeouts.rollout_response_seconds,
    )
    if isinstance(snapshot_result, Rejected):
        return snapshot_result
    stopped_result = _receive_worker_sampling_stopped(
        receiver=pools.worker_pool.response_receiver,
        expected_count=len(pools.worker_pool.handles),
        policy_version=policy_version,
        timeout_seconds=execution_config.timeouts.rollout_response_seconds,
    )
    if isinstance(stopped_result, Rejected):
        return stopped_result
    update_result = _run_compute_updates(
        worker_pool=pools.worker_pool,
        model_rank_pool=pools.model_rank_pool,
        policy_version=policy_version,
        update_timeout_seconds=execution_config.timeouts.update_seconds,
    )
    if isinstance(update_result, Rejected):
        return update_result
    return Ok(
        value=TrainingUpdateResult(
            snapshot=snapshot_result.value,
            states=update_result.value.states,
            update_stats=update_result.value.update_stats,
        )
    )


def _sync_compute_rank_states(
    *,
    worker_pool: _WorkerPool,
    model_rank_pool: _ModelRankPool | None,
    state: RuntimeTrainingState,
    policy_version: int,
    state_sync_timeout_seconds: float,
) -> _result.Ok[None] | _result.Rejected:
    if model_rank_pool is None:
        return _sync_worker_states(
            worker_pool=worker_pool,
            state=state,
            policy_version=policy_version,
            state_sync_timeout_seconds=state_sync_timeout_seconds,
        )
    return _sync_model_rank_states(
        model_rank_pool=model_rank_pool,
        state=state,
        policy_version=policy_version,
        state_sync_timeout_seconds=state_sync_timeout_seconds,
    )


def _start_worker_sampling(
    *,
    worker_pool: _WorkerPool,
    execution_config: ExecutionConfig,
    policy_version: int,
) -> _result.Ok[None] | _result.Rejected:
    assert worker_pool.handles
    for handle in worker_pool.handles:
        handle.command_sender.put(
            WorkerStartSamplingCommand(
                policy_version=policy_version,
                game_env_count=execution_config.game_envs_per_worker,
            )
        )
    return Ok(value=None)


def _run_compute_updates(
    *,
    worker_pool: _WorkerPool,
    model_rank_pool: _ModelRankPool | None,
    policy_version: int,
    update_timeout_seconds: float,
) -> _result.Ok[_UpdateResult] | _result.Rejected:
    if model_rank_pool is not None:
        return _run_model_rank_updates(
            model_rank_pool=model_rank_pool,
            policy_version=policy_version,
            update_timeout_seconds=update_timeout_seconds,
        )
    return _run_worker_updates(
        worker_pool=worker_pool,
        policy_version=policy_version,
        update_timeout_seconds=update_timeout_seconds,
    )


def _run_worker_updates(
    *,
    worker_pool: _WorkerPool,
    policy_version: int,
    update_timeout_seconds: float,
) -> _result.Ok[_UpdateResult] | _result.Rejected:
    _send_worker_update_commands(
        worker_pool=worker_pool,
        policy_version=policy_version,
    )
    responses_result = _receive_worker_updates(
        receiver=worker_pool.response_receiver,
        expected_count=len(worker_pool.handles),
        update_timeout_seconds=update_timeout_seconds,
        unexpected_sampling_reason=(
            "worker returned sampling during synchronized update"
        ),
    )
    if isinstance(responses_result, Rejected):
        return responses_result
    ordered = responses_result.value
    update_stats = tuple(response.update_stats for response in ordered)
    return Ok(
        value=_UpdateResult(
            states=tuple(response.state for response in ordered),
            update_stats=_aggregate_ppo_update_stats(update_stats),
        )
    )


def _send_worker_update_commands(
    *,
    worker_pool: _WorkerPool,
    policy_version: int,
) -> None:
    for handle in worker_pool.handles:
        handle.command_sender.put(
            WorkerUpdateCommand(
                policy_version=policy_version,
            )
        )


def _receive_worker_sampling_stopped(
    *,
    receiver: WorkerResponseReceiver,
    expected_count: int,
    policy_version: int,
    timeout_seconds: float,
) -> _result.Ok[tuple[WorkerSamplingStopped, ...]] | _result.Rejected:
    responses: list[WorkerSamplingStopped] = []
    for _ in range(expected_count):
        try:
            response = receiver.get(True, timeout_seconds)
        except queue.Empty:
            return Rejected(reason="training worker response timed out")
        if isinstance(response, WorkerRejected):
            reason = (
                f"worker-{response.worker_index}: {response.reason}"
            )
            return Rejected(reason=reason)
        if isinstance(response, WorkerUpdateCompleted):
            return Rejected(
                reason="worker returned update during sampling"
            )
        if isinstance(response, WorkerStateLoaded):
            return Rejected(
                reason="worker returned state sync during sampling"
            )
        if response.policy_version != policy_version:
            return Rejected(
                reason="worker returned stale sampling policy version"
            )
        responses.append(response)
    return Ok(
        value=tuple(
            sorted(responses, key=lambda item: item.worker_index)
        )
    )


def _receive_worker_updates(
    *,
    receiver: WorkerResponseReceiver,
    expected_count: int,
    update_timeout_seconds: float,
    unexpected_sampling_reason: str,
) -> _result.Ok[tuple[WorkerUpdateCompleted, ...]] | _result.Rejected:
    responses: list[WorkerUpdateCompleted] = []
    for _ in range(expected_count):
        try:
            response = receiver.get(True, update_timeout_seconds)
        except queue.Empty:
            return Rejected(reason="training worker response timed out")
        if isinstance(response, WorkerRejected):
            reason = (
                f"worker-{response.worker_index}: {response.reason}"
            )
            return Rejected(reason=reason)
        if isinstance(response, WorkerSamplingStopped):
            return Rejected(reason=unexpected_sampling_reason)
        if isinstance(response, WorkerStateLoaded):
            return Rejected(
                reason=(
                    "worker returned state sync during "
                    "synchronized update"
                )
            )
        responses.append(response)
    return Ok(
        value=tuple(
            sorted(responses, key=lambda item: item.worker_index)
        )
    )


def _sync_worker_states(
    *,
    worker_pool: _WorkerPool,
    state: RuntimeTrainingState,
    policy_version: int,
    state_sync_timeout_seconds: float,
) -> _result.Ok[None] | _result.Rejected:
    expected_indices = {handle.index for handle in worker_pool.handles}
    for handle in worker_pool.handles:
        handle.command_sender.put(
            WorkerLoadStateCommand(
                state=state,
                policy_version=policy_version,
            )
        )
    loaded_indices: set[int] = set()
    for _ in worker_pool.handles:
        try:
            response = worker_pool.response_receiver.get(
                True,
                state_sync_timeout_seconds,
            )
        except queue.Empty:
            return Rejected(reason="worker state sync timed out")
        if isinstance(response, WorkerRejected):
            return Rejected(
                reason=(
                    f"worker-{response.worker_index}: {response.reason}"
                )
            )
        if isinstance(response, WorkerSamplingStopped):
            return Rejected(
                reason="worker returned sampling during state sync"
            )
        if isinstance(response, WorkerUpdateCompleted):
            return Rejected(
                reason="worker returned update during state sync"
            )
        if response.policy_version != policy_version:
            return Rejected(
                reason="worker state sync returned stale policy version"
            )
        loaded_indices.add(response.worker_index)
    if loaded_indices != expected_indices:
        return Rejected(reason="worker state sync rank set mismatch")
    return Ok(value=None)


def _sync_model_rank_states(
    *,
    model_rank_pool: _ModelRankPool,
    state: RuntimeTrainingState,
    policy_version: int,
    state_sync_timeout_seconds: float,
) -> _result.Ok[None] | _result.Rejected:
    expected_indices = {
        handle.index for handle in model_rank_pool.handles
    }
    for handle in model_rank_pool.handles:
        handle.command_sender.send(
            ModelRankLoadStateCommand(
                state=state,
                policy_version=policy_version,
            )
        )
    loaded: list[ModelRankStateLoaded] = []
    for _ in model_rank_pool.handles:
        try:
            response = model_rank_pool.response_receiver.get(
                True,
                state_sync_timeout_seconds,
            )
        except queue.Empty:
            return Rejected(reason="model-rank state sync timed out")
        if isinstance(response, ModelRankRejected):
            return Rejected(
                reason=(
                    f"model-rank-{response.model_rank_index}: "
                    f"{response.reason}"
                )
            )
        if isinstance(response, ModelRankUpdateCompleted):
            return Rejected(
                reason="model rank returned update during state sync"
            )
        loaded.append(response)
    loaded_indices = {item.model_rank_index for item in loaded}
    if loaded_indices != expected_indices:
        return Rejected(
            reason="model-rank state sync rank set mismatch"
        )
    if any(item.policy_version != policy_version for item in loaded):
        return Rejected(
            reason="model-rank state sync returned stale policy version"
        )
    return Ok(value=None)


def _run_model_rank_updates(
    *,
    model_rank_pool: _ModelRankPool,
    policy_version: int,
    update_timeout_seconds: float,
) -> _result.Ok[_UpdateResult] | _result.Rejected:
    _send_model_rank_update_commands(
        model_rank_pool=model_rank_pool,
        policy_version=policy_version,
    )
    responses: list[ModelRankUpdateCompleted] = []
    for _ in model_rank_pool.handles:
        try:
            response = model_rank_pool.response_receiver.get(
                True,
                update_timeout_seconds,
            )
        except queue.Empty:
            return Rejected(reason="model-rank update timed out")
        if isinstance(response, ModelRankRejected):
            return Rejected(
                reason=(
                    f"model-rank-{response.model_rank_index}: "
                    f"{response.reason}"
                )
            )
        if isinstance(response, ModelRankStateLoaded):
            return Rejected(
                reason="model rank returned state sync during update"
            )
        responses.append(response)
    ordered = tuple(sorted(responses, key=lambda item: item.rank_index))
    update_stats = tuple(response.update_stats for response in ordered)
    return Ok(
        value=_UpdateResult(
            states=tuple(response.state for response in ordered),
            update_stats=_aggregate_ppo_update_stats(update_stats),
        )
    )


def _send_model_rank_update_commands(
    *,
    model_rank_pool: _ModelRankPool,
    policy_version: int,
) -> None:
    for handle in model_rank_pool.handles:
        handle.command_sender.send(
            ModelRankUpdateCommand(
                policy_version=policy_version,
            )
        )


def _stop_runtime_pools(pools: _RuntimePools) -> None:
    _stop_worker_pool(pools.worker_pool)
    if pools.model_rank_pool is not None:
        _stop_model_rank_pool(pools.model_rank_pool)
    _close_worker_inference_links(pools.worker_inference_links)
    close_shared_rollout_arenas(pools.rollout_arena_group)


def _stop_worker_pool(pool: _WorkerPool) -> None:
    for handle in pool.handles:
        handle.command_sender.put(StopWorkerCommand(reason="complete"))
    for handle in pool.handles:
        _stop_process(handle.process)


def _stop_model_rank_pool(pool: _ModelRankPool) -> None:
    for handle in pool.handles:
        handle.command_sender.send(
            ModelRankStopCommand(reason="complete")
        )
    for handle in pool.handles:
        _stop_process(handle.process)


def _stop_process(process: BaseProcess) -> None:
    process.join(timeout=_GRACEFUL_PROCESS_STOP_SECONDS)
    if not process.is_alive():
        return
    process.terminate()
    process.join(timeout=_TERMINATED_PROCESS_STOP_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(timeout=_TERMINATED_PROCESS_STOP_SECONDS)


def _close_worker_inference_links(
    links: tuple[_WorkerInferenceLink, ...],
) -> None:
    for link in links:
        link.request_sender.close()
        link.request_receiver.close()
        link.response_sender.close()
        link.response_receiver.close()


def _aggregate_ppo_update_stats(
    stats: tuple[PPOUpdateStats, ...],
) -> PPOUpdateStats:
    assert stats
    return PPOUpdateStats(
        policy_loss=_mean(tuple(item.policy_loss for item in stats)),
        value_loss=_mean(tuple(item.value_loss for item in stats)),
        entropy=_mean(tuple(item.entropy for item in stats)),
        total_loss=_mean(tuple(item.total_loss for item in stats)),
        approx_kl=_mean(tuple(item.approx_kl for item in stats)),
        clip_fraction=_mean(
            tuple(item.clip_fraction for item in stats)
        ),
        profile=_aggregate_ppo_update_profiles(
            tuple(item.profile for item in stats)
        ),
    )


def _aggregate_ppo_update_profiles(
    profiles: tuple[PPOUpdateProfile, ...],
) -> PPOUpdateProfile:
    assert profiles
    update_seconds = max(profile.update_seconds for profile in profiles)
    argument_decode_seconds = max(
        profile.argument_decode_seconds for profile in profiles
    )
    decode_fraction = (
        0.0
        if update_seconds <= 0.0
        else argument_decode_seconds / update_seconds
    )
    return PPOUpdateProfile(
        update_seconds=update_seconds,
        minibatch_loss_seconds=max(
            profile.minibatch_loss_seconds for profile in profiles
        ),
        observation_batch_seconds=max(
            profile.observation_batch_seconds for profile in profiles
        ),
        observation_encode_seconds=max(
            profile.observation_encode_seconds for profile in profiles
        ),
        value_head_seconds=max(
            profile.value_head_seconds for profile in profiles
        ),
        argument_select_seconds=max(
            profile.argument_select_seconds for profile in profiles
        ),
        argument_prefix_tensorize_seconds=max(
            profile.argument_prefix_tensorize_seconds
            for profile in profiles
        ),
        argument_decode_seconds=argument_decode_seconds,
        argument_distribution_seconds=max(
            profile.argument_distribution_seconds
            for profile in profiles
        ),
        backward_seconds=max(
            profile.backward_seconds for profile in profiles
        ),
        optimizer_step_seconds=max(
            profile.optimizer_step_seconds for profile in profiles
        ),
        argument_decode_fraction=decode_fraction,
        argument_prefix_batch_count=sum(
            profile.argument_prefix_batch_count for profile in profiles
        ),
        argument_prefix_row_count=sum(
            profile.argument_prefix_row_count for profile in profiles
        ),
        argument_prefix_token_count=sum(
            profile.argument_prefix_token_count for profile in profiles
        ),
        argument_prefix_valid_token_count=sum(
            profile.argument_prefix_valid_token_count
            for profile in profiles
        ),
        argument_prefix_padding_token_count=sum(
            profile.argument_prefix_padding_token_count
            for profile in profiles
        ),
    )


def _mean(values: tuple[float, ...]) -> float:
    assert values
    return sum(values) / len(values)
