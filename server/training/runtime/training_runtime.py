"""Training runtime topology hidden behind an update-cycle interface."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.context import SpawnContext
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Protocol

from server import result as _result
from server.result import Ok, Rejected
from server.training.config import ModelConfig, TrainConfig
from server.training.ppo import PPOUpdateProfile, PPOUpdateStats
from server.training.runtime.async_ipc import (
    AsyncCoordinatorControlEndpoint,
    ProcessControlProtocol,
    create_async_process_control_link,
    create_async_socket_pair,
    wait_async_control_responses,
)
from server.training.runtime.config import ExecutionConfig
from server.training.runtime.distributed import (
    DistributedBackend,
    DistributedRankConfig,
)
from server.training.runtime.messages import (
    StopWorkerCommand,
    WorkerCommand,
    WorkerCommandRejected,
    WorkerLoadStateCommand,
    WorkerResponse,
    WorkerSamplingAlreadyStopped,
    WorkerSamplingStarted,
    WorkerSamplingStopped,
    WorkerSnapshotCommand,
    WorkerSnapshotCompleted,
    WorkerStateLoaded,
    WorkerUpdateCommand,
    WorkerUpdateCompleted,
    decode_worker_command,
    decode_worker_response,
)
from server.training.runtime.model_rank import run_model_rank_process
from server.training.runtime.model_rank.inference_transport import (
    AsyncPolicyPeer,
)
from server.training.runtime.model_rank.messages import (
    ModelRankCommand,
    ModelRankLoadStateCommand,
    ModelRankRejected,
    ModelRankResponse,
    ModelRankSnapshotCommand,
    ModelRankSnapshotCompleted,
    ModelRankStateLoaded,
    ModelRankStopCommand,
    ModelRankUpdateCommand,
    ModelRankUpdateCompleted,
    decode_model_rank_command,
    decode_model_rank_response,
)
from server.training.runtime.process_signals import (
    start_child_process_ignoring_terminal_interrupt,
)
from server.training.runtime.rendezvous import create_file_rendezvous
from server.training.runtime.shared_rollout_arena import (
    RolloutArenaHandle,
    RolloutArenaSnapshot,
    SharedRolloutArenaGroup,
    close_shared_rollout_arenas,
    create_shared_rollout_arena_group,
    reset_rollout_arenas,
    snapshot_rollout_arenas,
    wait_rollout_sample_target,
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
from server.training.runtime.worker_sampling_lifecycle import (
    reject_after_sampling_cleanup,
    start_worker_sampling_session,
    stop_worker_sampling_session,
)

_GRACEFUL_PROCESS_STOP_SECONDS = 1.0
_TERMINATED_PROCESS_STOP_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class TrainingUpdateResult:
    """Result produced by one arena-backed training update."""

    snapshot: RolloutArenaSnapshot
    update_stats: PPOUpdateStats


class TrainingRuntime(Protocol):
    """Coordinator-facing training runtime interface."""

    async def load_state(
        self, *, state: RuntimeTrainingState, policy_version: int
    ) -> _result.Ok[None] | _result.Rejected: ...

    async def run_update(
        self, *, policy_version: int
    ) -> _result.Ok[TrainingUpdateResult] | _result.Rejected: ...

    async def snapshot(
        self, *, policy_version: int
    ) -> _result.Ok[RuntimeTrainingState] | _result.Rejected: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _WorkerHandle:
    index: int
    control: AsyncCoordinatorControlEndpoint[
        WorkerCommand, WorkerResponse
    ]
    process: BaseProcess


@dataclass(frozen=True, slots=True)
class _WorkerPool:
    handles: tuple[_WorkerHandle, ...]


@dataclass(frozen=True, slots=True)
class _ModelRankHandle:
    index: int
    control: AsyncCoordinatorControlEndpoint[
        ModelRankCommand, ModelRankResponse
    ]
    process: BaseProcess


@dataclass(frozen=True, slots=True)
class _ModelRankPool:
    handles: tuple[_ModelRankHandle, ...]


@dataclass(frozen=True, slots=True)
class _RuntimePools:
    worker_pool: _WorkerPool
    model_rank_pool: _ModelRankPool | None
    worker_inference_links: tuple[_WorkerInferenceLink, ...]
    rollout_arena_group: SharedRolloutArenaGroup


@dataclass(frozen=True, slots=True)
class _WorkerInferenceLink:
    worker_peer: AsyncPolicyPeer
    model_rank_peer: AsyncPolicyPeer


@dataclass(frozen=True, slots=True)
class _DistributedUpdateGroup:
    backend: DistributedBackend
    init_method: str
    world_size: int
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class _UpdateResult:
    update_stats: PPOUpdateStats


_WORKER_CONTROL_PROTOCOL: ProcessControlProtocol[
    WorkerCommand, WorkerResponse
] = ProcessControlProtocol(
    name="worker",
    decode_command=decode_worker_command,
    decode_response=decode_worker_response,
)
_MODEL_RANK_CONTROL_PROTOCOL: ProcessControlProtocol[
    ModelRankCommand, ModelRankResponse
] = ProcessControlProtocol(
    name="model-rank",
    decode_command=decode_model_rank_command,
    decode_response=decode_model_rank_response,
)


@dataclass(slots=True)
class _ProcessTrainingRuntime:
    execution_config: ExecutionConfig
    pools: _RuntimePools

    async def load_state(
        self, *, state: RuntimeTrainingState, policy_version: int
    ) -> _result.Ok[None] | _result.Rejected:
        return await _sync_compute_rank_states(
            worker_pool=self.pools.worker_pool,
            model_rank_pool=self.pools.model_rank_pool,
            state=state,
            policy_version=policy_version,
            state_sync_timeout_seconds=(
                self.execution_config.timeouts.state_sync_seconds
            ),
        )

    async def run_update(
        self, *, policy_version: int
    ) -> _result.Ok[TrainingUpdateResult] | _result.Rejected:
        return await _run_training_update(
            pools=self.pools,
            execution_config=self.execution_config,
            policy_version=policy_version,
        )

    async def snapshot(
        self, *, policy_version: int
    ) -> _result.Ok[RuntimeTrainingState] | _result.Rejected:
        return await _snapshot_compute_rank_state(
            worker_pool=self.pools.worker_pool,
            model_rank_pool=self.pools.model_rank_pool,
            policy_version=policy_version,
            snapshot_timeout_seconds=(
                self.execution_config.timeouts.state_sync_seconds
            ),
        )

    async def close(self) -> None:
        await _stop_runtime_pools(self.pools)


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
        samples_per_update=execution_config.samples_per_update,
        slack_sample_count=_rollout_arena_slack_sample_count(
            execution_config
        ),
    )
    if isinstance(arena_group_result, Rejected):
        return arena_group_result
    arena_group = arena_group_result.value
    worker_inference_links = _worker_inference_links(
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
        rank_inference_peers=(
            _rank_inference_peers(
                execution_config=execution_config,
                worker_inference_links=worker_inference_links,
            )
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
        control_link = create_async_process_control_link(
            protocol=_WORKER_CONTROL_PROTOCOL,
        )
        inference_peer = (
            None
            if model_rank_pool is None
            else worker_inference_links[index].worker_peer
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
                "control": control_link.child,
                "telemetry_sink": _telemetry_sink(
                    run_dir=run_dir,
                    execution_config=execution_config,
                ),
                "inference_peer": inference_peer,
                "rollout_arena_handle": arena_group.handles[index],
                "distributed_rank_config": (
                    _worker_distributed_rank_config(
                        execution_config=execution_config,
                        group=distributed_group,
                        worker_index=index,
                    )
                ),
            },
        )
        start_child_process_ignoring_terminal_interrupt(process)
        control_link.child.close()
        worker_handles.append(
            _WorkerHandle(
                index=index,
                control=control_link.coordinator,
                process=process,
            )
        )
    return Ok(
        value=_RuntimePools(
            worker_pool=_WorkerPool(
                handles=tuple(worker_handles),
            ),
            model_rank_pool=model_rank_pool,
            worker_inference_links=worker_inference_links,
            rollout_arena_group=arena_group,
        )
    )


def _rollout_arena_slack_sample_count(
    execution_config: ExecutionConfig,
) -> int:
    """Return physical arena slack for already-running game envs."""
    per_env_slack = 256
    return execution_config.game_envs_per_worker * per_env_slack


def _start_model_rank_pool(
    *,
    context: SpawnContext,
    run_dir: Path,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    distributed_group: _DistributedUpdateGroup | None,
    rank_inference_peers: tuple[tuple[AsyncPolicyPeer, ...], ...],
    rollout_arena_handles: tuple[RolloutArenaHandle, ...],
) -> _result.Ok[_ModelRankPool | None] | _result.Rejected:
    if not execution_config.uses_model_rank_processes():
        return Ok(value=None)
    handles: list[_ModelRankHandle] = []
    for index, model_rank_device in enumerate(
        execution_config.model_ranks.devices
    ):
        control_link = create_async_process_control_link(
            protocol=_MODEL_RANK_CONTROL_PROTOCOL,
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
                "control": control_link.child,
                "inference_peers": rank_inference_peers[index],
                "assigned_rollout_arena_handles": (
                    _model_rank_rollout_arena_handles(
                        execution_config=execution_config,
                        rollout_arena_handles=rollout_arena_handles,
                        model_rank_index=index,
                    )
                ),
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
        start_child_process_ignoring_terminal_interrupt(process)
        control_link.child.close()
        handles.append(
            _ModelRankHandle(
                index=index,
                control=control_link.coordinator,
                process=process,
            )
        )
    return Ok(
        value=_ModelRankPool(
            handles=tuple(handles),
        )
    )


def _model_rank_rollout_arena_handles(
    *,
    execution_config: ExecutionConfig,
    rollout_arena_handles: tuple[RolloutArenaHandle, ...],
    model_rank_index: int,
) -> tuple[RolloutArenaHandle, ...]:
    assert model_rank_index >= 0
    return tuple(
        handle
        for handle in rollout_arena_handles
        if execution_config.model_rank_index_for_worker(
            handle.worker_index
        )
        == model_rank_index
    )


def _worker_inference_links(
    *,
    worker_count: int,
) -> tuple[_WorkerInferenceLink, ...]:
    assert worker_count > 0
    links: list[_WorkerInferenceLink] = []
    for worker_index in range(worker_count):
        pair = create_async_socket_pair()
        links.append(
            _WorkerInferenceLink(
                worker_peer=AsyncPolicyPeer(
                    worker_index=worker_index,
                    endpoint=pair.first,
                ),
                model_rank_peer=AsyncPolicyPeer(
                    worker_index=worker_index,
                    endpoint=pair.second,
                ),
            )
        )
    return tuple(links)


def _rank_inference_peers(
    *,
    execution_config: ExecutionConfig,
    worker_inference_links: tuple[_WorkerInferenceLink, ...],
) -> tuple[tuple[AsyncPolicyPeer, ...], ...]:
    if not execution_config.uses_model_rank_processes():
        return ()
    groups: list[list[AsyncPolicyPeer]] = [
        [] for _ in range(execution_config.model_rank_process_count())
    ]
    for worker_index, link in enumerate(worker_inference_links):
        model_rank_index = execution_config.model_rank_index_for_worker(
            worker_index
        )
        groups[model_rank_index].append(link.model_rank_peer)
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


async def _run_training_update(
    *,
    pools: _RuntimePools,
    execution_config: ExecutionConfig,
    policy_version: int,
) -> _result.Ok[TrainingUpdateResult] | _result.Rejected:
    reset_result = reset_rollout_arenas(
        group=pools.rollout_arena_group,
        policy_version=policy_version,
    )
    if isinstance(reset_result, Rejected):
        return reset_result
    session_result = await start_worker_sampling_session(
        handles=pools.worker_pool.handles,
        execution_config=execution_config,
        policy_version=policy_version,
    )
    if isinstance(session_result, Rejected):
        return session_result
    session = session_result.value
    target_snapshot_result = wait_rollout_sample_target(
        group=pools.rollout_arena_group,
        policy_version=policy_version,
        target_sample_count=execution_config.samples_per_update,
        timeout_seconds=execution_config.timeouts.rollout_response_seconds,
    )
    if isinstance(target_snapshot_result, Rejected):
        return await reject_after_sampling_cleanup(
            session=session,
            timeout_seconds=(
                execution_config.timeouts.rollout_response_seconds
            ),
            failure=target_snapshot_result,
        )
    stopped_result = await stop_worker_sampling_session(
        session=session,
        timeout_seconds=execution_config.timeouts.rollout_response_seconds,
    )
    if isinstance(stopped_result, Rejected):
        return stopped_result
    update_result = await _run_compute_updates(
        worker_pool=pools.worker_pool,
        model_rank_pool=pools.model_rank_pool,
        policy_version=policy_version,
        update_timeout_seconds=execution_config.timeouts.update_seconds,
    )
    if isinstance(update_result, Rejected):
        return update_result
    snapshot_result = snapshot_rollout_arenas(
        group=pools.rollout_arena_group,
        policy_version=policy_version,
    )
    if isinstance(snapshot_result, Rejected):
        return snapshot_result
    return Ok(
        value=TrainingUpdateResult(
            snapshot=snapshot_result.value,
            update_stats=update_result.value.update_stats,
        )
    )


async def _sync_compute_rank_states(
    *,
    worker_pool: _WorkerPool,
    model_rank_pool: _ModelRankPool | None,
    state: RuntimeTrainingState,
    policy_version: int,
    state_sync_timeout_seconds: float,
) -> _result.Ok[None] | _result.Rejected:
    if model_rank_pool is None:
        return await _sync_worker_states(
            worker_pool=worker_pool,
            state=state,
            policy_version=policy_version,
            state_sync_timeout_seconds=state_sync_timeout_seconds,
        )
    return await _sync_model_rank_states(
        model_rank_pool=model_rank_pool,
        state=state,
        policy_version=policy_version,
        state_sync_timeout_seconds=state_sync_timeout_seconds,
    )


async def _run_compute_updates(
    *,
    worker_pool: _WorkerPool,
    model_rank_pool: _ModelRankPool | None,
    policy_version: int,
    update_timeout_seconds: float,
) -> _result.Ok[_UpdateResult] | _result.Rejected:
    if model_rank_pool is not None:
        return await _run_model_rank_updates(
            model_rank_pool=model_rank_pool,
            policy_version=policy_version,
            update_timeout_seconds=update_timeout_seconds,
        )
    return await _run_worker_updates(
        worker_pool=worker_pool,
        policy_version=policy_version,
        update_timeout_seconds=update_timeout_seconds,
    )


async def _run_worker_updates(
    *,
    worker_pool: _WorkerPool,
    policy_version: int,
    update_timeout_seconds: float,
) -> _result.Ok[_UpdateResult] | _result.Rejected:
    send_result = await _send_worker_update_commands(
        worker_pool=worker_pool,
        policy_version=policy_version,
    )
    if isinstance(send_result, Rejected):
        return send_result
    responses_result = await _receive_worker_updates(
        worker_pool=worker_pool,
        policy_version=policy_version,
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
            update_stats=_aggregate_ppo_update_stats(update_stats),
        )
    )


async def _send_worker_update_commands(
    *,
    worker_pool: _WorkerPool,
    policy_version: int,
) -> _result.Ok[None] | _result.Rejected:
    for handle in worker_pool.handles:
        send_result = await handle.control.send_command(
            WorkerUpdateCommand(
                policy_version=policy_version,
            )
        )
        if isinstance(send_result, Rejected):
            return send_result
    return Ok(value=None)


async def _wait_worker_responses(
    *,
    handles: tuple[_WorkerHandle, ...],
    timeout_seconds: float,
) -> _result.Ok[tuple[_WorkerHandle, ...]] | _result.Rejected:
    ready_result = await wait_async_control_responses(
        endpoints=tuple(handle.control for handle in handles),
        timeout_seconds=timeout_seconds,
    )
    if isinstance(ready_result, Rejected):
        return ready_result
    return Ok(
        value=tuple(
            _worker_handle_for_control(
                handles=handles,
                control=control,
            )
            for control in ready_result.value
        )
    )


def _worker_handle_for_control(
    *,
    handles: tuple[_WorkerHandle, ...],
    control: AsyncCoordinatorControlEndpoint[
        WorkerCommand, WorkerResponse
    ],
) -> _WorkerHandle:
    for handle in handles:
        if handle.control is control:
            return handle
    raise AssertionError("ready worker control endpoint is unknown")


def _worker_command_rejection(
    response: WorkerCommandRejected,
) -> Rejected:
    return Rejected(
        reason=f"worker-{response.worker_index}: {response.reason}"
    )


async def _receive_worker_updates(
    *,
    worker_pool: _WorkerPool,
    policy_version: int,
    update_timeout_seconds: float,
    unexpected_sampling_reason: str,
) -> _result.Ok[tuple[WorkerUpdateCompleted, ...]] | _result.Rejected:
    responses: list[WorkerUpdateCompleted] = []
    pending = list(worker_pool.handles)
    while pending:
        ready_result = await _wait_worker_responses(
            handles=tuple(pending),
            timeout_seconds=update_timeout_seconds,
        )
        if isinstance(ready_result, Rejected):
            return ready_result
        for handle in ready_result.value:
            response_result = await handle.control.recv_response()
            if isinstance(response_result, Rejected):
                return response_result
            response = response_result.value
            pending.remove(handle)
            if isinstance(response, WorkerCommandRejected):
                return _worker_command_rejection(response)
            if isinstance(response, WorkerSamplingStarted):
                return Rejected(reason=unexpected_sampling_reason)
            if isinstance(
                response,
                WorkerSamplingStopped | WorkerSamplingAlreadyStopped,
            ):
                return Rejected(reason=unexpected_sampling_reason)
            if isinstance(response, WorkerStateLoaded):
                return Rejected(
                    reason=(
                        "worker returned state sync during "
                        "synchronized update"
                    )
                )
            if isinstance(response, WorkerSnapshotCompleted):
                return Rejected(
                    reason=(
                        "worker returned snapshot during "
                        "synchronized update"
                    )
                )
            if response.policy_version != policy_version:
                return Rejected(
                    reason="worker returned stale update policy version"
                )
            responses.append(response)
    return Ok(
        value=tuple(
            sorted(responses, key=lambda item: item.worker_index)
        )
    )


async def _sync_worker_states(
    *,
    worker_pool: _WorkerPool,
    state: RuntimeTrainingState,
    policy_version: int,
    state_sync_timeout_seconds: float,
) -> _result.Ok[None] | _result.Rejected:
    expected_indices = {handle.index for handle in worker_pool.handles}
    for handle in worker_pool.handles:
        send_result = await handle.control.send_command(
            WorkerLoadStateCommand(
                state=state,
                policy_version=policy_version,
            )
        )
        if isinstance(send_result, Rejected):
            return send_result
    loaded_indices: set[int] = set()
    pending = list(worker_pool.handles)
    while pending:
        ready_result = await _wait_worker_responses(
            handles=tuple(pending),
            timeout_seconds=state_sync_timeout_seconds,
        )
        if isinstance(ready_result, Rejected):
            return ready_result
        for handle in ready_result.value:
            response_result = await handle.control.recv_response()
            if isinstance(response_result, Rejected):
                return response_result
            response = response_result.value
            pending.remove(handle)
            if isinstance(response, WorkerCommandRejected):
                return _worker_command_rejection(response)
            if isinstance(
                response,
                WorkerSamplingStarted
                | WorkerSamplingStopped
                | WorkerSamplingAlreadyStopped,
            ):
                return Rejected(
                    reason="worker returned sampling during state sync"
                )
            if isinstance(response, WorkerUpdateCompleted):
                return Rejected(
                    reason="worker returned update during state sync"
                )
            if isinstance(response, WorkerSnapshotCompleted):
                return Rejected(
                    reason="worker returned snapshot during state sync"
                )
            if response.policy_version != policy_version:
                return Rejected(
                    reason=(
                        "worker state sync returned stale policy "
                        "version"
                    )
                )
            loaded_indices.add(response.worker_index)
    if loaded_indices != expected_indices:
        return Rejected(reason="worker state sync rank set mismatch")
    return Ok(value=None)


async def _sync_model_rank_states(
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
        send_result = await handle.control.send_command(
            ModelRankLoadStateCommand(
                state=state,
                policy_version=policy_version,
            )
        )
        if isinstance(send_result, Rejected):
            return send_result
    loaded: list[ModelRankStateLoaded] = []
    pending = list(model_rank_pool.handles)
    while pending:
        ready_result = await _wait_model_rank_responses(
            handles=tuple(pending),
            timeout_seconds=state_sync_timeout_seconds,
        )
        if isinstance(ready_result, Rejected):
            return ready_result
        for handle in ready_result.value:
            response_result = await handle.control.recv_response()
            if isinstance(response_result, Rejected):
                return response_result
            response = response_result.value
            pending.remove(handle)
            if isinstance(response, ModelRankRejected):
                return Rejected(
                    reason=(
                        f"model-rank-{response.model_rank_index}: "
                        f"{response.reason}"
                    )
                )
            if isinstance(response, ModelRankUpdateCompleted):
                return Rejected(
                    reason=(
                        "model rank returned update during state sync"
                    )
                )
            if isinstance(response, ModelRankSnapshotCompleted):
                return Rejected(
                    reason=(
                        "model rank returned snapshot during state sync"
                    )
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


async def _snapshot_compute_rank_state(
    *,
    worker_pool: _WorkerPool,
    model_rank_pool: _ModelRankPool | None,
    policy_version: int,
    snapshot_timeout_seconds: float,
) -> _result.Ok[RuntimeTrainingState] | _result.Rejected:
    if model_rank_pool is not None:
        return await _snapshot_model_rank_state(
            model_rank_pool=model_rank_pool,
            policy_version=policy_version,
            snapshot_timeout_seconds=snapshot_timeout_seconds,
        )
    return await _snapshot_worker_state(
        worker_pool=worker_pool,
        policy_version=policy_version,
        snapshot_timeout_seconds=snapshot_timeout_seconds,
    )


async def _snapshot_worker_state(
    *,
    worker_pool: _WorkerPool,
    policy_version: int,
    snapshot_timeout_seconds: float,
) -> _result.Ok[RuntimeTrainingState] | _result.Rejected:
    handle = _rank_zero_worker(worker_pool)
    send_result = await handle.control.send_command(
        WorkerSnapshotCommand(policy_version=policy_version)
    )
    if isinstance(send_result, Rejected):
        return send_result
    ready_result = await _wait_worker_responses(
        handles=(handle,), timeout_seconds=snapshot_timeout_seconds
    )
    if isinstance(ready_result, Rejected):
        return ready_result
    response_result = await ready_result.value[
        0
    ].control.recv_response()
    if isinstance(response_result, Rejected):
        return response_result
    response = response_result.value
    if isinstance(response, WorkerCommandRejected):
        return _worker_command_rejection(response)
    if isinstance(
        response,
        WorkerSamplingStarted
        | WorkerSamplingStopped
        | WorkerSamplingAlreadyStopped,
    ):
        return Rejected(
            reason="worker returned sampling during snapshot"
        )
    if isinstance(response, WorkerStateLoaded):
        return Rejected(
            reason="worker returned state sync during snapshot"
        )
    if isinstance(response, WorkerUpdateCompleted):
        return Rejected(reason="worker returned update during snapshot")
    if response.policy_version != policy_version:
        return Rejected(
            reason="worker snapshot returned stale policy version"
        )
    return Ok(value=response.state)


def _rank_zero_worker(worker_pool: _WorkerPool) -> _WorkerHandle:
    assert worker_pool.handles
    return min(worker_pool.handles, key=lambda item: item.index)


async def _snapshot_model_rank_state(
    *,
    model_rank_pool: _ModelRankPool,
    policy_version: int,
    snapshot_timeout_seconds: float,
) -> _result.Ok[RuntimeTrainingState] | _result.Rejected:
    handle = _rank_zero_model_rank(model_rank_pool)
    send_result = await handle.control.send_command(
        ModelRankSnapshotCommand(policy_version=policy_version)
    )
    if isinstance(send_result, Rejected):
        return send_result
    ready_result = await _wait_model_rank_responses(
        handles=(handle,), timeout_seconds=snapshot_timeout_seconds
    )
    if isinstance(ready_result, Rejected):
        return ready_result
    response_result = await ready_result.value[
        0
    ].control.recv_response()
    if isinstance(response_result, Rejected):
        return response_result
    response = response_result.value
    if isinstance(response, ModelRankRejected):
        return Rejected(
            reason=(
                f"model-rank-{response.model_rank_index}: "
                f"{response.reason}"
            )
        )
    if isinstance(response, ModelRankStateLoaded):
        return Rejected(
            reason="model rank returned state sync during snapshot"
        )
    if isinstance(response, ModelRankUpdateCompleted):
        return Rejected(
            reason="model rank returned update during snapshot"
        )
    if response.policy_version != policy_version:
        return Rejected(
            reason="model rank snapshot returned stale policy version"
        )
    return Ok(value=response.state)


def _rank_zero_model_rank(
    model_rank_pool: _ModelRankPool,
) -> _ModelRankHandle:
    assert model_rank_pool.handles
    return min(model_rank_pool.handles, key=lambda item: item.index)


async def _wait_model_rank_responses(
    *,
    handles: tuple[_ModelRankHandle, ...],
    timeout_seconds: float,
) -> _result.Ok[tuple[_ModelRankHandle, ...]] | _result.Rejected:
    ready_result = await wait_async_control_responses(
        endpoints=tuple(handle.control for handle in handles),
        timeout_seconds=timeout_seconds,
    )
    if isinstance(ready_result, Rejected):
        return ready_result
    return Ok(
        value=tuple(
            _model_rank_handle_for_control(
                handles=handles,
                control=control,
            )
            for control in ready_result.value
        )
    )


def _model_rank_handle_for_control(
    *,
    handles: tuple[_ModelRankHandle, ...],
    control: AsyncCoordinatorControlEndpoint[
        ModelRankCommand, ModelRankResponse
    ],
) -> _ModelRankHandle:
    for handle in handles:
        if handle.control is control:
            return handle
    raise AssertionError("ready model-rank control endpoint is unknown")


async def _run_model_rank_updates(
    *,
    model_rank_pool: _ModelRankPool,
    policy_version: int,
    update_timeout_seconds: float,
) -> _result.Ok[_UpdateResult] | _result.Rejected:
    send_result = await _send_model_rank_update_commands(
        model_rank_pool=model_rank_pool,
        policy_version=policy_version,
    )
    if isinstance(send_result, Rejected):
        return send_result
    responses: list[ModelRankUpdateCompleted] = []
    pending = list(model_rank_pool.handles)
    while pending:
        ready_result = await _wait_model_rank_responses(
            handles=tuple(pending),
            timeout_seconds=update_timeout_seconds,
        )
        if isinstance(ready_result, Rejected):
            return ready_result
        for handle in ready_result.value:
            response_result = await handle.control.recv_response()
            if isinstance(response_result, Rejected):
                return response_result
            response = response_result.value
            pending.remove(handle)
            if isinstance(response, ModelRankRejected):
                return Rejected(
                    reason=(
                        f"model-rank-{response.model_rank_index}: "
                        f"{response.reason}"
                    )
                )
            if isinstance(response, ModelRankStateLoaded):
                return Rejected(
                    reason=(
                        "model rank returned state sync during update"
                    )
                )
            if isinstance(response, ModelRankSnapshotCompleted):
                return Rejected(
                    reason=(
                        "model rank returned snapshot during update"
                    )
                )
            if response.policy_version != policy_version:
                return Rejected(
                    reason=(
                        "model rank returned stale update policy "
                        "version"
                    )
                )
            responses.append(response)
    ordered = tuple(sorted(responses, key=lambda item: item.rank_index))
    update_stats = tuple(response.update_stats for response in ordered)
    return Ok(
        value=_UpdateResult(
            update_stats=_aggregate_ppo_update_stats(update_stats),
        )
    )


async def _send_model_rank_update_commands(
    *,
    model_rank_pool: _ModelRankPool,
    policy_version: int,
) -> _result.Ok[None] | _result.Rejected:
    for handle in model_rank_pool.handles:
        send_result = await handle.control.send_command(
            ModelRankUpdateCommand(
                policy_version=policy_version,
            )
        )
        if isinstance(send_result, Rejected):
            return send_result
    return Ok(value=None)


async def _stop_runtime_pools(pools: _RuntimePools) -> None:
    await _stop_worker_pool(pools.worker_pool)
    if pools.model_rank_pool is not None:
        await _stop_model_rank_pool(pools.model_rank_pool)
    _close_worker_inference_links(pools.worker_inference_links)
    close_shared_rollout_arenas(pools.rollout_arena_group)


async def _stop_worker_pool(pool: _WorkerPool) -> None:
    for handle in pool.handles:
        await handle.control.send_command(
            StopWorkerCommand(reason="complete")
        )
    for handle in pool.handles:
        _stop_process(handle.process)
        handle.control.close()


async def _stop_model_rank_pool(pool: _ModelRankPool) -> None:
    for handle in pool.handles:
        await handle.control.send_command(
            ModelRankStopCommand(reason="complete")
        )
    for handle in pool.handles:
        _stop_process(handle.process)
        handle.control.close()


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
        link.worker_peer.close()
        link.model_rank_peer.close()


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
        argument_trace_batch_count=sum(
            profile.argument_trace_batch_count for profile in profiles
        ),
        argument_trace_row_count=sum(
            profile.argument_trace_row_count for profile in profiles
        ),
        argument_trace_token_count=sum(
            profile.argument_trace_token_count for profile in profiles
        ),
        argument_trace_valid_token_count=sum(
            profile.argument_trace_valid_token_count
            for profile in profiles
        ),
        argument_trace_padding_token_count=sum(
            profile.argument_trace_padding_token_count
            for profile in profiles
        ),
    )


def _mean(values: tuple[float, ...]) -> float:
    assert values
    return sum(values) / len(values)
