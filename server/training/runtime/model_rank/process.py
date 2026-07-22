"""Model-rank process for delegated policy inference and PPO update."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import torch

from server.foundation import result as _result
from server.foundation.result import Ok, Rejected
from server.training.config import TrainConfig
from server.training.model import ModelConfig
from server.training.policy_inference_batch import (
    PolicyRequestRoute,
    build_completed_policy_responses,
    build_rejected_policy_responses,
    encode_policy_response_batch_wire,
)
from server.training.policy_sampling import CompactPolicyDecisionBatch
from server.training.ppo.distributed import (
    PPOUpdatePartition,
    single_update_partition,
)
from server.training.runtime.async_ipc import AsyncChildControlEndpoint
from server.training.runtime.config import ExecutionConfig
from server.training.runtime.distributed import (
    DistributedRankConfig,
    destroy_distributed_rank,
    initialize_distributed_rank,
)
from server.training.runtime.model_rank.core import (
    ModelReplica,
    create_model_replica,
)
from server.training.runtime.model_rank.data_plane import (
    ModelRankDataPlane,
)
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
)
from server.training.runtime.model_rank.staging import (
    ModelRankInferenceBatch,
    PolicyRequestIngress,
)
from server.training.runtime.process_signals import (
    ignore_terminal_interrupt_in_child_process,
)
from server.training.runtime.shared_rollout_arena import (
    RolloutArenaHandle,
    SharedRolloutArenaReader,
    attach_rollout_arena_reader,
)
from server.training_events import EventContext, StructuredEventSink


@dataclass(slots=True)
class _WorkerResponseBucket:
    worker_index: int
    routes: list[PolicyRequestRoute]
    row_indices: list[int]

    def __post_init__(self) -> None:
        assert self.worker_index >= 0


def run_model_rank_process(
    *,
    model_rank_index: int,
    model_rank_device: str,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    control: AsyncChildControlEndpoint[
        ModelRankCommand, ModelRankResponse
    ],
    inference_peers: tuple[AsyncPolicyPeer, ...],
    assigned_rollout_arena_handles: tuple[RolloutArenaHandle, ...],
    event_sink: StructuredEventSink,
    distributed_rank_config: DistributedRankConfig | None,
) -> None:
    """Model-rank process main loop."""
    ignore_terminal_interrupt_in_child_process()
    asyncio.run(
        _run_model_rank_process_async(
            model_rank_index=model_rank_index,
            model_rank_device=model_rank_device,
            run_id=run_id,
            model_config=model_config,
            train_config=train_config,
            execution_config=execution_config,
            control=control,
            inference_peers=inference_peers,
            assigned_rollout_arena_handles=(
                assigned_rollout_arena_handles
            ),
            event_sink=event_sink,
            distributed_rank_config=distributed_rank_config,
        )
    )


async def _run_model_rank_process_async(
    *,
    model_rank_index: int,
    model_rank_device: str,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    control: AsyncChildControlEndpoint[
        ModelRankCommand, ModelRankResponse
    ],
    inference_peers: tuple[AsyncPolicyPeer, ...],
    assigned_rollout_arena_handles: tuple[RolloutArenaHandle, ...],
    event_sink: StructuredEventSink,
    distributed_rank_config: DistributedRankConfig | None,
) -> None:
    """Async model-rank process main loop."""
    assert model_rank_index >= 0
    setup_result = _setup_model_rank_runtime(
        model_rank_device=model_rank_device,
        execution_config=execution_config,
    )
    if isinstance(setup_result, Rejected):
        event_sink.emit(
            "process.start",
            error=setup_result.reason,
        )
        event_sink.close()
        await control.send_response(
            ModelRankRejected(
                model_rank_index=model_rank_index,
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
            ModelRankRejected(
                model_rank_index=model_rank_index,
                reason=sync_result.reason,
            )
        )
        return
    update_partition = _model_rank_update_partition(
        distributed_rank_config
    )
    core = create_model_replica(
        model_rank_index=model_rank_index,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
        device=setup_result.value,
        update_partition=update_partition,
    )
    arena_reader = attach_rollout_arena_reader(
        assigned_rollout_arena_handles
    )
    event_sink.emit(
        "process.start",
        fields={
            "model_rank_index": model_rank_index,
            "device": model_rank_device,
        },
    )
    try:
        loop_result = await _run_model_rank_event_loop(
            model_rank_index=model_rank_index,
            run_id=run_id,
            core=core,
            execution_config=execution_config,
            control=control,
            inference_peers=inference_peers,
            rollout_arena_reader=arena_reader,
            event_sink=event_sink,
        )
        if isinstance(loop_result, Rejected):
            await control.send_response(
                ModelRankRejected(
                    model_rank_index=model_rank_index,
                    reason=loop_result.reason,
                )
            )
        return
    finally:
        arena_reader.close()
        destroy_distributed_rank()
        event_sink.emit("process.stop")
        event_sink.close()


def _setup_model_rank_runtime(
    *,
    model_rank_device: str,
    execution_config: ExecutionConfig,
) -> _result.Ok[torch.device] | _result.Rejected:
    return resolve_model_rank_device(
        model_rank_kind=execution_config.model_ranks.kind,
        model_rank_device=model_rank_device,
    )


def resolve_model_rank_device(
    *,
    model_rank_kind: str,
    model_rank_device: str,
) -> _result.Ok[torch.device] | _result.Rejected:
    device = torch.device(model_rank_device)
    if model_rank_kind == "cuda":
        if device.type != "cuda":
            return Rejected(
                reason=f"invalid CUDA model rank: {model_rank_device}"
            )
        if not torch.cuda.is_available():
            return Rejected(
                reason=(
                    "--model-ranks cuda is unavailable in this "
                    "PyTorch runtime"
                )
            )
        device_index = _cuda_device_index(model_rank_device)
        if (
            device_index is None
            or device_index >= torch.cuda.device_count()
        ):
            return Rejected(
                reason=(
                    "CUDA model rank is unavailable: "
                    f"{model_rank_device}"
                )
            )
        return Ok(value=device)
    if model_rank_kind == "mps":
        if device.type != "mps":
            return Rejected(
                reason=f"invalid MPS model rank: {model_rank_device}"
            )
        if not torch.backends.mps.is_available():
            return Rejected(
                reason=(
                    "--model-ranks mps is unavailable in this "
                    "PyTorch runtime"
                )
            )
        return Ok(value=torch.device("mps:0"))
    return Rejected(
        reason="CPU compute does not use model-rank process"
    )


async def _run_model_rank_event_loop(
    *,
    model_rank_index: int,
    run_id: str,
    core: ModelReplica,
    execution_config: ExecutionConfig,
    control: AsyncChildControlEndpoint[
        ModelRankCommand, ModelRankResponse
    ],
    inference_peers: tuple[AsyncPolicyPeer, ...],
    rollout_arena_reader: SharedRolloutArenaReader,
    event_sink: StructuredEventSink,
) -> _result.Ok[None] | _result.Rejected:
    data_plane = ModelRankDataPlane(
        control=control,
        request_peers=inference_peers,
        ingress=PolicyRequestIngress(
            batch_size=execution_config.model_inference_batch_size,
            device=core.device,
        ),
    )

    async def process_batch(
        batch: ModelRankInferenceBatch,
    ) -> _result.Ok[None] | _result.Rejected:
        return await _process_staged_inference_batch(
            core=core,
            staged_batch=batch,
            response_peers=inference_peers,
        )

    async def reject_batch(
        *, routes: tuple[PolicyRequestRoute, ...], reason: str
    ) -> _result.Ok[None] | _result.Rejected:
        return await _send_rejected_inference_batch(
            routes=routes,
            reason=reason,
            response_peers=inference_peers,
        )

    command_result = await control.recv_command()
    if isinstance(command_result, Rejected):
        return command_result
    command = command_result.value
    while True:
        if isinstance(command, ModelRankStopCommand):
            return Ok(value=None)
        if isinstance(command, ModelRankLoadStateCommand):
            core.load_state(snapshot=command.state)
            state_loaded_result = await control.send_response(
                ModelRankStateLoaded(
                    model_rank_index=model_rank_index,
                    policy_version=command.policy_version,
                )
            )
            if isinstance(state_loaded_result, Rejected):
                return state_loaded_result
            next_command = await data_plane.run_until_command(
                policy_version=command.policy_version,
                process_batch=process_batch,
                reject_batch=reject_batch,
            )
            if isinstance(next_command, Rejected):
                return next_command
            command = next_command.value
            continue
        if isinstance(command, ModelRankSnapshotCommand):
            snapshot_result = await control.send_response(
                ModelRankSnapshotCompleted(
                    model_rank_index=model_rank_index,
                    policy_version=command.policy_version,
                    state=core.snapshot(),
                )
            )
            if isinstance(snapshot_result, Rejected):
                return snapshot_result
            next_command = await data_plane.run_until_command(
                policy_version=command.policy_version,
                process_batch=process_batch,
                reject_batch=reject_batch,
            )
            if isinstance(next_command, Rejected):
                return next_command
            command = next_command.value
            continue
        assert isinstance(command, ModelRankUpdateCommand)
        update_response = _run_model_rank_update(
            model_rank_index=model_rank_index,
            run_id=run_id,
            core=core,
            command=command,
            rollout_arena_reader=rollout_arena_reader,
            event_sink=event_sink,
        )
        update_send_result = await control.send_response(
            update_response
        )
        if isinstance(update_send_result, Rejected):
            return update_send_result
        if isinstance(update_response, ModelRankRejected):
            command_result = await control.recv_command()
            if isinstance(command_result, Rejected):
                return command_result
            command = command_result.value
            continue
        next_command = await data_plane.run_until_command(
            policy_version=command.policy_version + 1,
            process_batch=process_batch,
            reject_batch=reject_batch,
        )
        if isinstance(next_command, Rejected):
            return next_command
        command = next_command.value


async def _process_staged_inference_batch(
    *,
    core: ModelReplica,
    staged_batch: ModelRankInferenceBatch,
    response_peers: tuple[AsyncPolicyPeer, ...],
) -> _result.Ok[None] | _result.Rejected:
    return await _process_inference_batch(
        core=core,
        staged_batch=staged_batch,
        response_peers=response_peers,
    )


def _cuda_device_index(model_rank_device: str) -> int | None:
    if not model_rank_device.startswith("cuda:"):
        return None
    return int(model_rank_device.removeprefix("cuda:"))


def _model_rank_update_partition(
    config: DistributedRankConfig | None,
) -> PPOUpdatePartition:
    if config is None:
        return single_update_partition()
    return PPOUpdatePartition(
        rank=config.rank,
        world_size=config.world_size,
    )


def _run_model_rank_update(
    *,
    model_rank_index: int,
    run_id: str,
    core: ModelReplica,
    command: ModelRankUpdateCommand,
    rollout_arena_reader: SharedRolloutArenaReader,
    event_sink: StructuredEventSink,
) -> ModelRankUpdateCompleted | ModelRankRejected:
    context = EventContext(
        policy_version=command.policy_version,
        rollout_id=command.rollout_id,
        model_rank_index=model_rank_index,
    )
    started = time.perf_counter()
    read_start = time.perf_counter()
    returns_result = rollout_arena_reader.read_rank_batch(
        policy_version=command.policy_version,
        model_rank_index=model_rank_index,
        device=core.device,
    )
    arena_read_seconds = time.perf_counter() - read_start
    if isinstance(returns_result, Rejected):
        event_sink.emit(
            "update.rank",
            context=context,
            fields={
                "duration_seconds": max(
                    time.perf_counter() - started, 0.0
                )
            },
            error=returns_result.reason,
        )
        return ModelRankRejected(
            model_rank_index=model_rank_index,
            reason=returns_result.reason,
        )
    update_start = time.perf_counter()
    update_result = core.update_returns(returns=returns_result.value)
    update_seconds = time.perf_counter() - update_start
    if isinstance(update_result, Rejected):
        event_sink.emit(
            "update.rank",
            context=context,
            fields={
                "duration_seconds": max(
                    time.perf_counter() - started, 0.0
                )
            },
            error=update_result.reason,
        )
        return ModelRankRejected(
            model_rank_index=model_rank_index,
            reason=update_result.reason,
        )
    event_sink.emit(
        "update.rank",
        context=context,
        fields={
            "arena_read_seconds": arena_read_seconds,
            "update_seconds": update_seconds,
            "sample_count": int(
                returns_result.value.row_indices.shape[0]
            ),
            "step_count": returns_result.value.total_step_count,
            "round_count": returns_result.value.round_count,
        },
    )
    return ModelRankUpdateCompleted(
        model_rank_index=model_rank_index,
        rank_index=model_rank_index,
        policy_version=command.policy_version,
        update_stats=update_result.value,
    )


async def _send_rejected_inference_batch(
    *,
    routes: tuple[PolicyRequestRoute, ...],
    reason: str,
    response_peers: tuple[AsyncPolicyPeer, ...],
) -> _result.Ok[None] | _result.Rejected:
    routed_workers_result = _validate_response_routes(
        routes=routes, response_peers=response_peers
    )
    if isinstance(routed_workers_result, Rejected):
        return routed_workers_result
    for peer in response_peers:
        sender_routes = tuple(
            route
            for route in routes
            if route.worker_index == peer.worker_index
        )
        if not sender_routes:
            continue
        responses_result = build_rejected_policy_responses(
            routes=sender_routes, reason=reason
        )
        if isinstance(responses_result, Rejected):
            return responses_result
        wire_result = encode_policy_response_batch_wire(
            responses_result.value
        )
        if isinstance(wire_result, Rejected):
            return wire_result
        send_result = await peer.send_response(wire_result.value)
        if isinstance(send_result, Rejected):
            return send_result
    return Ok(value=None)


async def _process_inference_batch(
    *,
    core: ModelReplica,
    staged_batch: ModelRankInferenceBatch,
    response_peers: tuple[AsyncPolicyPeer, ...],
) -> _result.Ok[None] | _result.Rejected:
    decision_result = core.decide_batch(staged_batch.device_batch)
    if isinstance(decision_result, Rejected):
        return await _send_rejected_inference_batch(
            routes=staged_batch.routes,
            reason=decision_result.reason,
            response_peers=response_peers,
        )
    return await _send_response_batches(
        routes=staged_batch.routes,
        decisions=decision_result.value,
        response_peers=response_peers,
    )


async def _send_response_batches(
    *,
    routes: tuple[PolicyRequestRoute, ...],
    decisions: CompactPolicyDecisionBatch,
    response_peers: tuple[AsyncPolicyPeer, ...],
) -> _result.Ok[None] | _result.Rejected:
    assert len(routes) == decisions.row_count()
    route_validation = _validate_response_routes(
        routes=routes, response_peers=response_peers
    )
    if isinstance(route_validation, Rejected):
        return route_validation
    grouped = _group_response_batches_by_worker(
        routes=routes, decisions=decisions
    )
    for peer in response_peers:
        bucket = grouped.get(peer.worker_index)
        if bucket is None:
            continue
        responses_result = build_completed_policy_responses(
            routes=tuple(bucket.routes),
            decisions=_select_compact_decision_rows(
                decisions=decisions,
                rows=tuple(bucket.row_indices),
            ),
        )
        if isinstance(responses_result, Rejected):
            return responses_result
        wire_result = encode_policy_response_batch_wire(
            responses_result.value
        )
        if isinstance(wire_result, Rejected):
            return wire_result
        send_result = await peer.send_response(wire_result.value)
        if isinstance(send_result, Rejected):
            return send_result
    return Ok(value=None)


def _group_response_batches_by_worker(
    *,
    routes: tuple[PolicyRequestRoute, ...],
    decisions: CompactPolicyDecisionBatch,
) -> dict[int, _WorkerResponseBucket]:
    assert len(routes) == decisions.row_count()
    buckets: dict[int, _WorkerResponseBucket] = {}
    for row_index, route in enumerate(routes):
        bucket = buckets.get(route.worker_index)
        if bucket is None:
            bucket = _WorkerResponseBucket(
                worker_index=route.worker_index,
                routes=[],
                row_indices=[],
            )
            buckets[route.worker_index] = bucket
        bucket.routes.append(route)
        bucket.row_indices.append(row_index)
    return buckets


def _select_compact_decision_rows(
    *, decisions: CompactPolicyDecisionBatch, rows: tuple[int, ...]
) -> CompactPolicyDecisionBatch:
    assert rows
    if rows == tuple(range(decisions.row_count())):
        return decisions
    return CompactPolicyDecisionBatch(
        model_rank_index=decisions.model_rank_index,
        policy_versions=tuple(
            decisions.policy_versions[row] for row in rows
        ),
        row_indices=tuple(decisions.row_indices[row] for row in rows),
        choice_counts=tuple(
            decisions.choice_counts[row] for row in rows
        ),
        action_choice_batch=decisions.action_choice_batch.select_rows(
            rows
        ),
    )


def _validate_response_routes(
    *,
    routes: tuple[PolicyRequestRoute, ...],
    response_peers: tuple[AsyncPolicyPeer, ...],
) -> Ok[None] | Rejected:
    routed_workers = {route.worker_index for route in routes}
    sender_workers = {peer.worker_index for peer in response_peers}
    if not routed_workers.issubset(sender_workers):
        return Rejected(
            reason="policy request worker route is out of shard"
        )
    return Ok(value=None)
