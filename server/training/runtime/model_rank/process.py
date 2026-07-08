"""Model-rank process for delegated policy inference and PPO update."""

from __future__ import annotations

import time
from multiprocessing.connection import Connection

import torch

from server import result as _result
from server.result import Ok, Rejected
from server.training.config import ModelConfig, TrainConfig
from server.training.policy_inference_wire import (
    PolicyRequestRoute,
    build_policy_response_wire_batch,
    build_rejected_policy_response_wire,
)
from server.training.ppo.distributed import (
    PPOUpdatePartition,
    single_update_partition,
)
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
    ConnectionPolicyRequestReceiver,
    send_policy_response,
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
    PolicyRequestStager,
    StagedPolicyRequestBatch,
)
from server.training.runtime.process_control import ChildControlEndpoint
from server.training.runtime.shared_rollout_arena import (
    RolloutArenaHandle,
    SharedRolloutArenaReader,
    attach_rollout_arena_reader,
)
from server.training.runtime.telemetry import (
    TelemetryEvent,
    TelemetryMeasurement,
    TelemetrySink,
)


def run_model_rank_process(
    *,
    model_rank_index: int,
    model_rank_device: str,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    control: ChildControlEndpoint[ModelRankCommand, ModelRankResponse],
    inference_request_receivers: tuple[
        ConnectionPolicyRequestReceiver, ...
    ],
    inference_response_senders: tuple[Connection, ...],
    rollout_arena_handles: tuple[RolloutArenaHandle, ...],
    telemetry_sink: TelemetrySink,
    distributed_rank_config: DistributedRankConfig | None,
) -> None:
    """Model-rank process main loop."""
    assert model_rank_index >= 0
    setup_result = _setup_model_rank_runtime(
        model_rank_device=model_rank_device,
        execution_config=execution_config,
    )
    if isinstance(setup_result, Rejected):
        control.send_response(
            ModelRankRejected(
                model_rank_index=model_rank_index,
                reason=setup_result.reason,
            )
        )
        return
    sync_result = initialize_distributed_rank(distributed_rank_config)
    if isinstance(sync_result, Rejected):
        control.send_response(
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
    arena_reader = attach_rollout_arena_reader(rollout_arena_handles)
    try:
        loop_result = _run_model_rank_event_loop(
            model_rank_index=model_rank_index,
            run_id=run_id,
            core=core,
            execution_config=execution_config,
            control=control,
            inference_request_receivers=inference_request_receivers,
            inference_response_senders=inference_response_senders,
            rollout_arena_reader=arena_reader,
            telemetry_sink=telemetry_sink,
        )
        if isinstance(loop_result, Rejected):
            control.send_response(
                ModelRankRejected(
                    model_rank_index=model_rank_index,
                    reason=loop_result.reason,
                )
            )
        return
    finally:
        arena_reader.close()
        destroy_distributed_rank()


def _setup_model_rank_runtime(
    *,
    model_rank_device: str,
    execution_config: ExecutionConfig,
) -> _result.Ok[torch.device] | _result.Rejected:
    return _resolve_model_rank_device(
        model_rank_kind=execution_config.model_ranks.kind,
        model_rank_device=model_rank_device,
    )


def _resolve_model_rank_device(
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
        return Ok(value=device)
    return Rejected(
        reason="CPU compute does not use model-rank process"
    )


def _run_model_rank_event_loop(
    *,
    model_rank_index: int,
    run_id: str,
    core: ModelReplica,
    execution_config: ExecutionConfig,
    control: ChildControlEndpoint[ModelRankCommand, ModelRankResponse],
    inference_request_receivers: tuple[
        ConnectionPolicyRequestReceiver, ...
    ],
    inference_response_senders: tuple[Connection, ...],
    rollout_arena_reader: SharedRolloutArenaReader,
    telemetry_sink: TelemetrySink,
) -> _result.Ok[None] | _result.Rejected:
    data_plane = ModelRankDataPlane(
        control=control,
        request_receivers=inference_request_receivers,
        stager=PolicyRequestStager(
            batch_size=execution_config.model_inference_batch_size,
            max_observation_tokens=core.model_config.max_tokens,
            device=core.device,
        ),
    )

    def process_batch(
        batch: StagedPolicyRequestBatch,
    ) -> _result.Ok[None] | _result.Rejected:
        return _process_staged_inference_batch(
            model_rank_index=model_rank_index,
            run_id=run_id,
            core=core,
            staged_batch=batch,
            response_senders=inference_response_senders,
            telemetry_sink=telemetry_sink,
            configured_batch_size=(
                execution_config.model_inference_batch_size
            ),
        )

    def reject_batch(
        *, routes: tuple[PolicyRequestRoute, ...], reason: str
    ) -> _result.Ok[None] | _result.Rejected:
        return _send_rejected_inference_batch(
            routes=routes,
            reason=reason,
            response_senders=inference_response_senders,
        )

    command_result = control.recv_command()
    if isinstance(command_result, Rejected):
        return command_result
    command = command_result.value
    while True:
        if isinstance(command, ModelRankStopCommand):
            return Ok(value=None)
        if isinstance(command, ModelRankLoadStateCommand):
            core.load_state(snapshot=command.state)
            state_loaded_result = control.send_response(
                ModelRankStateLoaded(
                    model_rank_index=model_rank_index,
                    policy_version=command.policy_version,
                )
            )
            if isinstance(state_loaded_result, Rejected):
                return state_loaded_result
            next_command = data_plane.run_until_command(
                policy_version=command.policy_version,
                process_batch=process_batch,
                reject_batch=reject_batch,
            )
            if isinstance(next_command, Rejected):
                return next_command
            command = next_command.value
            continue
        if isinstance(command, ModelRankSnapshotCommand):
            snapshot_result = control.send_response(
                ModelRankSnapshotCompleted(
                    model_rank_index=model_rank_index,
                    policy_version=command.policy_version,
                    state=core.snapshot(),
                )
            )
            if isinstance(snapshot_result, Rejected):
                return snapshot_result
            next_command = data_plane.run_until_command(
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
            telemetry_sink=telemetry_sink,
        )
        update_send_result = control.send_response(update_response)
        if isinstance(update_send_result, Rejected):
            return update_send_result
        if isinstance(update_response, ModelRankRejected):
            command_result = control.recv_command()
            if isinstance(command_result, Rejected):
                return command_result
            command = command_result.value
            continue
        next_command = data_plane.run_until_command(
            policy_version=command.policy_version + 1,
            process_batch=process_batch,
            reject_batch=reject_batch,
        )
        if isinstance(next_command, Rejected):
            return next_command
        command = next_command.value


def _process_staged_inference_batch(
    *,
    model_rank_index: int,
    run_id: str,
    core: ModelReplica,
    staged_batch: StagedPolicyRequestBatch,
    response_senders: tuple[Connection, ...],
    telemetry_sink: TelemetrySink,
    configured_batch_size: int,
) -> _result.Ok[None] | _result.Rejected:
    process_start = time.perf_counter()
    process_result = _process_inference_batch(
        core=core,
        staged_batch=staged_batch,
        response_senders=response_senders,
    )
    process_seconds = time.perf_counter() - process_start
    if isinstance(process_result, Rejected):
        return process_result
    telemetry_result = _record_model_rank_stage(
        telemetry_sink=telemetry_sink,
        run_id=run_id,
        model_rank_index=model_rank_index,
        stage="inference",
        total_rounds=0,
        total_updates=0,
        measurements=(
            TelemetryMeasurement(
                key="model_rank_inference_batch_size",
                value=staged_batch.batch_size(),
            ),
            TelemetryMeasurement(
                key="model_rank_inference_batch_fill_ratio",
                value=(
                    staged_batch.batch_size()
                    / float(configured_batch_size)
                ),
            ),
            TelemetryMeasurement(
                key="inference_wire_bytes",
                value=staged_batch.wire_byte_count,
            ),
            TelemetryMeasurement(
                key="model_rank_recv_seconds",
                value=staged_batch.recv_seconds,
            ),
            TelemetryMeasurement(
                key="model_rank_h2d_seconds",
                value=staged_batch.h2d_seconds,
            ),
            TelemetryMeasurement(
                key="model_rank_device_decode_seconds",
                value=staged_batch.device_decode_seconds,
            ),
            TelemetryMeasurement(
                key="model_rank_inference_seconds",
                value=process_seconds,
            ),
        ),
    )
    if isinstance(telemetry_result, Rejected):
        return telemetry_result
    return Ok(value=None)


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
    telemetry_sink: TelemetrySink,
) -> ModelRankUpdateCompleted | ModelRankRejected:
    telemetry_result = _record_model_rank_stage(
        telemetry_sink=telemetry_sink,
        run_id=run_id,
        model_rank_index=model_rank_index,
        stage="update",
        total_rounds=0,
        total_updates=command.policy_version,
    )
    if isinstance(telemetry_result, Rejected):
        return ModelRankRejected(
            model_rank_index=model_rank_index,
            reason=telemetry_result.reason,
        )
    returns_result = rollout_arena_reader.read_rank_batch(
        policy_version=command.policy_version,
        model_rank_index=model_rank_index,
    )
    if isinstance(returns_result, Rejected):
        return ModelRankRejected(
            model_rank_index=model_rank_index,
            reason=returns_result.reason,
        )
    update_result = core.update_returns(returns=returns_result.value)
    if isinstance(update_result, Rejected):
        return ModelRankRejected(
            model_rank_index=model_rank_index,
            reason=update_result.reason,
        )
    return ModelRankUpdateCompleted(
        model_rank_index=model_rank_index,
        rank_index=model_rank_index,
        policy_version=command.policy_version,
        update_stats=update_result.value,
    )


def _send_rejected_inference_batch(
    *,
    routes: tuple[PolicyRequestRoute, ...],
    reason: str,
    response_senders: tuple[Connection, ...],
) -> _result.Ok[None] | _result.Rejected:
    for route in routes:
        worker_index = route.worker_index
        if worker_index >= len(response_senders):
            return Rejected(
                reason="policy request worker route is out of range"
            )
        response = build_rejected_policy_response_wire(
            route=route,
            reason=reason,
        )
        send_result = send_policy_response(
            sender=response_senders[worker_index],
            response=response,
        )
        if isinstance(send_result, Rejected):
            return send_result
    return Ok(value=None)


def _process_inference_batch(
    *,
    core: ModelReplica,
    staged_batch: StagedPolicyRequestBatch,
    response_senders: tuple[Connection, ...],
) -> _result.Ok[None] | _result.Rejected:
    decisions = core.decide_batch(staged_batch.device_batch)
    response_batch = build_policy_response_wire_batch(
        routes=staged_batch.routes,
        decisions=decisions,
    )
    if isinstance(response_batch, Rejected):
        return response_batch
    for route, response in zip(
        staged_batch.routes, response_batch.value, strict=True
    ):
        if route.worker_index >= len(response_senders):
            return Rejected(
                reason="policy request worker route is out of range"
            )
        sender = response_senders[route.worker_index]
        send_result = send_policy_response(
            sender=sender,
            response=response,
        )
        if isinstance(send_result, Rejected):
            return send_result
    return Ok(value=None)


def _record_model_rank_stage(
    *,
    telemetry_sink: TelemetrySink,
    run_id: str,
    model_rank_index: int,
    stage: str,
    total_rounds: int,
    total_updates: int,
    measurements: tuple[TelemetryMeasurement, ...] = (),
) -> _result.Ok[None] | _result.Rejected:
    assert stage in ("inference", "update")
    return telemetry_sink.append(
        TelemetryEvent(
            run_id=run_id,
            process_label=f"model-rank-{model_rank_index}",
            stage=stage,
            total_rounds=total_rounds,
            total_updates=total_updates,
            progress_numerator=0,
            progress_denominator=1,
            unix_seconds=time.time(),
            measurements=measurements,
        )
    )
