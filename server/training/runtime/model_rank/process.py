"""Model-rank process for delegated policy inference and PPO update."""

from __future__ import annotations

import time
from multiprocessing.connection import Connection
from typing import cast

import torch

from server import result as _result
from server.result import Ok, Rejected
from server.training.config import ModelConfig, TrainConfig
from server.training.policy_inference_wire import (
    build_completed_policy_response_wire,
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
from server.training.runtime.model_rank.inference_transport import (
    ConnectionPolicyRequestReceiver,
    send_policy_response,
    wait_for_ready_receivers,
)
from server.training.runtime.model_rank.messages import (
    ModelRankCommand,
    ModelRankCommandReceiver,
    ModelRankLoadStateCommand,
    ModelRankRejected,
    ModelRankResponseSender,
    ModelRankStateLoaded,
    ModelRankStopCommand,
    ModelRankUpdateCommand,
    ModelRankUpdateCompleted,
)
from server.training.runtime.model_rank.staging import (
    PolicyRequestStager,
    StagedPolicyRequestBatch,
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
    command_receiver: ModelRankCommandReceiver,
    response_sender: ModelRankResponseSender,
    inference_request_receivers: tuple[
        ConnectionPolicyRequestReceiver, ...
    ],
    inference_response_senders: tuple[Connection, ...],
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
        response_sender.put(
            ModelRankRejected(
                model_rank_index=model_rank_index,
                reason=setup_result.reason,
            )
        )
        return
    sync_result = initialize_distributed_rank(distributed_rank_config)
    if isinstance(sync_result, Rejected):
        response_sender.put(
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
    try:
        loop_result = _run_model_rank_event_loop(
            model_rank_index=model_rank_index,
            run_id=run_id,
            core=core,
            execution_config=execution_config,
            command_receiver=command_receiver,
            response_sender=response_sender,
            inference_request_receivers=inference_request_receivers,
            inference_response_senders=inference_response_senders,
            telemetry_sink=telemetry_sink,
        )
        if isinstance(loop_result, Rejected):
            response_sender.put(
                ModelRankRejected(
                    model_rank_index=model_rank_index,
                    reason=loop_result.reason,
                )
            )
        return
    finally:
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
    command_receiver: ModelRankCommandReceiver,
    response_sender: ModelRankResponseSender,
    inference_request_receivers: tuple[
        ConnectionPolicyRequestReceiver, ...
    ],
    inference_response_senders: tuple[Connection, ...],
    telemetry_sink: TelemetrySink,
) -> _result.Ok[None] | _result.Rejected:
    command_connection = cast(Connection, command_receiver)
    stager = PolicyRequestStager(
        batch_size=execution_config.model_inference_batch_size,
        max_observation_tokens=core.model_config.max_tokens,
        device=core.device,
    )
    while True:
        staged_result = stager.receive_ready_batch(
            inference_request_receivers
        )
        if isinstance(staged_result, Rejected):
            return staged_result
        staged_batch = staged_result.value
        if staged_batch is not None:
            serve_result = _serve_staged_inference_batch(
                model_rank_index=model_rank_index,
                run_id=run_id,
                core=core,
                staged_batch=staged_batch,
                response_senders=inference_response_senders,
                telemetry_sink=telemetry_sink,
                configured_batch_size=(
                    execution_config.model_inference_batch_size
                ),
            )
            if isinstance(serve_result, Rejected):
                return serve_result
            continue
        command = _try_receive_model_rank_command(command_receiver)
        if command is not None:
            command_result = _handle_model_rank_command(
                model_rank_index=model_rank_index,
                run_id=run_id,
                core=core,
                command=command,
                response_sender=response_sender,
                telemetry_sink=telemetry_sink,
            )
            if command_result == "stopped":
                return Ok(value=None)
            continue
        wait_for_ready_receivers(
            receivers=inference_request_receivers,
            extra_connections=(command_connection,),
            timeout_seconds=None,
        )


def _serve_staged_inference_batch(
    *,
    model_rank_index: int,
    run_id: str,
    core: ModelReplica,
    staged_batch: StagedPolicyRequestBatch,
    response_senders: tuple[Connection, ...],
    telemetry_sink: TelemetrySink,
    configured_batch_size: int,
) -> _result.Ok[None] | _result.Rejected:
    serve_start = time.perf_counter()
    serve_result = _serve_inference_batch(
        core=core,
        staged_batch=staged_batch,
        response_senders=response_senders,
    )
    serve_seconds = time.perf_counter() - serve_start
    if isinstance(serve_result, Rejected):
        return serve_result
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
                value=serve_seconds,
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


def _try_receive_model_rank_command(
    receiver: ModelRankCommandReceiver,
) -> ModelRankCommand | None:
    if not receiver.poll(0.0):
        return None
    return receiver.recv()


def _handle_model_rank_command(
    *,
    model_rank_index: int,
    run_id: str,
    core: ModelReplica,
    command: ModelRankCommand,
    response_sender: ModelRankResponseSender,
    telemetry_sink: TelemetrySink,
) -> str:
    if isinstance(command, ModelRankStopCommand):
        return "stopped"
    if isinstance(command, ModelRankLoadStateCommand):
        core.load_state(snapshot=command.state)
        response_sender.put(
            ModelRankStateLoaded(
                model_rank_index=model_rank_index,
                policy_version=command.policy_version,
            )
        )
        return "running"
    assert isinstance(command, ModelRankUpdateCommand)
    update_response = _run_model_rank_update(
        model_rank_index=model_rank_index,
        run_id=run_id,
        core=core,
        command=command,
        telemetry_sink=telemetry_sink,
    )
    response_sender.put(update_response)
    return "running"


def _run_model_rank_update(
    *,
    model_rank_index: int,
    run_id: str,
    core: ModelReplica,
    command: ModelRankUpdateCommand,
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
    if command.shard.rank_index != model_rank_index:
        return ModelRankRejected(
            model_rank_index=model_rank_index,
            reason="model-rank update shard targets a different rank",
        )
    update_result = core.update_shard(shard=command.shard)
    if isinstance(update_result, Rejected):
        return ModelRankRejected(
            model_rank_index=model_rank_index,
            reason=update_result.reason,
        )
    return ModelRankUpdateCompleted(
        model_rank_index=model_rank_index,
        rank_index=command.shard.rank_index,
        update_stats=update_result.value,
        state=core.snapshot(),
    )


def _serve_inference_batch(
    *,
    core: ModelReplica,
    staged_batch: StagedPolicyRequestBatch,
    response_senders: tuple[Connection, ...],
) -> _result.Ok[None] | _result.Rejected:
    decisions = core.decide_batch(staged_batch.device_batch)
    for route, decision_result in zip(
        staged_batch.routes, decisions, strict=True
    ):
        if route.worker_index >= len(response_senders):
            return Rejected(
                reason="policy request worker route is out of range"
            )
        sender = response_senders[route.worker_index]
        if isinstance(decision_result, Rejected):
            response = build_rejected_policy_response_wire(
                route=route,
                reason=decision_result.reason,
            )
        else:
            response = build_completed_policy_response_wire(
                route=route,
                decision=decision_result.value,
            )
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
