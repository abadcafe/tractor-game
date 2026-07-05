"""Model-rank process for delegated policy inference and PPO update."""

from __future__ import annotations

import queue
import time
from multiprocessing.connection import Connection

import torch

from server import result as _result
from server.result import Ok, Rejected
from server.training.config import ModelConfig, TrainConfig
from server.training.policy_request_frame import (
    CompletedPolicyResponseFrame,
    RejectedPolicyResponseFrame,
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
    PolicyInferenceRequestBatch,
    PolicyInferenceResponseEnvelope,
    SharedMemoryPolicyRequestReceiver,
    receive_policy_request_batch,
    send_policy_response,
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
        SharedMemoryPolicyRequestReceiver, ...
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
        while True:
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
                    return
                continue
            receive_start = time.perf_counter()
            requests_result = receive_policy_request_batch(
                receivers=inference_request_receivers,
                batch_size=(
                    execution_config.model_inference_batch_size
                ),
                wait_seconds=(
                    execution_config.model_inference_max_wait_ms
                    / 1000.0
                ),
            )
            receive_seconds = time.perf_counter() - receive_start
            if isinstance(requests_result, Rejected):
                response_sender.put(
                    ModelRankRejected(
                        model_rank_index=model_rank_index,
                        reason=requests_result.reason,
                    )
                )
                return
            request_batch = requests_result.value
            if request_batch is None:
                continue
            serve_start = time.perf_counter()
            serve_result = _serve_inference_batch(
                core=core,
                request_batch=request_batch,
                response_senders=inference_response_senders,
            )
            serve_seconds = time.perf_counter() - serve_start
            if isinstance(serve_result, Rejected):
                response_sender.put(
                    ModelRankRejected(
                        model_rank_index=model_rank_index,
                        reason=serve_result.reason,
                    )
                )
                return
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
                        value=len(request_batch.requests),
                    ),
                    TelemetryMeasurement(
                        key="inference_frame_bytes",
                        value=request_batch.byte_count(),
                    ),
                    TelemetryMeasurement(
                        key="inference_transport_wait_seconds",
                        value=receive_seconds,
                    ),
                    TelemetryMeasurement(
                        key="model_rank_inference_seconds",
                        value=serve_seconds,
                    ),
                ),
            )
            if isinstance(telemetry_result, Rejected):
                response_sender.put(
                    ModelRankRejected(
                        model_rank_index=model_rank_index,
                        reason=telemetry_result.reason,
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
    try:
        return receiver.get(False, None)
    except queue.Empty:
        return None


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
    request_batch: PolicyInferenceRequestBatch,
    response_senders: tuple[Connection, ...],
) -> _result.Ok[None] | _result.Rejected:
    valid_requests = tuple(
        request
        for request in request_batch.requests
        if request.worker_index < len(response_senders)
    )
    if not valid_requests:
        return Ok(value=None)
    decisions = core.decide_batch(
        PolicyInferenceRequestBatch(
            requests=valid_requests
        ).request_frame_batch()
    )
    for request, decision_result in zip(
        valid_requests, decisions, strict=True
    ):
        sender = response_senders[request.worker_index]
        if isinstance(decision_result, Rejected):
            send_result = send_policy_response(
                sender=sender,
                envelope=PolicyInferenceResponseEnvelope(
                    worker_index=request.worker_index,
                    request_id=request.request_id,
                    frame=RejectedPolicyResponseFrame(
                        reason=decision_result.reason
                    ),
                ),
            )
        else:
            send_result = send_policy_response(
                sender=sender,
                envelope=PolicyInferenceResponseEnvelope(
                    worker_index=request.worker_index,
                    request_id=request.request_id,
                    frame=CompletedPolicyResponseFrame(
                        trace_token_ids=(
                            decision_result.value.trace_token_ids
                        ),
                        decision_handle=(
                            decision_result.value.decision_handle
                        ),
                        choice_count=decision_result.value.choice_count,
                    ),
                ),
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
