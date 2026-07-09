"""Device materialization for columnar policy request batches."""

from __future__ import annotations

import torch
from torch import Tensor

from server import result as _result
from server.result import Ok, Rejected
from server.training.feature_schema import NUMERIC_FEATURE_COUNT
from server.training.packed_observation import (
    OBSERVATION_COMPONENT_COUNT,
    padded_packed_observation,
)
from server.training.policy_inference_batch.compiler import (
    PolicyRequestBatchBuilder,
)
from server.training.policy_inference_batch.frame import (
    decode_policy_request_frame_metadata,
)
from server.training.policy_inference_batch.schema import (
    F32,
    F64,
    I64,
    MAX_PAIR_PLAN_COUNT,
    MAX_TRACE_COUNT,
    ColumnLayout,
    policy_request_batch_layout,
)
from server.training.policy_inference_batch.types import (
    DevicePolicyRequestBatch,
    PolicyRequestBatch,
    PolicyRequestFrameMetadata,
    PolicyRequestInput,
    PolicyRequestWireFrame,
)
from server.training.semantic_action_plan import DeviceActionPlanBatch
from server.training.semantic_action_plan.frame import ActionPlanFrame
from server.training.semantic_action_plan.spec import ACTION_FACE_COUNT
from server.training.tensor_staging import staged_tensor
from server.training.tensorize import ObservationTensorBatch


def materialize_policy_request_inputs(
    *,
    requests: tuple[PolicyRequestInput, ...],
    batch_capacity: int,
    max_observation_tokens: int,
    device: torch.device,
) -> _result.Ok[DevicePolicyRequestBatch] | _result.Rejected:
    """Compile raw inputs and materialize them on one torch device."""
    assert requests
    assert len(requests) <= batch_capacity
    preparer = PolicyRequestBatchBuilder(
        batch_capacity=batch_capacity,
        max_observation_tokens=max_observation_tokens,
    )
    batch_result = preparer.compile_batch(requests)
    if isinstance(batch_result, Rejected):
        return batch_result
    return materialize_policy_request_batch(
        batch=batch_result.value, device=device
    )


def materialize_policy_request_batch(
    *, batch: PolicyRequestBatch, device: torch.device
) -> _result.Ok[DevicePolicyRequestBatch] | _result.Rejected:
    """Materialize one compiled request batch without wire framing."""
    padded_observations = tuple(
        padded_packed_observation(
            row.packed_observation,
            max_observation_tokens=batch.max_observation_tokens,
        )
        for row in batch.rows
    )
    action_plans = tuple(row.action_plan for row in batch.rows)
    return Ok(
        value=DevicePolicyRequestBatch(
            observation_batch=ObservationTensorBatch(
                component_ids=staged_tensor(
                    tuple(
                        observation.component_rows
                        for observation in padded_observations
                    ),
                    dtype=torch.long,
                    device=device,
                ),
                numeric_values=staged_tensor(
                    tuple(
                        observation.numeric_value_rows
                        for observation in padded_observations
                    ),
                    dtype=torch.float32,
                    device=device,
                ),
                numeric_masks=staged_tensor(
                    tuple(
                        observation.numeric_mask_rows
                        for observation in padded_observations
                    ),
                    dtype=torch.float32,
                    device=device,
                ),
            ),
            action_plan_batch=_materialize_action_plan_batch(
                plans=action_plans,
                padded_generation_steps=batch.padded_generation_steps,
                device=device,
            ),
            sampling_thresholds=staged_tensor(
                tuple(
                    _padded_thresholds(
                        row.sampling_thresholds,
                        padded_generation_steps=(
                            batch.padded_generation_steps
                        ),
                    )
                    for row in batch.rows
                ),
                dtype=torch.float64,
                device=device,
            ),
            generation_step_counts=staged_tensor(
                tuple(row.generation_step_count for row in batch.rows),
                dtype=torch.long,
                device=device,
            ),
            policy_versions=batch.policy_versions,
            padded_generation_steps=batch.padded_generation_steps,
        )
    )


def _materialize_action_plan_batch(
    *,
    plans: tuple[ActionPlanFrame, ...],
    padded_generation_steps: int,
    device: torch.device,
) -> DeviceActionPlanBatch:
    assert plans
    return DeviceActionPlanBatch(
        kind_codes=staged_tensor(
            tuple(plan.kind_code for plan in plans),
            dtype=torch.long,
            device=device,
        ),
        available_counts=staged_tensor(
            tuple(plan.available_counts for plan in plans),
            dtype=torch.long,
            device=device,
        ),
        effective_suits=staged_tensor(
            tuple(plan.effective_suits for plan in plans),
            dtype=torch.long,
            device=device,
        ),
        same_suit_mask=staged_tensor(
            tuple(plan.same_suit_mask for plan in plans),
            dtype=torch.bool,
            device=device,
        ),
        off_suit_mask=staged_tensor(
            tuple(plan.off_suit_mask for plan in plans),
            dtype=torch.bool,
            device=device,
        ),
        pair_face_mask=staged_tensor(
            tuple(plan.pair_face_mask for plan in plans),
            dtype=torch.bool,
            device=device,
        ),
        min_select=staged_tensor(
            tuple(plan.min_select for plan in plans),
            dtype=torch.long,
            device=device,
        ),
        max_select=staged_tensor(
            tuple(plan.max_select for plan in plans),
            dtype=torch.long,
            device=device,
        ),
        exact_select=staged_tensor(
            tuple(plan.exact_select for plan in plans),
            dtype=torch.long,
            device=device,
        ),
        required_same_suit_count=staged_tensor(
            tuple(plan.required_same_suit_count for plan in plans),
            dtype=torch.long,
            device=device,
        ),
        pair_floor=staged_tensor(
            tuple(plan.pair_floor for plan in plans),
            dtype=torch.long,
            device=device,
        ),
        has_tractor=staged_tensor(
            tuple(plan.has_tractor for plan in plans),
            dtype=torch.bool,
            device=device,
        ),
        trace_tokens=staged_tensor(
            tuple(
                _padded_trace_tokens(
                    plan.trace_tokens,
                    padded_generation_steps=padded_generation_steps,
                )
                for plan in plans
            ),
            dtype=torch.long,
            device=device,
        ),
        trace_token_mask=staged_tensor(
            tuple(
                _padded_trace_token_mask(
                    plan.trace_tokens,
                    padded_generation_steps=padded_generation_steps,
                )
                for plan in plans
            ),
            dtype=torch.bool,
            device=device,
        ),
        trace_lengths=staged_tensor(
            tuple(
                _padded_trace_lengths(plan.trace_tokens)
                for plan in plans
            ),
            dtype=torch.long,
            device=device,
        ),
        trace_row_mask=staged_tensor(
            tuple(
                _padded_trace_row_mask(plan.trace_tokens)
                for plan in plans
            ),
            dtype=torch.bool,
            device=device,
        ),
        pair_plan_masks=staged_tensor(
            tuple(
                _padded_pair_plan_masks(plan.pair_plan_masks)
                for plan in plans
            ),
            dtype=torch.bool,
            device=device,
        ),
        pair_plan_row_mask=staged_tensor(
            tuple(
                _padded_pair_plan_row_mask(plan.pair_plan_masks)
                for plan in plans
            ),
            dtype=torch.bool,
            device=device,
        ),
    )


def _padded_thresholds(
    values: tuple[float, ...], *, padded_generation_steps: int
) -> tuple[float, ...]:
    assert len(values) <= padded_generation_steps
    padding = padded_generation_steps - len(values)
    return (*values, *(0.0 for _ in range(padding)))


def _padded_trace_tokens(
    traces: tuple[tuple[int, ...], ...],
    *,
    padded_generation_steps: int,
) -> tuple[tuple[int, ...], ...]:
    assert len(traces) <= MAX_TRACE_COUNT
    return tuple(
        _padded_i64_row(trace, width=padded_generation_steps)
        for trace in _pad_tuple(traces, count=MAX_TRACE_COUNT, empty=())
    )


def _padded_trace_token_mask(
    traces: tuple[tuple[int, ...], ...],
    *,
    padded_generation_steps: int,
) -> tuple[tuple[bool, ...], ...]:
    assert len(traces) <= MAX_TRACE_COUNT
    return tuple(
        _padded_bool_row(
            tuple(True for _ in trace),
            width=padded_generation_steps,
        )
        for trace in _pad_tuple(traces, count=MAX_TRACE_COUNT, empty=())
    )


def _padded_trace_lengths(
    traces: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    assert len(traces) <= MAX_TRACE_COUNT
    return _padded_i64_row(
        tuple(len(trace) for trace in traces), width=MAX_TRACE_COUNT
    )


def _padded_trace_row_mask(
    traces: tuple[tuple[int, ...], ...],
) -> tuple[bool, ...]:
    assert len(traces) <= MAX_TRACE_COUNT
    return _padded_bool_row(
        tuple(True for _ in traces), width=MAX_TRACE_COUNT
    )


def _padded_pair_plan_masks(
    pair_plans: tuple[tuple[bool, ...], ...],
) -> tuple[tuple[bool, ...], ...]:
    assert len(pair_plans) <= MAX_PAIR_PLAN_COUNT
    return tuple(
        _padded_bool_row(row, width=ACTION_FACE_COUNT)
        for row in _pad_tuple(
            pair_plans,
            count=MAX_PAIR_PLAN_COUNT,
            empty=tuple(False for _ in range(ACTION_FACE_COUNT)),
        )
    )


def _padded_pair_plan_row_mask(
    pair_plans: tuple[tuple[bool, ...], ...],
) -> tuple[bool, ...]:
    assert len(pair_plans) <= MAX_PAIR_PLAN_COUNT
    return _padded_bool_row(
        tuple(True for _ in pair_plans),
        width=MAX_PAIR_PLAN_COUNT,
    )


def _padded_i64_row(
    values: tuple[int, ...], *, width: int
) -> tuple[int, ...]:
    assert len(values) <= width
    return (*values, *(0 for _ in range(width - len(values))))


def _padded_bool_row(
    values: tuple[bool, ...], *, width: int
) -> tuple[bool, ...]:
    assert len(values) <= width
    return (*values, *(False for _ in range(width - len(values))))


def _pad_tuple[T](
    values: tuple[T, ...], *, count: int, empty: T
) -> tuple[T, ...]:
    assert len(values) <= count
    return (*values, *(empty for _ in range(count - len(values))))


def materialize_policy_request_batch_frame(
    *, frame: PolicyRequestWireFrame, device: torch.device
) -> _result.Ok[DevicePolicyRequestBatch] | _result.Rejected:
    """Materialize one owned request frame on one torch device."""
    metadata_result = decode_policy_request_frame_metadata(frame.view())
    if isinstance(metadata_result, Rejected):
        return metadata_result
    host_frame = _host_frame_tensor(frame.view(), device=device)
    device_frame = copy_policy_request_host_frame_to_device(
        host_frame=host_frame,
        device_slot=None,
        device=device,
    )
    return materialize_policy_request_frame(
        device_frame=device_frame,
        metadata=metadata_result.value,
    )


def copy_policy_request_host_frame_to_device(
    *,
    host_frame: Tensor,
    device_slot: Tensor | None,
    device: torch.device,
) -> Tensor:
    """Copy one host request frame to its target device."""
    assert host_frame.device.type == "cpu"
    if device.type == "cpu":
        return host_frame
    if device.type != "cuda":
        if device_slot is None:
            return host_frame.to(device=device)
        device_slot.copy_(host_frame)
        return device_slot
    if device_slot is None:
        return host_frame.to(device=device, non_blocking=True)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        device_slot.copy_(host_frame, non_blocking=True)
    event = torch.cuda.Event()
    event.record(stream)
    torch.cuda.current_stream(device).wait_event(event)
    return device_slot


def materialize_policy_request_frame(
    *,
    device_frame: Tensor,
    metadata: PolicyRequestFrameMetadata,
) -> _result.Ok[DevicePolicyRequestBatch] | _result.Rejected:
    """Return final request tensors as views over a staged frame."""
    validate_result = _validate_device_frame(
        device_frame=device_frame,
        metadata=metadata,
    )
    if isinstance(validate_result, Rejected):
        return validate_result
    layout = policy_request_batch_layout(
        batch_capacity=metadata.batch_capacity,
        max_observation_tokens=metadata.max_observation_tokens,
        padded_generation_steps=metadata.padded_generation_steps,
    )
    row_count = metadata.row_count
    return Ok(
        value=DevicePolicyRequestBatch(
            observation_batch=ObservationTensorBatch(
                component_ids=_view_i64_column(
                    device_frame,
                    layout.component_ids,
                    (
                        metadata.batch_capacity,
                        metadata.max_observation_tokens,
                        OBSERVATION_COMPONENT_COUNT,
                    ),
                )[:row_count],
                numeric_values=_view_f32_column(
                    device_frame,
                    layout.numeric_values,
                    (
                        metadata.batch_capacity,
                        metadata.max_observation_tokens,
                        NUMERIC_FEATURE_COUNT,
                    ),
                )[:row_count],
                numeric_masks=_view_f32_column(
                    device_frame,
                    layout.numeric_masks,
                    (
                        metadata.batch_capacity,
                        metadata.max_observation_tokens,
                        NUMERIC_FEATURE_COUNT,
                    ),
                )[:row_count],
            ),
            action_plan_batch=DeviceActionPlanBatch(
                kind_codes=_view_i64_column(
                    device_frame,
                    layout.kind_codes,
                    (metadata.batch_capacity,),
                )[:row_count],
                available_counts=_view_i64_column(
                    device_frame,
                    layout.available_counts,
                    (metadata.batch_capacity, ACTION_FACE_COUNT),
                )[:row_count],
                effective_suits=_view_i64_column(
                    device_frame,
                    layout.effective_suits,
                    (metadata.batch_capacity, ACTION_FACE_COUNT),
                )[:row_count],
                same_suit_mask=_view_bool_column(
                    device_frame,
                    layout.same_suit_mask,
                    (metadata.batch_capacity, ACTION_FACE_COUNT),
                )[:row_count],
                off_suit_mask=_view_bool_column(
                    device_frame,
                    layout.off_suit_mask,
                    (metadata.batch_capacity, ACTION_FACE_COUNT),
                )[:row_count],
                pair_face_mask=_view_bool_column(
                    device_frame,
                    layout.pair_face_mask,
                    (metadata.batch_capacity, ACTION_FACE_COUNT),
                )[:row_count],
                min_select=_view_i64_column(
                    device_frame,
                    layout.min_select,
                    (metadata.batch_capacity,),
                )[:row_count],
                max_select=_view_i64_column(
                    device_frame,
                    layout.max_select,
                    (metadata.batch_capacity,),
                )[:row_count],
                exact_select=_view_i64_column(
                    device_frame,
                    layout.exact_select,
                    (metadata.batch_capacity,),
                )[:row_count],
                required_same_suit_count=_view_i64_column(
                    device_frame,
                    layout.required_same_suit_count,
                    (metadata.batch_capacity,),
                )[:row_count],
                pair_floor=_view_i64_column(
                    device_frame,
                    layout.pair_floor,
                    (metadata.batch_capacity,),
                )[:row_count],
                has_tractor=_view_bool_column(
                    device_frame,
                    layout.has_tractor,
                    (metadata.batch_capacity,),
                )[:row_count],
                trace_tokens=_view_i64_column(
                    device_frame,
                    layout.trace_tokens,
                    (
                        metadata.batch_capacity,
                        MAX_TRACE_COUNT,
                        metadata.padded_generation_steps,
                    ),
                )[:row_count],
                trace_token_mask=_view_bool_column(
                    device_frame,
                    layout.trace_token_mask,
                    (
                        metadata.batch_capacity,
                        MAX_TRACE_COUNT,
                        metadata.padded_generation_steps,
                    ),
                )[:row_count],
                trace_lengths=_view_i64_column(
                    device_frame,
                    layout.trace_lengths,
                    (metadata.batch_capacity, MAX_TRACE_COUNT),
                )[:row_count],
                trace_row_mask=_view_bool_column(
                    device_frame,
                    layout.trace_row_mask,
                    (metadata.batch_capacity, MAX_TRACE_COUNT),
                )[:row_count],
                pair_plan_masks=_view_bool_column(
                    device_frame,
                    layout.pair_plan_masks,
                    (
                        metadata.batch_capacity,
                        MAX_PAIR_PLAN_COUNT,
                        ACTION_FACE_COUNT,
                    ),
                )[:row_count],
                pair_plan_row_mask=_view_bool_column(
                    device_frame,
                    layout.pair_plan_row_mask,
                    (metadata.batch_capacity, MAX_PAIR_PLAN_COUNT),
                )[:row_count],
            ),
            sampling_thresholds=_view_f64_column(
                device_frame,
                layout.sampling_thresholds,
                (
                    metadata.batch_capacity,
                    metadata.padded_generation_steps,
                ),
            )[:row_count],
            generation_step_counts=_view_i64_column(
                device_frame,
                layout.generation_step_counts,
                (metadata.batch_capacity,),
            )[:row_count],
            policy_versions=metadata.policy_versions,
            padded_generation_steps=metadata.padded_generation_steps,
        )
    )


def _host_frame_tensor(
    data: memoryview, *, device: torch.device
) -> Tensor:
    host = torch.empty(
        (len(data),),
        dtype=torch.uint8,
        device=torch.device("cpu"),
        pin_memory=device.type == "cuda",
    )
    memoryview(host.numpy())[:] = data
    return host


def _validate_device_frame(
    *,
    device_frame: Tensor,
    metadata: PolicyRequestFrameMetadata,
) -> Ok[None] | Rejected:
    if device_frame.dtype != torch.uint8:
        return Rejected(
            reason="policy request frame tensor must be uint8"
        )
    if device_frame.ndim != 1:
        return Rejected(reason="policy request frame tensor must be 1D")
    if int(device_frame.shape[0]) != metadata.byte_count:
        return Rejected(reason="policy request frame length mismatch")
    return Ok(value=None)


def _view_i64_column(
    frame: Tensor, column: ColumnLayout, shape: tuple[int, ...]
) -> Tensor:
    return (
        _view_column(frame, column, I64.size)
        .view(torch.int64)
        .reshape(shape)
    )


def _view_f32_column(
    frame: Tensor, column: ColumnLayout, shape: tuple[int, ...]
) -> Tensor:
    return (
        _view_column(frame, column, F32.size)
        .view(torch.float32)
        .reshape(shape)
    )


def _view_f64_column(
    frame: Tensor, column: ColumnLayout, shape: tuple[int, ...]
) -> Tensor:
    return (
        _view_column(frame, column, F64.size)
        .view(torch.float64)
        .reshape(shape)
    )


def _view_bool_column(
    frame: Tensor, column: ColumnLayout, shape: tuple[int, ...]
) -> Tensor:
    return (
        _view_column(frame, column, 1).view(torch.bool).reshape(shape)
    )


def _view_column(
    frame: Tensor, column: ColumnLayout, element_bytes: int
) -> Tensor:
    assert column.total_bytes % element_bytes == 0
    return frame[column.offset : column.offset + column.total_bytes]
