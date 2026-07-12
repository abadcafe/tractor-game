"""Device materialization for columnar policy request batches."""

from __future__ import annotations

import torch
from torch import Tensor

from server.foundation import result as _result
from server.foundation.result import Ok, Rejected
from server.training.feature_schema import NUMERIC_FEATURE_COUNT
from server.training.packed_observation import (
    OBSERVATION_COMPONENT_COUNT,
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
)
from server.training.policy_inference_batch.types import (
    BorrowedPolicyRequestBatch,
    DevicePolicyRequestBatch,
    PolicyRequestFrameMetadata,
    PolicyRequestWireFrame,
)
from server.training.semantic_action_plan import DeviceActionPlanBatch
from server.training.semantic_action_plan.spec import ACTION_FACE_COUNT
from server.training.tensorize import ObservationTensorBatch


def materialize_borrowed_policy_request_batch(
    *, batch: BorrowedPolicyRequestBatch, device: torch.device
) -> _result.Ok[DevicePolicyRequestBatch] | _result.Rejected:
    """Materialize one borrowed request frame on one torch device."""
    host_frame = _host_frame_tensor(batch.frame.view(), device=device)
    device_frame = copy_policy_request_host_frame_to_device(
        host_frame=host_frame,
        device_slot=None,
        device=device,
    )
    return materialize_policy_request_frame(
        device_frame=device_frame,
        metadata=batch.metadata,
    )


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
    device_slot.copy_(host_frame, non_blocking=True)
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
    layout = metadata.layout
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
