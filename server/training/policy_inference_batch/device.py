"""Materialize typed columnar request frames on torch devices."""

from __future__ import annotations

import torch
from torch import Tensor

from server.foundation import result as _result
from server.foundation.result import Ok, Rejected
from server.training.observation_structure import STRUCTURE_AXIS_COUNT
from server.training.policy_inference_batch.frame import (
    decode_policy_request_frame_metadata,
)
from server.training.policy_inference_batch.schema import (
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
from server.training.semantic_actions.choices import CARD_CHOICE_COUNT
from server.training.tensorize import ObservationTensorBatch
from server.training.tokenization.encoding_schema import CATEGORY_COUNT

_FLOAT32_BELOW_ONE = float.fromhex("0x1.fffffep-1")


def materialize_borrowed_policy_request_batch(
    *, batch: BorrowedPolicyRequestBatch, device: torch.device
) -> _result.Ok[DevicePolicyRequestBatch] | _result.Rejected:
    host = _host_frame_tensor(batch.frame.view(), device=device)
    device_frame = copy_policy_request_host_frame_to_device(
        host_frame=host, device_slot=None, device=device
    )
    return materialize_policy_request_frame(
        device_frame=device_frame,
        metadata=batch.metadata,
        host_frame=host,
    )


def materialize_policy_request_batch_frame(
    *, frame: PolicyRequestWireFrame, device: torch.device
) -> _result.Ok[DevicePolicyRequestBatch] | _result.Rejected:
    metadata_result = decode_policy_request_frame_metadata(frame.view())
    if isinstance(metadata_result, Rejected):
        return metadata_result
    host = _host_frame_tensor(frame.view(), device=device)
    device_frame = copy_policy_request_host_frame_to_device(
        host_frame=host, device_slot=None, device=device
    )
    return materialize_policy_request_frame(
        device_frame=device_frame,
        metadata=metadata_result.value,
        host_frame=host,
    )


def copy_policy_request_host_frame_to_device(
    *,
    host_frame: Tensor,
    device_slot: Tensor | None,
    device: torch.device,
) -> Tensor:
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
    host_frame: Tensor | None = None,
) -> _result.Ok[DevicePolicyRequestBatch] | _result.Rejected:
    valid = _validate_device_frame(
        device_frame=device_frame, metadata=metadata
    )
    if isinstance(valid, Rejected):
        return valid
    thresholds = _materialize_sampling_thresholds(
        device_frame=device_frame,
        host_frame=host_frame,
        metadata=metadata,
    )
    if isinstance(thresholds, Rejected):
        return thresholds
    layout = metadata.layout
    capacity = metadata.batch_capacity
    tokens = metadata.observation_token_capacity
    rows = metadata.row_count
    observation = ObservationTensorBatch(
        category_ids=_view_i64_column(
            device_frame,
            layout.category_ids,
            (capacity, tokens, CATEGORY_COUNT),
        )[:rows],
        scalar_values=_view_f32_column(
            device_frame, layout.scalar_values, (capacity, tokens)
        )[:rows],
        card_rule_values=_view_f32_column(
            device_frame,
            layout.card_rule_values,
            (capacity, tokens, 2),
        )[:rows],
        encoded_structure_coordinates=_view_i64_column(
            device_frame,
            layout.encoded_structure_coordinates,
            (capacity, tokens, STRUCTURE_AXIS_COUNT),
        )[:rows],
        candidate_category_ids=_view_i64_column(
            device_frame,
            layout.candidate_category_ids,
            (capacity, CARD_CHOICE_COUNT, 3),
        )[:rows],
        candidate_counts=_view_f32_column(
            device_frame,
            layout.candidate_counts,
            (capacity, CARD_CHOICE_COUNT),
        )[:rows],
        candidate_card_rule_values=_view_f32_column(
            device_frame,
            layout.candidate_card_rule_values,
            (capacity, CARD_CHOICE_COUNT, 2),
        )[:rows],
        query_indices=_view_i64_column(
            device_frame, layout.query_indices, (capacity,)
        )[:rows],
    )
    action_plan = DeviceActionPlanBatch(
        kind_codes=_i64_vector(
            device_frame, layout.kind_codes, capacity, rows
        ),
        available_counts=_i64_matrix(
            device_frame,
            layout.available_counts,
            capacity,
            rows,
            ACTION_FACE_COUNT,
        ),
        effective_suits=_i64_matrix(
            device_frame,
            layout.effective_suits,
            capacity,
            rows,
            ACTION_FACE_COUNT,
        ),
        same_suit_mask=_bool_matrix(
            device_frame,
            layout.same_suit_mask,
            capacity,
            rows,
            ACTION_FACE_COUNT,
        ),
        off_suit_mask=_bool_matrix(
            device_frame,
            layout.off_suit_mask,
            capacity,
            rows,
            ACTION_FACE_COUNT,
        ),
        pair_face_mask=_bool_matrix(
            device_frame,
            layout.pair_face_mask,
            capacity,
            rows,
            ACTION_FACE_COUNT,
        ),
        min_select=_i64_vector(
            device_frame, layout.min_select, capacity, rows
        ),
        max_select=_i64_vector(
            device_frame, layout.max_select, capacity, rows
        ),
        exact_select=_i64_vector(
            device_frame, layout.exact_select, capacity, rows
        ),
        required_same_suit_count=_i64_vector(
            device_frame,
            layout.required_same_suit_count,
            capacity,
            rows,
        ),
        pair_floor=_i64_vector(
            device_frame, layout.pair_floor, capacity, rows
        ),
        has_tractor=_view_bool_column(
            device_frame, layout.has_tractor, (capacity,)
        )[:rows],
        trace_choice_ids=_view_i64_column(
            device_frame,
            layout.trace_choice_ids,
            (
                capacity,
                MAX_TRACE_COUNT,
                metadata.padded_generation_steps,
            ),
        )[:rows],
        trace_choice_mask=_view_bool_column(
            device_frame,
            layout.trace_choice_mask,
            (
                capacity,
                MAX_TRACE_COUNT,
                metadata.padded_generation_steps,
            ),
        )[:rows],
        trace_lengths=_i64_matrix(
            device_frame,
            layout.trace_lengths,
            capacity,
            rows,
            MAX_TRACE_COUNT,
        ),
        trace_row_mask=_bool_matrix(
            device_frame,
            layout.trace_row_mask,
            capacity,
            rows,
            MAX_TRACE_COUNT,
        ),
        pair_plan_masks=_view_bool_column(
            device_frame,
            layout.pair_plan_masks,
            (capacity, MAX_PAIR_PLAN_COUNT, ACTION_FACE_COUNT),
        )[:rows],
        pair_plan_row_mask=_bool_matrix(
            device_frame,
            layout.pair_plan_row_mask,
            capacity,
            rows,
            MAX_PAIR_PLAN_COUNT,
        ),
    )
    return Ok(
        value=DevicePolicyRequestBatch(
            observation_batch=observation,
            action_plan_batch=action_plan,
            sampling_thresholds=thresholds.value[:rows],
            generation_step_counts=_i64_vector(
                device_frame,
                layout.generation_step_counts,
                capacity,
                rows,
            ),
            policy_versions=metadata.policy_versions,
            padded_generation_steps=metadata.padded_generation_steps,
        )
    )


def sampling_threshold_dtype_for_device(
    device: torch.device,
) -> torch.dtype:
    return torch.float32 if device.type == "mps" else torch.float64


def materialize_sampling_thresholds_for_device(
    *, thresholds: Tensor, device: torch.device
) -> Tensor:
    assert thresholds.device.type == "cpu"
    assert thresholds.dtype == torch.float64
    if device.type != "mps":
        return thresholds.to(device=device)
    valid = (
        torch.isfinite(thresholds)
        & (thresholds >= 0.0)
        & (thresholds < 1.0)
    )
    normalized = torch.where(
        valid, thresholds.clamp(max=_FLOAT32_BELOW_ONE), thresholds
    )
    return normalized.to(dtype=torch.float32, device=device)


def _materialize_sampling_thresholds(
    *,
    device_frame: Tensor,
    host_frame: Tensor | None,
    metadata: PolicyRequestFrameMetadata,
) -> _result.Ok[Tensor] | _result.Rejected:
    shape = (
        metadata.batch_capacity,
        metadata.padded_generation_steps,
    )
    if device_frame.device.type != "mps":
        return Ok(
            value=_view_f64_column(
                device_frame, metadata.layout.sampling_thresholds, shape
            )
        )
    if host_frame is None or host_frame.device.type != "cpu":
        return Rejected(
            reason=(
                "MPS policy request requires host sampling thresholds"
            )
        )
    host_thresholds = _view_f64_column(
        host_frame, metadata.layout.sampling_thresholds, shape
    )
    return Ok(
        value=materialize_sampling_thresholds_for_device(
            thresholds=host_thresholds, device=device_frame.device
        )
    )


def _i64_vector(
    frame: Tensor,
    column: ColumnLayout,
    capacity: int,
    rows: int,
) -> Tensor:
    return _view_i64_column(frame, column, (capacity,))[:rows]


def _i64_matrix(
    frame: Tensor,
    column: ColumnLayout,
    capacity: int,
    rows: int,
    width: int,
) -> Tensor:
    return _view_i64_column(frame, column, (capacity, width))[:rows]


def _bool_matrix(
    frame: Tensor,
    column: ColumnLayout,
    capacity: int,
    rows: int,
    width: int,
) -> Tensor:
    return _view_bool_column(frame, column, (capacity, width))[:rows]


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
    *, device_frame: Tensor, metadata: PolicyRequestFrameMetadata
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
    return _view_column(frame, column).view(torch.int64).reshape(shape)


def _view_f32_column(
    frame: Tensor, column: ColumnLayout, shape: tuple[int, ...]
) -> Tensor:
    return (
        _view_column(frame, column).view(torch.float32).reshape(shape)
    )


def _view_f64_column(
    frame: Tensor, column: ColumnLayout, shape: tuple[int, ...]
) -> Tensor:
    return (
        _view_column(frame, column).view(torch.float64).reshape(shape)
    )


def _view_bool_column(
    frame: Tensor, column: ColumnLayout, shape: tuple[int, ...]
) -> Tensor:
    return _view_column(frame, column).view(torch.bool).reshape(shape)


def _view_column(frame: Tensor, column: ColumnLayout) -> Tensor:
    return frame[column.offset : column.offset + column.total_bytes]


__all__ = (
    "copy_policy_request_host_frame_to_device",
    "materialize_borrowed_policy_request_batch",
    "materialize_policy_request_batch_frame",
    "materialize_policy_request_frame",
    "materialize_sampling_thresholds_for_device",
    "sampling_threshold_dtype_for_device",
)
