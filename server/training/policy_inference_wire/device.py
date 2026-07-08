"""Device-side unpacking for policy inference request wires."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server import result as _result
from server.result import Ok, Rejected
from server.training.feature_schema import NUMERIC_FEATURE_COUNT
from server.training.packed_observation import (
    OBSERVATION_COMPONENT_COUNT,
)
from server.training.semantic_action_plan import DeviceActionPlanBatch
from server.training.semantic_action_plan.spec import ACTION_FACE_COUNT
from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.tensorize import ObservationTensorBatch

from .request import (
    REQUEST_EXACT_SELECT_INDEX,
    REQUEST_HAS_TRACTOR_INDEX,
    REQUEST_HEADER_BYTES,
    REQUEST_HEADER_WORD_COUNT,
    REQUEST_KIND_CODE_INDEX,
    REQUEST_MAX_SELECT_INDEX,
    REQUEST_MIN_SELECT_INDEX,
    REQUEST_PAIR_FLOOR_INDEX,
    REQUEST_REQUIRED_SAME_SUIT_COUNT_INDEX,
    WIRE_MAX_PAIR_PLAN_COUNT,
    WIRE_MAX_TRACE_COUNT,
    RequestSectionOffsets,
    request_section_offsets,
)
from .types import DevicePolicyRequestBatch, PolicyRequestMetadata


@dataclass(slots=True)
class DevicePolicyRequestBuffer:
    """Reusable final tensor layout for staged policy requests."""

    max_observation_tokens: int
    component_ids: Tensor
    numeric_values: Tensor
    numeric_masks: Tensor
    kind_codes: Tensor
    available_counts: Tensor
    effective_suits: Tensor
    same_suit_mask: Tensor
    off_suit_mask: Tensor
    pair_face_mask: Tensor
    min_select: Tensor
    max_select: Tensor
    exact_select: Tensor
    required_same_suit_count: Tensor
    pair_floor: Tensor
    has_tractor: Tensor
    trace_tokens: Tensor
    trace_token_mask: Tensor
    trace_lengths: Tensor
    trace_row_mask: Tensor
    pair_plan_masks: Tensor
    pair_plan_row_mask: Tensor
    sampling_thresholds: Tensor

    def __post_init__(self) -> None:
        assert self.max_observation_tokens > 0
        assert self.component_ids.ndim == 3
        batch_size = int(self.component_ids.shape[0])
        assert batch_size > 0
        assert self.component_ids.shape == (
            batch_size,
            self.max_observation_tokens,
            OBSERVATION_COMPONENT_COUNT,
        )
        assert self.numeric_values.shape == (
            batch_size,
            self.max_observation_tokens,
            NUMERIC_FEATURE_COUNT,
        )
        assert self.numeric_masks.shape == self.numeric_values.shape
        assert self.kind_codes.shape == (batch_size,)
        assert self.available_counts.shape == (
            batch_size,
            ACTION_FACE_COUNT,
        )
        assert self.effective_suits.shape == (
            batch_size,
            ACTION_FACE_COUNT,
        )
        assert self.same_suit_mask.shape == (
            batch_size,
            ACTION_FACE_COUNT,
        )
        assert self.off_suit_mask.shape == (
            batch_size,
            ACTION_FACE_COUNT,
        )
        assert self.pair_face_mask.shape == (
            batch_size,
            ACTION_FACE_COUNT,
        )
        assert self.min_select.shape == (batch_size,)
        assert self.max_select.shape == (batch_size,)
        assert self.exact_select.shape == (batch_size,)
        assert self.required_same_suit_count.shape == (batch_size,)
        assert self.pair_floor.shape == (batch_size,)
        assert self.has_tractor.shape == (batch_size,)
        assert self.trace_tokens.shape == (
            batch_size,
            WIRE_MAX_TRACE_COUNT,
            SEMANTIC_CODEC.max_argument_tokens,
        )
        assert self.trace_token_mask.shape == self.trace_tokens.shape
        assert self.trace_lengths.shape == (
            batch_size,
            WIRE_MAX_TRACE_COUNT,
        )
        assert self.trace_row_mask.shape == self.trace_lengths.shape
        assert self.pair_plan_masks.shape == (
            batch_size,
            WIRE_MAX_PAIR_PLAN_COUNT,
            ACTION_FACE_COUNT,
        )
        assert self.pair_plan_row_mask.shape == (
            batch_size,
            WIRE_MAX_PAIR_PLAN_COUNT,
        )
        assert self.sampling_thresholds.shape == (
            batch_size,
            SEMANTIC_CODEC.max_argument_tokens,
        )

    @property
    def batch_size(self) -> int:
        """Return buffer row capacity."""
        return int(self.component_ids.shape[0])

    @property
    def device(self) -> torch.device:
        """Return the torch device hosting this buffer."""
        return self.component_ids.device


def allocate_device_policy_request_buffer(
    *,
    batch_size: int,
    max_observation_tokens: int,
    device: torch.device,
) -> DevicePolicyRequestBuffer:
    """Allocate reusable final tensors for policy request unpacking."""
    assert batch_size > 0
    assert max_observation_tokens > 0
    return DevicePolicyRequestBuffer(
        max_observation_tokens=max_observation_tokens,
        component_ids=torch.empty(
            (
                batch_size,
                max_observation_tokens,
                OBSERVATION_COMPONENT_COUNT,
            ),
            dtype=torch.long,
            device=device,
        ),
        numeric_values=torch.empty(
            (batch_size, max_observation_tokens, NUMERIC_FEATURE_COUNT),
            dtype=torch.float32,
            device=device,
        ),
        numeric_masks=torch.empty(
            (batch_size, max_observation_tokens, NUMERIC_FEATURE_COUNT),
            dtype=torch.float32,
            device=device,
        ),
        kind_codes=torch.empty(
            (batch_size,), dtype=torch.long, device=device
        ),
        available_counts=torch.empty(
            (batch_size, ACTION_FACE_COUNT),
            dtype=torch.long,
            device=device,
        ),
        effective_suits=torch.empty(
            (batch_size, ACTION_FACE_COUNT),
            dtype=torch.long,
            device=device,
        ),
        same_suit_mask=torch.empty(
            (batch_size, ACTION_FACE_COUNT),
            dtype=torch.bool,
            device=device,
        ),
        off_suit_mask=torch.empty(
            (batch_size, ACTION_FACE_COUNT),
            dtype=torch.bool,
            device=device,
        ),
        pair_face_mask=torch.empty(
            (batch_size, ACTION_FACE_COUNT),
            dtype=torch.bool,
            device=device,
        ),
        min_select=torch.empty(
            (batch_size,), dtype=torch.long, device=device
        ),
        max_select=torch.empty(
            (batch_size,), dtype=torch.long, device=device
        ),
        exact_select=torch.empty(
            (batch_size,), dtype=torch.long, device=device
        ),
        required_same_suit_count=torch.empty(
            (batch_size,), dtype=torch.long, device=device
        ),
        pair_floor=torch.empty(
            (batch_size,), dtype=torch.long, device=device
        ),
        has_tractor=torch.empty(
            (batch_size,), dtype=torch.bool, device=device
        ),
        trace_tokens=torch.empty(
            (
                batch_size,
                WIRE_MAX_TRACE_COUNT,
                SEMANTIC_CODEC.max_argument_tokens,
            ),
            dtype=torch.long,
            device=device,
        ),
        trace_token_mask=torch.empty(
            (
                batch_size,
                WIRE_MAX_TRACE_COUNT,
                SEMANTIC_CODEC.max_argument_tokens,
            ),
            dtype=torch.bool,
            device=device,
        ),
        trace_lengths=torch.empty(
            (batch_size, WIRE_MAX_TRACE_COUNT),
            dtype=torch.long,
            device=device,
        ),
        trace_row_mask=torch.empty(
            (batch_size, WIRE_MAX_TRACE_COUNT),
            dtype=torch.bool,
            device=device,
        ),
        pair_plan_masks=torch.empty(
            (batch_size, WIRE_MAX_PAIR_PLAN_COUNT, ACTION_FACE_COUNT),
            dtype=torch.bool,
            device=device,
        ),
        pair_plan_row_mask=torch.empty(
            (batch_size, WIRE_MAX_PAIR_PLAN_COUNT),
            dtype=torch.bool,
            device=device,
        ),
        sampling_thresholds=torch.empty(
            (batch_size, SEMANTIC_CODEC.max_argument_tokens),
            dtype=torch.float64,
            device=device,
        ),
    )


def unpack_policy_request_batch_into(
    *,
    device_bytes: Tensor,
    metadata: tuple[PolicyRequestMetadata, ...],
    output: DevicePolicyRequestBuffer,
) -> _result.Ok[DevicePolicyRequestBatch] | _result.Rejected:
    """Unpack staged request bytes into a reusable tensor buffer."""
    validate_result = _validate_unpack_inputs(
        device_bytes=device_bytes,
        metadata=metadata,
        output=output,
    )
    if isinstance(validate_result, Rejected):
        return validate_result
    row_count = len(metadata)
    header_words = (
        device_bytes[:, :REQUEST_HEADER_BYTES]
        .contiguous()
        .view(torch.int64)
        .reshape(row_count, REQUEST_HEADER_WORD_COUNT)
    )
    _copy_header_columns(
        header_words=header_words,
        output=output,
        row_count=row_count,
    )
    offsets = _canonical_offsets(output.max_observation_tokens)
    for row_index in range(row_count):
        _copy_request_row(
            row=device_bytes[row_index],
            offsets=offsets,
            row_index=row_index,
            output=output,
        )
    return Ok(
        value=DevicePolicyRequestBatch(
            observation_batch=ObservationTensorBatch(
                component_ids=output.component_ids[:row_count],
                numeric_values=output.numeric_values[:row_count],
                numeric_masks=output.numeric_masks[:row_count],
            ),
            action_plan_batch=DeviceActionPlanBatch(
                kind_codes=output.kind_codes[:row_count],
                available_counts=output.available_counts[:row_count],
                effective_suits=output.effective_suits[:row_count],
                same_suit_mask=output.same_suit_mask[:row_count],
                off_suit_mask=output.off_suit_mask[:row_count],
                pair_face_mask=output.pair_face_mask[:row_count],
                min_select=output.min_select[:row_count],
                max_select=output.max_select[:row_count],
                exact_select=output.exact_select[:row_count],
                required_same_suit_count=(
                    output.required_same_suit_count[:row_count]
                ),
                pair_floor=output.pair_floor[:row_count],
                has_tractor=output.has_tractor[:row_count],
                trace_tokens=output.trace_tokens[:row_count],
                trace_token_mask=output.trace_token_mask[:row_count],
                trace_lengths=output.trace_lengths[:row_count],
                trace_row_mask=output.trace_row_mask[:row_count],
                pair_plan_masks=output.pair_plan_masks[:row_count],
                pair_plan_row_mask=(
                    output.pair_plan_row_mask[:row_count]
                ),
            ),
            sampling_thresholds=output.sampling_thresholds[:row_count],
            policy_versions=tuple(
                item.policy_version for item in metadata
            ),
        )
    )


def _validate_unpack_inputs(
    *,
    device_bytes: Tensor,
    metadata: tuple[PolicyRequestMetadata, ...],
    output: DevicePolicyRequestBuffer,
) -> _result.Ok[None] | _result.Rejected:
    if device_bytes.dtype != torch.uint8:
        return Rejected(
            reason="policy request wire tensor must be uint8"
        )
    if device_bytes.ndim != 2:
        return Rejected(reason="policy request wire tensor must be 2D")
    if device_bytes.device != output.device:
        return Rejected(reason="policy request wire device mismatch")
    row_count = int(device_bytes.shape[0])
    if row_count != len(metadata):
        return Rejected(reason="policy request metadata batch mismatch")
    if row_count <= 0:
        return Rejected(reason="policy request batch is empty")
    if row_count > output.batch_size:
        return Rejected(reason="policy request batch exceeds buffer")
    expected_bytes = _canonical_offsets(
        output.max_observation_tokens
    ).total_bytes
    if int(device_bytes.shape[1]) != expected_bytes:
        return Rejected(reason="policy request wire layout mismatch")
    for item in metadata:
        if item.byte_count != expected_bytes:
            return Rejected(
                reason="policy request wire length mismatch"
            )
        if item.token_count != output.max_observation_tokens:
            return Rejected(
                reason="policy request observation layout mismatch"
            )
        if item.trace_count != WIRE_MAX_TRACE_COUNT:
            return Rejected(
                reason="policy request trace layout mismatch"
            )
        if item.trace_steps != SEMANTIC_CODEC.max_argument_tokens:
            return Rejected(
                reason="policy request trace width mismatch"
            )
        if item.pair_plan_count != WIRE_MAX_PAIR_PLAN_COUNT:
            return Rejected(
                reason="policy request pair plan layout mismatch"
            )
    return Ok(value=None)


def _canonical_offsets(
    max_observation_tokens: int,
) -> RequestSectionOffsets:
    return request_section_offsets(
        token_count=max_observation_tokens,
        trace_count=WIRE_MAX_TRACE_COUNT,
        trace_steps=SEMANTIC_CODEC.max_argument_tokens,
        pair_plan_count=WIRE_MAX_PAIR_PLAN_COUNT,
    )


def _copy_header_columns(
    *,
    header_words: Tensor,
    output: DevicePolicyRequestBuffer,
    row_count: int,
) -> None:
    output.kind_codes[:row_count].copy_(
        header_words[:, REQUEST_KIND_CODE_INDEX]
    )
    output.min_select[:row_count].copy_(
        header_words[:, REQUEST_MIN_SELECT_INDEX]
    )
    output.max_select[:row_count].copy_(
        header_words[:, REQUEST_MAX_SELECT_INDEX]
    )
    output.exact_select[:row_count].copy_(
        header_words[:, REQUEST_EXACT_SELECT_INDEX]
    )
    output.required_same_suit_count[:row_count].copy_(
        header_words[:, REQUEST_REQUIRED_SAME_SUIT_COUNT_INDEX]
    )
    output.pair_floor[:row_count].copy_(
        header_words[:, REQUEST_PAIR_FLOOR_INDEX]
    )
    output.has_tractor[:row_count].copy_(
        header_words[:, REQUEST_HAS_TRACTOR_INDEX] != 0
    )


def _copy_request_row(
    *,
    row: Tensor,
    offsets: RequestSectionOffsets,
    row_index: int,
    output: DevicePolicyRequestBuffer,
) -> None:
    output.component_ids[row_index].copy_(
        _read_i64_matrix(
            row,
            offsets.component_ids,
            output.max_observation_tokens,
            OBSERVATION_COMPONENT_COUNT,
        )
    )
    output.numeric_values[row_index].copy_(
        _read_f32_matrix(
            row,
            offsets.numeric_values,
            output.max_observation_tokens,
            NUMERIC_FEATURE_COUNT,
        )
    )
    output.numeric_masks[row_index].copy_(
        _read_f32_matrix(
            row,
            offsets.numeric_masks,
            output.max_observation_tokens,
            NUMERIC_FEATURE_COUNT,
        )
    )
    output.available_counts[row_index].copy_(
        _read_i64_vector(
            row, offsets.available_counts, ACTION_FACE_COUNT
        )
    )
    output.effective_suits[row_index].copy_(
        _read_i64_vector(
            row, offsets.effective_suits, ACTION_FACE_COUNT
        )
    )
    output.same_suit_mask[row_index].copy_(
        _read_bool_vector(
            row, offsets.same_suit_mask, ACTION_FACE_COUNT
        )
    )
    output.off_suit_mask[row_index].copy_(
        _read_bool_vector(row, offsets.off_suit_mask, ACTION_FACE_COUNT)
    )
    output.pair_face_mask[row_index].copy_(
        _read_bool_vector(
            row, offsets.pair_face_mask, ACTION_FACE_COUNT
        )
    )
    output.trace_tokens[row_index].copy_(
        _read_i64_matrix(
            row,
            offsets.trace_tokens,
            WIRE_MAX_TRACE_COUNT,
            SEMANTIC_CODEC.max_argument_tokens,
        )
    )
    output.trace_token_mask[row_index].copy_(
        _read_bool_matrix(
            row,
            offsets.trace_token_mask,
            WIRE_MAX_TRACE_COUNT,
            SEMANTIC_CODEC.max_argument_tokens,
        )
    )
    output.trace_lengths[row_index].copy_(
        _read_i64_vector(
            row, offsets.trace_lengths, WIRE_MAX_TRACE_COUNT
        )
    )
    output.trace_row_mask[row_index].copy_(
        _read_bool_vector(
            row, offsets.trace_row_mask, WIRE_MAX_TRACE_COUNT
        )
    )
    output.pair_plan_masks[row_index].copy_(
        _read_bool_matrix(
            row,
            offsets.pair_plan_masks,
            WIRE_MAX_PAIR_PLAN_COUNT,
            ACTION_FACE_COUNT,
        )
    )
    output.pair_plan_row_mask[row_index].copy_(
        _read_bool_vector(
            row,
            offsets.pair_plan_row_mask,
            WIRE_MAX_PAIR_PLAN_COUNT,
        )
    )
    output.sampling_thresholds[row_index].copy_(
        _read_f64_vector(
            row,
            offsets.sampling_thresholds,
            SEMANTIC_CODEC.max_argument_tokens,
        )
    )


def _read_i64_vector(row: Tensor, offset: int, length: int) -> Tensor:
    return row[offset : offset + length * 8].view(torch.int64)


def _read_f32_vector(row: Tensor, offset: int, length: int) -> Tensor:
    return row[offset : offset + length * 4].view(torch.float32)


def _read_f64_vector(row: Tensor, offset: int, length: int) -> Tensor:
    return row[offset : offset + length * 8].view(torch.float64)


def _read_bool_vector(row: Tensor, offset: int, length: int) -> Tensor:
    return row[offset : offset + length] != 0


def _read_i64_matrix(
    row: Tensor, offset: int, rows: int, columns: int
) -> Tensor:
    return _read_i64_vector(row, offset, rows * columns).reshape(
        rows, columns
    )


def _read_f32_matrix(
    row: Tensor, offset: int, rows: int, columns: int
) -> Tensor:
    return _read_f32_vector(row, offset, rows * columns).reshape(
        rows, columns
    )


def _read_bool_matrix(
    row: Tensor, offset: int, rows: int, columns: int
) -> Tensor:
    return _read_bool_vector(row, offset, rows * columns).reshape(
        rows, columns
    )
