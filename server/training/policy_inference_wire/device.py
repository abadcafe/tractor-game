"""Device-side unpacking for policy inference request wires."""

from __future__ import annotations

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
    RequestSectionOffsets,
    request_section_offsets,
)
from .types import DevicePolicyRequestBatch, PolicyRequestMetadata


def device_policy_request_batch_from_wire(
    *,
    device_bytes: Tensor,
    metadata: tuple[PolicyRequestMetadata, ...],
    max_observation_tokens: int,
) -> _result.Ok[DevicePolicyRequestBatch] | _result.Rejected:
    """Unpack staged request wire bytes into model input tensors."""
    assert max_observation_tokens > 0
    if device_bytes.dtype != torch.uint8:
        return Rejected(
            reason="policy request wire tensor must be uint8"
        )
    if device_bytes.ndim != 2:
        return Rejected(reason="policy request wire tensor must be 2D")
    batch_size = int(device_bytes.shape[0])
    if batch_size != len(metadata):
        return Rejected(reason="policy request metadata batch mismatch")
    if batch_size <= 0:
        return Rejected(reason="policy request batch is empty")
    for item in metadata:
        if item.token_count > max_observation_tokens:
            return Rejected(
                reason="policy request observation exceeds token budget"
            )
    header_words = (
        device_bytes[:, :REQUEST_HEADER_BYTES]
        .contiguous()
        .view(torch.int64)
        .reshape(batch_size, REQUEST_HEADER_WORD_COUNT)
    )
    max_token_count = max(item.token_count for item in metadata)
    max_trace_count = max(item.trace_count for item in metadata)
    max_trace_steps = max(item.trace_steps for item in metadata)
    max_pair_plan_count = max(item.pair_plan_count for item in metadata)
    component_rows: list[Tensor] = []
    numeric_value_rows: list[Tensor] = []
    numeric_mask_rows: list[Tensor] = []
    available_counts: list[Tensor] = []
    effective_suits: list[Tensor] = []
    same_suit_mask: list[Tensor] = []
    off_suit_mask: list[Tensor] = []
    pair_face_mask: list[Tensor] = []
    trace_tokens: list[Tensor] = []
    trace_token_mask: list[Tensor] = []
    trace_lengths: list[Tensor] = []
    trace_row_mask: list[Tensor] = []
    pair_plan_masks: list[Tensor] = []
    pair_plan_row_mask: list[Tensor] = []
    sampling_thresholds: list[Tensor] = []
    for row_index, item in enumerate(metadata):
        row = device_bytes[row_index]
        offsets = request_section_offsets(
            token_count=item.token_count,
            trace_count=item.trace_count,
            trace_steps=item.trace_steps,
            pair_plan_count=item.pair_plan_count,
        )
        component_rows.append(
            _pad_matrix(
                _read_i64_matrix(
                    row,
                    offsets.component_ids,
                    item.token_count,
                    OBSERVATION_COMPONENT_COUNT,
                ),
                rows=max_token_count,
                columns=OBSERVATION_COMPONENT_COUNT,
            )
        )
        numeric_value_rows.append(
            _pad_matrix(
                _read_f32_matrix(
                    row,
                    offsets.numeric_values,
                    item.token_count,
                    NUMERIC_FEATURE_COUNT,
                ),
                rows=max_token_count,
                columns=NUMERIC_FEATURE_COUNT,
            )
        )
        numeric_mask_rows.append(
            _pad_matrix(
                _read_f32_matrix(
                    row,
                    offsets.numeric_masks,
                    item.token_count,
                    NUMERIC_FEATURE_COUNT,
                ),
                rows=max_token_count,
                columns=NUMERIC_FEATURE_COUNT,
            )
        )
        _append_action_sections(
            row=row,
            offsets=offsets,
            item=item,
            max_trace_count=max_trace_count,
            max_trace_steps=max_trace_steps,
            max_pair_plan_count=max_pair_plan_count,
            available_counts=available_counts,
            effective_suits=effective_suits,
            same_suit_mask=same_suit_mask,
            off_suit_mask=off_suit_mask,
            pair_face_mask=pair_face_mask,
            trace_tokens=trace_tokens,
            trace_token_mask=trace_token_mask,
            trace_lengths=trace_lengths,
            trace_row_mask=trace_row_mask,
            pair_plan_masks=pair_plan_masks,
            pair_plan_row_mask=pair_plan_row_mask,
            sampling_thresholds=sampling_thresholds,
        )
    return Ok(
        value=DevicePolicyRequestBatch(
            observation_batch=ObservationTensorBatch(
                component_ids=torch.stack(component_rows, dim=0),
                numeric_values=torch.stack(numeric_value_rows, dim=0),
                numeric_masks=torch.stack(numeric_mask_rows, dim=0),
            ),
            action_plan_batch=DeviceActionPlanBatch(
                kind_codes=header_words[:, REQUEST_KIND_CODE_INDEX],
                available_counts=torch.stack(available_counts, dim=0),
                effective_suits=torch.stack(effective_suits, dim=0),
                same_suit_mask=torch.stack(same_suit_mask, dim=0),
                off_suit_mask=torch.stack(off_suit_mask, dim=0),
                pair_face_mask=torch.stack(pair_face_mask, dim=0),
                min_select=header_words[:, REQUEST_MIN_SELECT_INDEX],
                max_select=header_words[:, REQUEST_MAX_SELECT_INDEX],
                exact_select=header_words[
                    :, REQUEST_EXACT_SELECT_INDEX
                ],
                required_same_suit_count=header_words[
                    :, REQUEST_REQUIRED_SAME_SUIT_COUNT_INDEX
                ],
                pair_floor=header_words[:, REQUEST_PAIR_FLOOR_INDEX],
                has_tractor=(
                    header_words[:, REQUEST_HAS_TRACTOR_INDEX] != 0
                ),
                trace_tokens=torch.stack(trace_tokens, dim=0),
                trace_token_mask=torch.stack(trace_token_mask, dim=0),
                trace_lengths=torch.stack(trace_lengths, dim=0),
                trace_row_mask=torch.stack(trace_row_mask, dim=0),
                pair_plan_masks=torch.stack(pair_plan_masks, dim=0),
                pair_plan_row_mask=torch.stack(
                    pair_plan_row_mask, dim=0
                ),
            ),
            sampling_thresholds=torch.stack(sampling_thresholds, dim=0),
            policy_versions=tuple(
                item.policy_version for item in metadata
            ),
        )
    )


def _append_action_sections(
    *,
    row: Tensor,
    offsets: RequestSectionOffsets,
    item: PolicyRequestMetadata,
    max_trace_count: int,
    max_trace_steps: int,
    max_pair_plan_count: int,
    available_counts: list[Tensor],
    effective_suits: list[Tensor],
    same_suit_mask: list[Tensor],
    off_suit_mask: list[Tensor],
    pair_face_mask: list[Tensor],
    trace_tokens: list[Tensor],
    trace_token_mask: list[Tensor],
    trace_lengths: list[Tensor],
    trace_row_mask: list[Tensor],
    pair_plan_masks: list[Tensor],
    pair_plan_row_mask: list[Tensor],
    sampling_thresholds: list[Tensor],
) -> None:
    available_counts.append(
        _read_i64_vector(
            row, offsets.available_counts, ACTION_FACE_COUNT
        )
    )
    effective_suits.append(
        _read_i64_vector(
            row, offsets.effective_suits, ACTION_FACE_COUNT
        )
    )
    same_suit_mask.append(
        _read_bool_vector(
            row, offsets.same_suit_mask, ACTION_FACE_COUNT
        )
    )
    off_suit_mask.append(
        _read_bool_vector(row, offsets.off_suit_mask, ACTION_FACE_COUNT)
    )
    pair_face_mask.append(
        _read_bool_vector(
            row, offsets.pair_face_mask, ACTION_FACE_COUNT
        )
    )
    trace_tokens.append(
        _pad_matrix(
            _read_i64_matrix(
                row,
                offsets.trace_tokens,
                item.trace_count,
                item.trace_steps,
            ),
            rows=max_trace_count,
            columns=max_trace_steps,
        )
    )
    trace_token_mask.append(
        _pad_matrix(
            _read_bool_matrix(
                row,
                offsets.trace_token_mask,
                item.trace_count,
                item.trace_steps,
            ),
            rows=max_trace_count,
            columns=max_trace_steps,
        )
    )
    trace_lengths.append(
        _pad_vector(
            _read_i64_vector(
                row, offsets.trace_lengths, item.trace_count
            ),
            length=max_trace_count,
        )
    )
    trace_row_mask.append(
        _pad_vector(
            _read_bool_vector(
                row, offsets.trace_row_mask, item.trace_count
            ),
            length=max_trace_count,
        )
    )
    pair_plan_masks.append(
        _pad_matrix(
            _read_bool_matrix(
                row,
                offsets.pair_plan_masks,
                item.pair_plan_count,
                ACTION_FACE_COUNT,
            ),
            rows=max_pair_plan_count,
            columns=ACTION_FACE_COUNT,
        )
    )
    pair_plan_row_mask.append(
        _pad_vector(
            _read_bool_vector(
                row,
                offsets.pair_plan_row_mask,
                item.pair_plan_count,
            ),
            length=max_pair_plan_count,
        )
    )
    sampling_thresholds.append(
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


def _pad_vector(values: Tensor, *, length: int) -> Tensor:
    assert values.ndim == 1
    if int(values.shape[0]) == length:
        return values
    result = values.new_zeros((length,))
    result[: int(values.shape[0])] = values
    return result


def _pad_matrix(values: Tensor, *, rows: int, columns: int) -> Tensor:
    assert values.ndim == 2
    assert int(values.shape[1]) == columns
    if int(values.shape[0]) == rows:
        return values
    result = values.new_zeros((rows, columns))
    result[: int(values.shape[0]), :] = values
    return result
