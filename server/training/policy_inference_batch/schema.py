"""Columnar policy inference request batch schema."""

from __future__ import annotations

import struct
from dataclasses import dataclass

from server.training.feature_schema import NUMERIC_FEATURE_COUNT
from server.training.packed_observation import (
    OBSERVATION_COMPONENT_COUNT,
)
from server.training.semantic_action_plan.spec import (
    ACTION_FACE_COUNT,
    MAX_PAIR_PLAN_COUNT,
    MAX_TRACE_COUNT,
)

REQUEST_BATCH_MAGIC = 0x5452504F4C495143

I64 = struct.Struct("<q")
F32 = struct.Struct("<f")
F64 = struct.Struct("<d")

HEADER_WORD_COUNT = 8
HEADER_BYTES = HEADER_WORD_COUNT * I64.size

MAGIC_INDEX = 0
TOTAL_BYTES_INDEX = 1
ROW_COUNT_INDEX = 2
BATCH_CAPACITY_INDEX = 3
MAX_OBSERVATION_TOKENS_INDEX = 4
TRACE_COUNT_INDEX = 5
PADDED_GENERATION_STEPS_INDEX = 6
PAIR_PLAN_COUNT_INDEX = 7


@dataclass(frozen=True, slots=True)
class ColumnLayout:
    """One row-major column range inside a request frame."""

    offset: int
    row_bytes: int
    total_bytes: int

    def __post_init__(self) -> None:
        assert self.offset >= HEADER_BYTES
        assert self.row_bytes > 0
        assert self.total_bytes > 0


@dataclass(frozen=True, slots=True)
class PolicyRequestBatchLayout:
    """Byte layout for one fixed-capacity request batch frame."""

    batch_capacity: int
    max_observation_tokens: int
    padded_generation_steps: int
    route_worker_indices: ColumnLayout
    route_request_ids: ColumnLayout
    policy_versions: ColumnLayout
    generation_step_counts: ColumnLayout
    kind_codes: ColumnLayout
    min_select: ColumnLayout
    max_select: ColumnLayout
    exact_select: ColumnLayout
    required_same_suit_count: ColumnLayout
    pair_floor: ColumnLayout
    has_tractor: ColumnLayout
    component_ids: ColumnLayout
    numeric_values: ColumnLayout
    numeric_masks: ColumnLayout
    available_counts: ColumnLayout
    effective_suits: ColumnLayout
    same_suit_mask: ColumnLayout
    off_suit_mask: ColumnLayout
    pair_face_mask: ColumnLayout
    trace_tokens: ColumnLayout
    trace_token_mask: ColumnLayout
    trace_lengths: ColumnLayout
    trace_row_mask: ColumnLayout
    pair_plan_masks: ColumnLayout
    pair_plan_row_mask: ColumnLayout
    sampling_thresholds: ColumnLayout
    total_bytes: int

    def __post_init__(self) -> None:
        assert self.batch_capacity > 0
        assert self.max_observation_tokens > 0
        assert self.padded_generation_steps > 0
        assert self.total_bytes > HEADER_BYTES

    def row_columns(self) -> tuple[ColumnLayout, ...]:
        """Return all per-row columns in frame order."""
        return (
            self.route_worker_indices,
            self.route_request_ids,
            self.policy_versions,
            self.generation_step_counts,
            self.kind_codes,
            self.min_select,
            self.max_select,
            self.exact_select,
            self.required_same_suit_count,
            self.pair_floor,
            self.has_tractor,
            self.component_ids,
            self.numeric_values,
            self.numeric_masks,
            self.available_counts,
            self.effective_suits,
            self.same_suit_mask,
            self.off_suit_mask,
            self.pair_face_mask,
            self.trace_tokens,
            self.trace_token_mask,
            self.trace_lengths,
            self.trace_row_mask,
            self.pair_plan_masks,
            self.pair_plan_row_mask,
            self.sampling_thresholds,
        )


def policy_request_batch_layout(
    *,
    batch_capacity: int,
    max_observation_tokens: int,
    padded_generation_steps: int,
) -> PolicyRequestBatchLayout:
    """Return the canonical columnar layout for one frame capacity."""
    assert batch_capacity > 0
    assert max_observation_tokens > 0
    assert padded_generation_steps > 0
    builder = _LayoutBuilder(offset=HEADER_BYTES)
    route_worker_indices = builder.column(batch_capacity, I64.size)
    route_request_ids = builder.column(batch_capacity, I64.size)
    policy_versions = builder.column(batch_capacity, I64.size)
    generation_step_counts = builder.column(batch_capacity, I64.size)
    kind_codes = builder.column(batch_capacity, I64.size)
    min_select = builder.column(batch_capacity, I64.size)
    max_select = builder.column(batch_capacity, I64.size)
    exact_select = builder.column(batch_capacity, I64.size)
    required_same_suit_count = builder.column(batch_capacity, I64.size)
    pair_floor = builder.column(batch_capacity, I64.size)
    has_tractor = builder.column(batch_capacity, 1)
    component_ids = builder.column(
        batch_capacity,
        max_observation_tokens * OBSERVATION_COMPONENT_COUNT * I64.size,
    )
    numeric_values = builder.column(
        batch_capacity,
        max_observation_tokens * NUMERIC_FEATURE_COUNT * F32.size,
    )
    numeric_masks = builder.column(
        batch_capacity,
        max_observation_tokens * NUMERIC_FEATURE_COUNT * F32.size,
    )
    available_counts = builder.column(
        batch_capacity, ACTION_FACE_COUNT * I64.size
    )
    effective_suits = builder.column(
        batch_capacity, ACTION_FACE_COUNT * I64.size
    )
    same_suit_mask = builder.column(batch_capacity, ACTION_FACE_COUNT)
    off_suit_mask = builder.column(batch_capacity, ACTION_FACE_COUNT)
    pair_face_mask = builder.column(batch_capacity, ACTION_FACE_COUNT)
    trace_tokens = builder.column(
        batch_capacity,
        MAX_TRACE_COUNT * padded_generation_steps * I64.size,
    )
    trace_token_mask = builder.column(
        batch_capacity,
        MAX_TRACE_COUNT * padded_generation_steps,
    )
    trace_lengths = builder.column(
        batch_capacity, MAX_TRACE_COUNT * I64.size
    )
    trace_row_mask = builder.column(batch_capacity, MAX_TRACE_COUNT)
    pair_plan_masks = builder.column(
        batch_capacity, MAX_PAIR_PLAN_COUNT * ACTION_FACE_COUNT
    )
    pair_plan_row_mask = builder.column(
        batch_capacity, MAX_PAIR_PLAN_COUNT
    )
    sampling_thresholds = builder.column(
        batch_capacity, padded_generation_steps * F64.size
    )
    return PolicyRequestBatchLayout(
        batch_capacity=batch_capacity,
        max_observation_tokens=max_observation_tokens,
        padded_generation_steps=padded_generation_steps,
        route_worker_indices=route_worker_indices,
        route_request_ids=route_request_ids,
        policy_versions=policy_versions,
        generation_step_counts=generation_step_counts,
        kind_codes=kind_codes,
        min_select=min_select,
        max_select=max_select,
        exact_select=exact_select,
        required_same_suit_count=required_same_suit_count,
        pair_floor=pair_floor,
        has_tractor=has_tractor,
        component_ids=component_ids,
        numeric_values=numeric_values,
        numeric_masks=numeric_masks,
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
        total_bytes=builder.offset,
    )


def max_policy_request_batch_frame_bytes(
    *,
    batch_capacity: int,
    max_observation_tokens: int,
    padded_generation_steps: int,
) -> int:
    """Return the exact bytes for one fixed-capacity request frame."""
    return policy_request_batch_layout(
        batch_capacity=batch_capacity,
        max_observation_tokens=max_observation_tokens,
        padded_generation_steps=padded_generation_steps,
    ).total_bytes


@dataclass(slots=True)
class _LayoutBuilder:
    offset: int

    def column(
        self, batch_capacity: int, row_bytes: int
    ) -> ColumnLayout:
        alignment = _alignment_for_row(row_bytes)
        self.offset = _align(self.offset, alignment)
        total_bytes = batch_capacity * row_bytes
        layout = ColumnLayout(
            offset=self.offset,
            row_bytes=row_bytes,
            total_bytes=total_bytes,
        )
        self.offset += total_bytes
        return layout


def _alignment_for_row(row_bytes: int) -> int:
    if row_bytes % I64.size == 0:
        return I64.size
    if row_bytes % F32.size == 0:
        return F32.size
    return 1


def _align(offset: int, alignment: int) -> int:
    remainder = offset % alignment
    if remainder == 0:
        return offset
    return offset + alignment - remainder
