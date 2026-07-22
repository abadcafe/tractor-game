"""Columnar wire schema for typed policy inference requests."""

from __future__ import annotations

import struct
from dataclasses import dataclass

from server.training.observation_structure import STRUCTURE_AXIS_COUNT
from server.training.packed_observation import (
    MAX_LOSSLESS_OBSERVATION_TOKENS,
)
from server.training.semantic_action_plan.spec import (
    ACTION_FACE_COUNT,
    MAX_PAIR_PLAN_COUNT,
    MAX_TRACE_COUNT,
)
from server.training.semantic_actions.choices import CARD_CHOICE_COUNT
from server.training.tokenization.encoding_schema import CATEGORY_COUNT

REQUEST_BATCH_MAGIC = 0x5452504F4C495145

I64 = struct.Struct("<q")
F32 = struct.Struct("<f")
F64 = struct.Struct("<d")

HEADER_WORD_COUNT = 8
HEADER_BYTES = HEADER_WORD_COUNT * I64.size

MAGIC_INDEX = 0
TOTAL_BYTES_INDEX = 1
ROW_COUNT_INDEX = 2
BATCH_CAPACITY_INDEX = 3
OBSERVATION_TOKEN_CAPACITY_INDEX = 4
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
    """Exact typed-column layout for one fixed-capacity frame."""

    batch_capacity: int
    observation_token_capacity: int
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
    category_ids: ColumnLayout
    scalar_values: ColumnLayout
    card_rule_values: ColumnLayout
    encoded_structure_coordinates: ColumnLayout
    candidate_category_ids: ColumnLayout
    candidate_counts: ColumnLayout
    candidate_card_rule_values: ColumnLayout
    query_indices: ColumnLayout
    available_counts: ColumnLayout
    effective_suits: ColumnLayout
    same_suit_mask: ColumnLayout
    off_suit_mask: ColumnLayout
    pair_face_mask: ColumnLayout
    trace_choice_ids: ColumnLayout
    trace_choice_mask: ColumnLayout
    trace_lengths: ColumnLayout
    trace_row_mask: ColumnLayout
    pair_plan_masks: ColumnLayout
    pair_plan_row_mask: ColumnLayout
    sampling_thresholds: ColumnLayout
    total_bytes: int

    def __post_init__(self) -> None:
        assert self.batch_capacity > 0
        assert (
            0
            < self.observation_token_capacity
            <= MAX_LOSSLESS_OBSERVATION_TOKENS
        )
        assert self.padded_generation_steps > 0
        assert self.total_bytes > HEADER_BYTES


def policy_request_batch_layout(
    *,
    batch_capacity: int,
    observation_token_capacity: int,
    padded_generation_steps: int,
) -> PolicyRequestBatchLayout:
    """Return the sole valid request layout for these dimensions."""
    assert batch_capacity > 0
    assert (
        0
        < observation_token_capacity
        <= MAX_LOSSLESS_OBSERVATION_TOKENS
    )
    assert padded_generation_steps > 0
    builder = _LayoutBuilder(offset=HEADER_BYTES)

    def column(row_bytes: int) -> ColumnLayout:
        return builder.column(batch_capacity, row_bytes)

    route_worker_indices = column(I64.size)
    route_request_ids = column(I64.size)
    policy_versions = column(I64.size)
    generation_step_counts = column(I64.size)
    kind_codes = column(I64.size)
    min_select = column(I64.size)
    max_select = column(I64.size)
    exact_select = column(I64.size)
    required_same_suit_count = column(I64.size)
    pair_floor = column(I64.size)
    has_tractor = column(1)
    category_ids = column(
        observation_token_capacity * CATEGORY_COUNT * I64.size
    )
    scalar_values = column(observation_token_capacity * F32.size)
    card_rule_values = column(observation_token_capacity * 2 * F32.size)
    encoded_structure_coordinates = column(
        observation_token_capacity * STRUCTURE_AXIS_COUNT * I64.size
    )
    candidate_category_ids = column(CARD_CHOICE_COUNT * 3 * I64.size)
    candidate_counts = column(CARD_CHOICE_COUNT * F32.size)
    candidate_card_rule_values = column(
        CARD_CHOICE_COUNT * 2 * F32.size
    )
    query_indices = column(I64.size)
    available_counts = column(ACTION_FACE_COUNT * I64.size)
    effective_suits = column(ACTION_FACE_COUNT * I64.size)
    same_suit_mask = column(ACTION_FACE_COUNT)
    off_suit_mask = column(ACTION_FACE_COUNT)
    pair_face_mask = column(ACTION_FACE_COUNT)
    trace_choice_ids = column(
        MAX_TRACE_COUNT * padded_generation_steps * I64.size
    )
    trace_choice_mask = column(
        MAX_TRACE_COUNT * padded_generation_steps
    )
    trace_lengths = column(MAX_TRACE_COUNT * I64.size)
    trace_row_mask = column(MAX_TRACE_COUNT)
    pair_plan_masks = column(MAX_PAIR_PLAN_COUNT * ACTION_FACE_COUNT)
    pair_plan_row_mask = column(MAX_PAIR_PLAN_COUNT)
    sampling_thresholds = column(padded_generation_steps * F64.size)
    return PolicyRequestBatchLayout(
        batch_capacity=batch_capacity,
        observation_token_capacity=observation_token_capacity,
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
        category_ids=category_ids,
        scalar_values=scalar_values,
        card_rule_values=card_rule_values,
        encoded_structure_coordinates=encoded_structure_coordinates,
        candidate_category_ids=candidate_category_ids,
        candidate_counts=candidate_counts,
        candidate_card_rule_values=candidate_card_rule_values,
        query_indices=query_indices,
        available_counts=available_counts,
        effective_suits=effective_suits,
        same_suit_mask=same_suit_mask,
        off_suit_mask=off_suit_mask,
        pair_face_mask=pair_face_mask,
        trace_choice_ids=trace_choice_ids,
        trace_choice_mask=trace_choice_mask,
        trace_lengths=trace_lengths,
        trace_row_mask=trace_row_mask,
        pair_plan_masks=pair_plan_masks,
        pair_plan_row_mask=pair_plan_row_mask,
        sampling_thresholds=sampling_thresholds,
        total_bytes=builder.offset,
    )


def max_policy_request_batch_frame_bytes(
    *, batch_capacity: int, padded_generation_steps: int
) -> int:
    """Return bytes for a lossless worst-case request frame."""
    return policy_request_batch_layout(
        batch_capacity=batch_capacity,
        observation_token_capacity=MAX_LOSSLESS_OBSERVATION_TOKENS,
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
        layout = ColumnLayout(
            offset=self.offset,
            row_bytes=row_bytes,
            total_bytes=batch_capacity * row_bytes,
        )
        self.offset += layout.total_bytes
        return layout


def _alignment_for_row(row_bytes: int) -> int:
    if row_bytes % I64.size == 0:
        return I64.size
    if row_bytes % F32.size == 0:
        return F32.size
    return 1


def _align(offset: int, alignment: int) -> int:
    remainder = offset % alignment
    return offset if remainder == 0 else offset + alignment - remainder


__all__ = (
    "F32",
    "F64",
    "I64",
    "MAX_PAIR_PLAN_COUNT",
    "MAX_TRACE_COUNT",
    "ColumnLayout",
    "PolicyRequestBatchLayout",
    "max_policy_request_batch_frame_bytes",
    "policy_request_batch_layout",
)
