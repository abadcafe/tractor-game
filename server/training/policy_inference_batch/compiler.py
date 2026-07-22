"""Compile policy requests into the strict columnar wire schema."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from server.foundation import result as _result
from server.foundation.result import Ok, Rejected
from server.training.packed_observation import (
    MAX_LOSSLESS_OBSERVATION_TOKENS,
    PackedObservation,
    pack_observation,
    padded_packed_observation,
)
from server.training.policy_inference_batch.frame import (
    initialize_policy_request_frame,
)
from server.training.policy_inference_batch.schema import (
    I64,
    MAX_PAIR_PLAN_COUNT,
    MAX_TRACE_COUNT,
    ColumnLayout,
    PolicyRequestBatchLayout,
    max_policy_request_batch_frame_bytes,
    policy_request_batch_layout,
)
from server.training.policy_inference_batch.types import (
    BorrowedPolicyRequestBatch,
    PolicyRequestFrameMetadata,
    PolicyRequestInput,
    PolicyRequestRoute,
    PolicyRequestWireFrame,
)
from server.training.sampling import (
    PolicyDecisionKey,
    policy_choice_threshold,
)
from server.training.semantic_action_plan import (
    compile_legal_action_frame,
)
from server.training.semantic_action_plan.frame import (
    ActionPlanFrame,
    action_plan_generation_step_count,
)
from server.training.semantic_action_plan.spec import ACTION_FACE_COUNT
from server.training.semantic_actions.choices import MAX_ACTION_STEPS


@dataclass(frozen=True, slots=True)
class _PlannedPolicyRequest:
    route: PolicyRequestRoute
    policy_version: int
    decision_key: PolicyDecisionKey
    observation: PackedObservation
    action_plan: ActionPlanFrame
    generation_step_count: int


@dataclass(slots=True)
class PolicyRequestCompiler:
    """Compile requests into a reusable lossless wire workspace."""

    batch_capacity: int
    _frame_buffer: bytearray = field(init=False, repr=False)
    _zero_buffer: bytearray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        assert self.batch_capacity > 0
        byte_count = max_policy_request_batch_frame_bytes(
            batch_capacity=self.batch_capacity,
            padded_generation_steps=MAX_ACTION_STEPS,
        )
        self._frame_buffer = bytearray(byte_count)
        self._zero_buffer = bytearray(byte_count)

    def compile_batch(
        self, requests: tuple[PolicyRequestInput, ...]
    ) -> _result.Ok[BorrowedPolicyRequestBatch] | _result.Rejected:
        """Compile a non-empty bounded request batch."""
        assert requests
        if len(requests) > self.batch_capacity:
            return Rejected(reason="policy request batch is full")
        planned: list[_PlannedPolicyRequest] = []
        for request in requests:
            result = _plan_request(request)
            if isinstance(result, Rejected):
                return result
            planned.append(result.value)
        observation_capacity = max(
            request.observation.token_count() for request in planned
        )
        generation_capacity = max(
            request.generation_step_count for request in planned
        )
        layout = policy_request_batch_layout(
            batch_capacity=len(planned),
            observation_token_capacity=observation_capacity,
            padded_generation_steps=generation_capacity,
        )
        self._frame_buffer[: layout.total_bytes] = self._zero_buffer[
            : layout.total_bytes
        ]
        writer = _PolicyRequestColumnWriter(
            data=self._frame_buffer, layout=layout
        )
        writer.initialize(row_count=len(planned))
        for row_index, request in enumerate(planned):
            writer.write_row(row_index=row_index, request=request)
        frame = PolicyRequestWireFrame(
            buffer=self._frame_buffer, byte_count=layout.total_bytes
        )
        return Ok(
            value=BorrowedPolicyRequestBatch(
                frame=frame,
                metadata=PolicyRequestFrameMetadata(
                    row_count=len(planned),
                    batch_capacity=len(planned),
                    observation_token_capacity=observation_capacity,
                    padded_generation_steps=generation_capacity,
                    generation_step_counts=tuple(
                        request.generation_step_count
                        for request in planned
                    ),
                    routes=tuple(request.route for request in planned),
                    policy_versions=tuple(
                        request.policy_version for request in planned
                    ),
                    byte_count=layout.total_bytes,
                    layout=layout,
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class _PolicyRequestColumnWriter:
    data: bytearray
    layout: PolicyRequestBatchLayout

    def initialize(self, *, row_count: int) -> None:
        initialize_policy_request_frame(
            memoryview(self.data)[: self.layout.total_bytes],
            row_count=row_count,
            layout=self.layout,
        )

    def write_row(
        self, *, row_index: int, request: _PlannedPolicyRequest
    ) -> None:
        self._write_i64(
            self.layout.route_worker_indices,
            row_index,
            request.route.worker_index,
        )
        self._write_i64(
            self.layout.route_request_ids,
            row_index,
            request.route.request_id,
        )
        self._write_i64(
            self.layout.policy_versions,
            row_index,
            request.policy_version,
        )
        self._write_i64(
            self.layout.generation_step_counts,
            row_index,
            request.generation_step_count,
        )
        self._write_observation(row_index, request.observation)
        self._write_action_plan(row_index, request.action_plan)
        self._write_thresholds(row_index, request)

    def _write_observation(
        self, row_index: int, observation: PackedObservation
    ) -> None:
        padded = padded_packed_observation(
            observation,
            token_count=self.layout.observation_token_capacity,
        )
        self._write_i64_values(
            self.layout.category_ids,
            row_index,
            tuple(
                value for row in padded.category_rows for value in row
            ),
        )
        self._write_f32_values(
            self.layout.scalar_values, row_index, padded.scalar_values
        )
        self._write_f32_values(
            self.layout.card_rule_values,
            row_index,
            tuple(
                value for row in padded.card_rule_rows for value in row
            ),
        )
        self._write_i64_values(
            self.layout.encoded_structure_coordinates,
            row_index,
            tuple(
                value
                for row in padded.encoded_structure_rows
                for value in row
            ),
        )
        self._write_i64_values(
            self.layout.candidate_category_ids,
            row_index,
            tuple(
                value
                for row in padded.candidate_category_rows
                for value in row
            ),
        )
        self._write_f32_values(
            self.layout.candidate_counts,
            row_index,
            padded.candidate_counts,
        )
        self._write_f32_values(
            self.layout.candidate_card_rule_values,
            row_index,
            tuple(
                value
                for row in padded.candidate_card_rule_rows
                for value in row
            ),
        )
        self._write_i64(
            self.layout.query_indices, row_index, padded.query_index
        )

    def _write_action_plan(
        self, row_index: int, plan: ActionPlanFrame
    ) -> None:
        self._write_i64(
            self.layout.kind_codes, row_index, plan.kind_code
        )
        self._write_i64(
            self.layout.min_select, row_index, plan.min_select
        )
        self._write_i64(
            self.layout.max_select, row_index, plan.max_select
        )
        self._write_i64(
            self.layout.exact_select, row_index, plan.exact_select
        )
        self._write_i64(
            self.layout.required_same_suit_count,
            row_index,
            plan.required_same_suit_count,
        )
        self._write_i64(
            self.layout.pair_floor, row_index, plan.pair_floor
        )
        self._write_bool_values(
            self.layout.has_tractor, row_index, (plan.has_tractor,)
        )
        self._write_i64_values(
            self.layout.available_counts,
            row_index,
            plan.available_counts,
        )
        self._write_i64_values(
            self.layout.effective_suits, row_index, plan.effective_suits
        )
        self._write_bool_values(
            self.layout.same_suit_mask, row_index, plan.same_suit_mask
        )
        self._write_bool_values(
            self.layout.off_suit_mask, row_index, plan.off_suit_mask
        )
        self._write_bool_values(
            self.layout.pair_face_mask, row_index, plan.pair_face_mask
        )
        trace_ids = [
            0
            for _ in range(
                MAX_TRACE_COUNT * self.layout.padded_generation_steps
            )
        ]
        trace_mask = [False for _ in trace_ids]
        trace_lengths = [0 for _ in range(MAX_TRACE_COUNT)]
        trace_rows = [False for _ in range(MAX_TRACE_COUNT)]
        for trace_index, trace in enumerate(plan.trace_choice_ids):
            trace_lengths[trace_index] = len(trace)
            trace_rows[trace_index] = True
            for step_index, choice_id in enumerate(trace):
                offset = (
                    trace_index * self.layout.padded_generation_steps
                    + step_index
                )
                trace_ids[offset] = choice_id
                trace_mask[offset] = True
        self._write_i64_values(
            self.layout.trace_choice_ids, row_index, tuple(trace_ids)
        )
        self._write_bool_values(
            self.layout.trace_choice_mask, row_index, tuple(trace_mask)
        )
        self._write_i64_values(
            self.layout.trace_lengths, row_index, tuple(trace_lengths)
        )
        self._write_bool_values(
            self.layout.trace_row_mask, row_index, tuple(trace_rows)
        )
        pair_masks = [
            False
            for _ in range(MAX_PAIR_PLAN_COUNT * ACTION_FACE_COUNT)
        ]
        pair_rows = [False for _ in range(MAX_PAIR_PLAN_COUNT)]
        for plan_index, mask in enumerate(plan.pair_plan_masks):
            pair_rows[plan_index] = True
            start = plan_index * ACTION_FACE_COUNT
            pair_masks[start : start + ACTION_FACE_COUNT] = mask
        self._write_bool_values(
            self.layout.pair_plan_masks, row_index, tuple(pair_masks)
        )
        self._write_bool_values(
            self.layout.pair_plan_row_mask, row_index, tuple(pair_rows)
        )

    def _write_thresholds(
        self, row_index: int, request: _PlannedPolicyRequest
    ) -> None:
        values = tuple(
            policy_choice_threshold(
                key=request.decision_key, step_index=step_index
            )
            if step_index < request.generation_step_count
            else 0.0
            for step_index in range(self.layout.padded_generation_steps)
        )
        struct.pack_into(
            f"<{len(values)}d",
            self.data,
            _row_offset(self.layout.sampling_thresholds, row_index),
            *values,
        )

    def _write_i64(
        self, column: ColumnLayout, row_index: int, value: int
    ) -> None:
        I64.pack_into(self.data, _row_offset(column, row_index), value)

    def _write_i64_values(
        self,
        column: ColumnLayout,
        row_index: int,
        values: tuple[int, ...],
    ) -> None:
        assert len(values) * I64.size == column.row_bytes
        struct.pack_into(
            f"<{len(values)}q",
            self.data,
            _row_offset(column, row_index),
            *values,
        )

    def _write_f32_values(
        self,
        column: ColumnLayout,
        row_index: int,
        values: tuple[float, ...],
    ) -> None:
        assert len(values) * 4 == column.row_bytes
        struct.pack_into(
            f"<{len(values)}f",
            self.data,
            _row_offset(column, row_index),
            *values,
        )

    def _write_bool_values(
        self,
        column: ColumnLayout,
        row_index: int,
        values: tuple[bool, ...],
    ) -> None:
        assert len(values) == column.row_bytes
        start = _row_offset(column, row_index)
        self.data[start : start + len(values)] = bytes(values)


def _plan_request(
    request: PolicyRequestInput,
) -> _result.Ok[_PlannedPolicyRequest] | _result.Rejected:
    action_plan = compile_legal_action_frame(request.legal_actions)
    generation_steps = action_plan_generation_step_count(action_plan)
    packed = pack_observation(request.observation)
    if packed.token_count() > MAX_LOSSLESS_OBSERVATION_TOKENS:
        return Rejected(
            reason="policy request observation is not lossless"
        )
    if len(action_plan.trace_choice_ids) > MAX_TRACE_COUNT:
        return Rejected(reason="policy request has too many traces")
    if len(action_plan.pair_plan_masks) > MAX_PAIR_PLAN_COUNT:
        return Rejected(reason="policy request has too many pair plans")
    if generation_steps > MAX_ACTION_STEPS:
        return Rejected(
            reason="policy request action trace is too wide"
        )
    return Ok(
        value=_PlannedPolicyRequest(
            route=request.route,
            policy_version=request.decision_key.policy_version,
            decision_key=request.decision_key,
            observation=packed,
            action_plan=action_plan,
            generation_step_count=generation_steps,
        )
    )


def _row_offset(column: ColumnLayout, row_index: int) -> int:
    return column.offset + row_index * column.row_bytes
