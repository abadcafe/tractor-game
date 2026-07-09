"""Prepare worker policy requests and encode remote request wire."""

from __future__ import annotations

import sys
from array import array
from collections.abc import Iterable
from dataclasses import dataclass, field

from server import result as _result
from server.result import Ok, Rejected
from server.training.packed_observation import (
    PackedObservation,
    pack_observation,
    padded_packed_observation,
)
from server.training.policy_inference_batch.frame import (
    initialize_policy_request_frame,
)
from server.training.policy_inference_batch.schema import (
    MAX_PAIR_PLAN_COUNT,
    MAX_TRACE_COUNT,
    PolicyRequestBatchLayout,
    policy_request_batch_layout,
)
from server.training.policy_inference_batch.types import (
    PolicyRequestBatch,
    PolicyRequestInput,
    PolicyRequestRoute,
    PolicyRequestWireFrame,
)
from server.training.sampling import policy_choice_threshold
from server.training.semantic_action_plan import (
    compile_legal_action_frame,
)
from server.training.semantic_action_plan.frame import (
    ActionPlanFrame,
    action_plan_generation_step_count,
)
from server.training.semantic_action_plan.spec import (
    ACTION_FACE_COUNT,
)
from server.training.semantic_actions.codec import SEMANTIC_CODEC


@dataclass(frozen=True, slots=True)
class _CompiledRequest:
    route: PolicyRequestRoute
    policy_version: int
    packed_observation: PackedObservation
    action_plan: ActionPlanFrame
    generation_step_count: int
    sampling_thresholds: tuple[float, ...]

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert self.generation_step_count > 0
        assert (
            len(self.sampling_thresholds) == self.generation_step_count
        )


def _request_row_list() -> list[_CompiledRequest]:
    return []


@dataclass(slots=True)
class PolicyRequestBatchBuilder:
    """Compile raw policy requests into transport-neutral batches."""

    batch_capacity: int
    max_observation_tokens: int
    _rows: list[_CompiledRequest] = field(
        default_factory=_request_row_list
    )
    _data: bytearray | None = field(init=False)
    _zero_data: bytearray = field(init=False)

    def __post_init__(self) -> None:
        assert self.batch_capacity > 0
        assert self.max_observation_tokens > 0
        self._data = None
        self._zero_data = bytearray()

    def reset(self) -> None:
        """Discard active rows while keeping reusable storage."""
        self._rows.clear()

    def has_rows(self) -> bool:
        """Return whether the current frame has active rows."""
        return bool(self._rows)

    def _compile_request(
        self, request: PolicyRequestInput
    ) -> _result.Ok[_CompiledRequest] | _result.Rejected:
        """Return a frame-ready row for a raw policy request."""
        packed = pack_observation(request.observation)
        action_plan = compile_legal_action_frame(request.legal_actions)
        generation_step_count = action_plan_generation_step_count(
            action_plan
        )
        validation_result = _validate_request(
            packed=packed,
            action_plan=action_plan,
            generation_step_count=generation_step_count,
            max_observation_tokens=self.max_observation_tokens,
        )
        if isinstance(validation_result, Rejected):
            return validation_result
        thresholds = tuple(
            policy_choice_threshold(
                key=request.decision_key,
                argument_index=argument_index,
            )
            for argument_index in range(generation_step_count)
        )
        return Ok(
            value=_CompiledRequest(
                route=request.route,
                policy_version=request.decision_key.policy_version,
                packed_observation=packed,
                action_plan=action_plan,
                generation_step_count=generation_step_count,
                sampling_thresholds=thresholds,
            )
        )

    def compile_batch(
        self, requests: tuple[PolicyRequestInput, ...]
    ) -> _result.Ok[PolicyRequestBatch] | _result.Rejected:
        """Compile one bounded request batch."""
        assert requests
        if len(requests) > self.batch_capacity:
            return Rejected(reason="policy request batch is full")
        self.reset()
        for request in requests:
            append_result = self.push_request(request)
            if isinstance(append_result, Rejected):
                return append_result
        return Ok(value=self.finish_batch())

    def push_request(
        self, request: PolicyRequestInput
    ) -> _result.Ok[None] | _result.Rejected:
        """Compile and append one raw request to the active frame."""
        row_result = self._compile_request(request)
        if isinstance(row_result, Rejected):
            return row_result
        return self._push_compiled(row_result.value)

    def _push_compiled(
        self, request: _CompiledRequest
    ) -> _result.Ok[None] | _result.Rejected:
        """Append one prepared row to the next emitted frame."""
        if len(self._rows) >= self.batch_capacity:
            return Rejected(reason="policy request batch is full")
        self._rows.append(request)
        return Ok(value=None)

    def finish_batch(self) -> PolicyRequestBatch:
        """Return the active transport-neutral batch."""
        assert self._rows
        padded_generation_steps = max(
            row.generation_step_count for row in self._rows
        )
        return PolicyRequestBatch(
            routes=tuple(row.route for row in self._rows),
            policy_versions=tuple(
                row.policy_version for row in self._rows
            ),
            packed_observations=tuple(
                row.packed_observation for row in self._rows
            ),
            action_plans=tuple(row.action_plan for row in self._rows),
            generation_step_counts=tuple(
                row.generation_step_count for row in self._rows
            ),
            sampling_threshold_rows=tuple(
                row.sampling_thresholds for row in self._rows
            ),
            max_observation_tokens=self.max_observation_tokens,
            padded_generation_steps=padded_generation_steps,
        )

    def encode_wire_frame(
        self, batch: PolicyRequestBatch
    ) -> PolicyRequestWireFrame:
        """Encode one prepared batch for connection transport."""
        assert batch.row_count() <= self.batch_capacity
        self._rows = [
            _CompiledRequest(
                route=route,
                policy_version=policy_version,
                packed_observation=packed_observation,
                action_plan=action_plan,
                generation_step_count=generation_step_count,
                sampling_thresholds=sampling_thresholds,
            )
            for (
                route,
                policy_version,
                packed_observation,
                action_plan,
                generation_step_count,
                sampling_thresholds,
            ) in zip(
                batch.routes,
                batch.policy_versions,
                batch.packed_observations,
                batch.action_plans,
                batch.generation_step_counts,
                batch.sampling_threshold_rows,
                strict=True,
            )
        ]
        return self.finish_wire_frame()

    def finish_wire_frame(self) -> PolicyRequestWireFrame:
        """Return active bytes for a remote request frame."""
        batch = self.finish_batch()
        padded_generation_steps = batch.padded_generation_steps
        frame_capacity = len(self._rows)
        layout = policy_request_batch_layout(
            batch_capacity=frame_capacity,
            max_observation_tokens=self.max_observation_tokens,
            padded_generation_steps=padded_generation_steps,
        )
        data = self._ensure_buffer(byte_count=layout.total_bytes)
        self._clear_frame(byte_count=layout.total_bytes)
        initialize_policy_request_frame(
            memoryview(data)[: layout.total_bytes],
            row_count=len(self._rows),
            batch_capacity=frame_capacity,
            max_observation_tokens=self.max_observation_tokens,
            padded_generation_steps=layout.padded_generation_steps,
        )
        rows = tuple(self._rows)
        _write_routes_and_versions(data=data, layout=layout, rows=rows)
        _write_observations(
            data=data,
            layout=layout,
            rows=rows,
            max_observation_tokens=self.max_observation_tokens,
        )
        _write_action_specs(data=data, layout=layout, rows=rows)
        _write_sampling_threshold_columns(
            data=data, layout=layout, rows=rows
        )
        return PolicyRequestWireFrame(
            data=data, byte_count=layout.total_bytes
        )

    def _ensure_buffer(self, *, byte_count: int) -> bytearray:
        data = self._data
        if data is None or len(data) != byte_count:
            data = bytearray(byte_count)
            self._data = data
        return data

    def _clear_frame(self, *, byte_count: int) -> None:
        data = self._data
        assert data is not None
        if len(self._zero_data) < byte_count:
            self._zero_data = bytearray(byte_count)
        data[:byte_count] = self._zero_data[:byte_count]


def _write_routes_and_versions(
    *,
    data: bytearray,
    layout: PolicyRequestBatchLayout,
    rows: tuple[_CompiledRequest, ...],
) -> None:
    _pack_i64_values(
        data,
        layout.route_worker_indices.offset,
        (row.route.worker_index for row in rows),
    )
    _pack_i64_values(
        data,
        layout.route_request_ids.offset,
        (row.route.request_id for row in rows),
    )
    _pack_i64_values(
        data,
        layout.policy_versions.offset,
        (row.policy_version for row in rows),
    )
    _pack_i64_values(
        data,
        layout.generation_step_counts.offset,
        (row.generation_step_count for row in rows),
    )


def _write_observations(
    *,
    data: bytearray,
    layout: PolicyRequestBatchLayout,
    rows: tuple[_CompiledRequest, ...],
    max_observation_tokens: int,
) -> None:
    padded = tuple(
        padded_packed_observation(
            row.packed_observation,
            max_observation_tokens=max_observation_tokens,
        )
        for row in rows
    )
    _pack_i64_values(
        data,
        layout.component_ids.offset,
        (
            value
            for observation in padded
            for token_row in observation.component_rows
            for value in token_row
        ),
    )
    _pack_f32_values(
        data,
        layout.numeric_values.offset,
        (
            value
            for observation in padded
            for numeric_row in observation.numeric_value_rows
            for value in numeric_row
        ),
    )
    _pack_f32_values(
        data,
        layout.numeric_masks.offset,
        (
            value
            for observation in padded
            for numeric_row in observation.numeric_mask_rows
            for value in numeric_row
        ),
    )


def _write_action_specs(
    *,
    data: bytearray,
    layout: PolicyRequestBatchLayout,
    rows: tuple[_CompiledRequest, ...],
) -> None:
    plans = tuple(row.action_plan for row in rows)
    _pack_i64_values(
        data,
        layout.kind_codes.offset,
        (plan.kind_code for plan in plans),
    )
    _write_selection_columns(
        data=data,
        layout=layout,
        plans=plans,
    )
    _write_trace_set_columns(
        data=data,
        layout=layout,
        plans=plans,
    )


def _write_selection_columns(
    *,
    data: bytearray,
    layout: PolicyRequestBatchLayout,
    plans: tuple[ActionPlanFrame, ...],
) -> None:
    _pack_i64_values(
        data,
        layout.min_select.offset,
        (plan.min_select for plan in plans),
    )
    _pack_i64_values(
        data,
        layout.max_select.offset,
        (plan.max_select for plan in plans),
    )
    _pack_i64_values(
        data,
        layout.exact_select.offset,
        (plan.exact_select for plan in plans),
    )
    _pack_i64_values(
        data,
        layout.required_same_suit_count.offset,
        (plan.required_same_suit_count for plan in plans),
    )
    _pack_i64_values(
        data,
        layout.pair_floor.offset,
        (plan.pair_floor for plan in plans),
    )
    _write_bool_values(
        data,
        layout.has_tractor.offset,
        (plan.has_tractor for plan in plans),
    )
    _pack_i64_values(
        data,
        layout.available_counts.offset,
        (value for plan in plans for value in plan.available_counts),
    )
    _pack_i64_values(
        data,
        layout.effective_suits.offset,
        (value for plan in plans for value in plan.effective_suits),
    )
    _write_bool_values(
        data,
        layout.same_suit_mask.offset,
        (value for plan in plans for value in plan.same_suit_mask),
    )
    _write_bool_values(
        data,
        layout.off_suit_mask.offset,
        (value for plan in plans for value in plan.off_suit_mask),
    )
    _write_bool_values(
        data,
        layout.pair_face_mask.offset,
        (value for plan in plans for value in plan.pair_face_mask),
    )
    _write_pair_plan_columns(
        data=data,
        layout=layout,
        plans=plans,
    )


def _write_trace_set_columns(
    *,
    data: bytearray,
    layout: PolicyRequestBatchLayout,
    plans: tuple[ActionPlanFrame, ...],
) -> None:
    trace_tokens = [0] * (
        len(plans) * MAX_TRACE_COUNT * layout.padded_generation_steps
    )
    trace_token_mask = bytearray(
        len(plans) * MAX_TRACE_COUNT * layout.padded_generation_steps
    )
    trace_lengths = [0] * (len(plans) * MAX_TRACE_COUNT)
    trace_row_mask = bytearray(len(plans) * MAX_TRACE_COUNT)
    for row_index, plan in enumerate(plans):
        row_trace_base = row_index * MAX_TRACE_COUNT
        token_row_base = (
            row_index * MAX_TRACE_COUNT * layout.padded_generation_steps
        )
        for trace_index, trace in enumerate(plan.trace_tokens):
            trace_lengths[row_trace_base + trace_index] = len(trace)
            trace_row_mask[row_trace_base + trace_index] = 1
            trace_base = (
                token_row_base
                + trace_index * layout.padded_generation_steps
            )
            for step_index, token_id in enumerate(trace):
                trace_tokens[trace_base + step_index] = token_id
                trace_token_mask[trace_base + step_index] = 1
    _pack_i64_values(data, layout.trace_tokens.offset, trace_tokens)
    _write_bytes(data, layout.trace_token_mask.offset, trace_token_mask)
    _pack_i64_values(data, layout.trace_lengths.offset, trace_lengths)
    _write_bytes(data, layout.trace_row_mask.offset, trace_row_mask)


def _write_pair_plan_columns(
    *,
    data: bytearray,
    layout: PolicyRequestBatchLayout,
    plans: tuple[ActionPlanFrame, ...],
) -> None:
    plan_masks = bytearray(
        len(plans) * MAX_PAIR_PLAN_COUNT * ACTION_FACE_COUNT
    )
    row_masks = bytearray(len(plans) * MAX_PAIR_PLAN_COUNT)
    for row_index, plan in enumerate(plans):
        row_plan_base = row_index * MAX_PAIR_PLAN_COUNT
        mask_row_base = (
            row_index * MAX_PAIR_PLAN_COUNT * ACTION_FACE_COUNT
        )
        for plan_index, row in enumerate(plan.pair_plan_masks):
            row_masks[row_plan_base + plan_index] = 1
            mask_base = mask_row_base + plan_index * ACTION_FACE_COUNT
            for face_index, value in enumerate(row):
                plan_masks[mask_base + face_index] = 1 if value else 0
    _write_bytes(data, layout.pair_plan_masks.offset, plan_masks)
    _write_bytes(data, layout.pair_plan_row_mask.offset, row_masks)


def _write_sampling_threshold_columns(
    *,
    data: bytearray,
    layout: PolicyRequestBatchLayout,
    rows: tuple[_CompiledRequest, ...],
) -> None:
    threshold_values = [0.0] * (
        len(rows) * layout.padded_generation_steps
    )
    for row_index, row in enumerate(rows):
        row_base = row_index * layout.padded_generation_steps
        for argument_index, threshold in enumerate(
            row.sampling_thresholds
        ):
            threshold_values[row_base + argument_index] = threshold
    _pack_f64_values(
        data, layout.sampling_thresholds.offset, threshold_values
    )


def _pack_i64_values(
    data: bytearray, offset: int, values: Iterable[int]
) -> None:
    _write_array_values(data, offset, array("q", values))


def _pack_f32_values(
    data: bytearray, offset: int, values: Iterable[float]
) -> None:
    _write_array_values(data, offset, array("f", values))


def _pack_f64_values(
    data: bytearray, offset: int, values: Iterable[float]
) -> None:
    _write_array_values(data, offset, array("d", values))


def _write_array_values(
    data: bytearray, offset: int, values: array[int] | array[float]
) -> None:
    assert len(values) > 0
    if sys.byteorder != "little":
        values.byteswap()
    payload = values.tobytes()
    _write_bytes(data, offset, payload)


def _write_bool_values(
    data: bytearray, offset: int, values: Iterable[bool]
) -> None:
    _write_bytes(
        data, offset, bytes(1 if value else 0 for value in values)
    )


def _write_bytes(
    data: bytearray, offset: int, values: bytes | bytearray
) -> None:
    data[offset : offset + len(values)] = values


def _validate_request(
    *,
    packed: PackedObservation,
    action_plan: ActionPlanFrame,
    generation_step_count: int,
    max_observation_tokens: int,
) -> Ok[None] | Rejected:
    if packed.token_count() > max_observation_tokens:
        return Rejected(
            reason="policy request observation exceeds token budget"
        )
    if packed.token_count() <= 0:
        return Rejected(reason="policy request observation is empty")
    if _trace_count(action_plan) > MAX_TRACE_COUNT:
        return Rejected(reason="policy request has too many traces")
    if _pair_plan_count(action_plan) > MAX_PAIR_PLAN_COUNT:
        return Rejected(reason="policy request has too many pair plans")
    if generation_step_count > SEMANTIC_CODEC.max_argument_tokens:
        return Rejected(
            reason="policy request action trace is too wide"
        )
    return Ok(value=None)


def _trace_count(action_plan: ActionPlanFrame) -> int:
    return max(len(action_plan.trace_tokens), 1)


def _pair_plan_count(action_plan: ActionPlanFrame) -> int:
    return max(len(action_plan.pair_plan_masks), 1)
