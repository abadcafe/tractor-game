"""Prepare worker policy requests and encode remote request wire."""

from __future__ import annotations

from dataclasses import dataclass

from server import result as _result
from server.result import Ok, Rejected
from server.training.feature_schema import NUMERIC_FEATURE_COUNT
from server.training.numeric_features import (
    NumericFeatureValues,
    numeric_feature_values,
)
from server.training.observation import Observation
from server.training.packed_observation import (
    OBSERVATION_COMPONENT_COUNT,
)
from server.training.policy_inference_batch.frame import (
    initialize_policy_request_frame,
)
from server.training.policy_inference_batch.schema import (
    F32,
    F64,
    I64,
    MAX_PAIR_PLAN_COUNT,
    MAX_TRACE_COUNT,
    ColumnLayout,
    PolicyRequestBatchLayout,
    policy_request_batch_layout,
)
from server.training.policy_inference_batch.types import (
    CompiledPolicyRequestBatch,
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
from server.training.semantic_action_plan.spec import (
    ACTION_FACE_COUNT,
)
from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.tokens import ObservationToken
from server.training.vocab import component_ids
from server.training.vocab_schema import TokenComponentIds


@dataclass(frozen=True, slots=True)
class _PlannedPolicyRequest:
    route: PolicyRequestRoute
    policy_version: int
    decision_key: PolicyDecisionKey
    observation: Observation
    action_plan: ActionPlanFrame
    generation_step_count: int
    observation_token_count: int

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert self.generation_step_count > 0
        assert self.observation_token_count > 0


@dataclass(slots=True)
class PolicyRequestCompiler:
    """Compile raw worker requests into one owned columnar frame."""

    batch_capacity: int
    max_observation_tokens: int

    def __post_init__(self) -> None:
        assert self.batch_capacity > 0
        assert self.max_observation_tokens > 0

    def compile_batch(
        self, requests: tuple[PolicyRequestInput, ...]
    ) -> _result.Ok[CompiledPolicyRequestBatch] | _result.Rejected:
        """Compile one bounded request batch into canonical columns."""
        assert requests
        if len(requests) > self.batch_capacity:
            return Rejected(reason="policy request batch is full")
        planned_requests: list[_PlannedPolicyRequest] = []
        for request in requests:
            planned_result = _plan_request(
                request=request,
                max_observation_tokens=self.max_observation_tokens,
            )
            if isinstance(planned_result, Rejected):
                return planned_result
            planned_requests.append(planned_result.value)
        return Ok(
            value=_PolicyRequestColumnWriter.compile(
                requests=tuple(planned_requests),
                max_observation_tokens=self.max_observation_tokens,
            )
        )


@dataclass(frozen=True, slots=True)
class _PolicyRequestColumnWriter:
    """Own one request frame and write rows directly into columns."""

    data: bytearray
    layout: PolicyRequestBatchLayout
    max_observation_tokens: int

    @classmethod
    def compile(
        cls,
        *,
        requests: tuple[_PlannedPolicyRequest, ...],
        max_observation_tokens: int,
    ) -> CompiledPolicyRequestBatch:
        """Return an owned compiled batch for planned requests."""
        assert requests
        padded_generation_steps = max(
            request.generation_step_count for request in requests
        )
        layout = policy_request_batch_layout(
            batch_capacity=len(requests),
            max_observation_tokens=max_observation_tokens,
            padded_generation_steps=padded_generation_steps,
        )
        data = bytearray(layout.total_bytes)
        writer = cls(
            data=data,
            layout=layout,
            max_observation_tokens=max_observation_tokens,
        )
        writer._initialize_header(row_count=len(requests))
        for row_index, request in enumerate(requests):
            writer._write_row(row_index=row_index, request=request)
        frame = PolicyRequestWireFrame(
            data=data, byte_count=layout.total_bytes
        )
        return CompiledPolicyRequestBatch(
            frame=frame,
            metadata=PolicyRequestFrameMetadata(
                row_count=len(requests),
                batch_capacity=len(requests),
                max_observation_tokens=max_observation_tokens,
                padded_generation_steps=padded_generation_steps,
                generation_step_counts=tuple(
                    request.generation_step_count
                    for request in requests
                ),
                routes=tuple(request.route for request in requests),
                policy_versions=tuple(
                    request.policy_version for request in requests
                ),
                byte_count=frame.byte_count,
            ),
        )

    def _initialize_header(self, *, row_count: int) -> None:
        initialize_policy_request_frame(
            memoryview(self.data)[: self.layout.total_bytes],
            row_count=row_count,
            batch_capacity=self.layout.batch_capacity,
            max_observation_tokens=self.max_observation_tokens,
            padded_generation_steps=self.layout.padded_generation_steps,
        )

    def _write_row(
        self, *, row_index: int, request: _PlannedPolicyRequest
    ) -> None:
        self._write_scalar_i64(
            self.layout.route_worker_indices,
            row_index,
            request.route.worker_index,
        )
        self._write_scalar_i64(
            self.layout.route_request_ids,
            row_index,
            request.route.request_id,
        )
        self._write_scalar_i64(
            self.layout.policy_versions,
            row_index,
            request.policy_version,
        )
        self._write_scalar_i64(
            self.layout.generation_step_counts,
            row_index,
            request.generation_step_count,
        )
        self._write_observation(
            row_index=row_index, observation=request.observation
        )
        self._write_action_plan(
            row_index=row_index, plan=request.action_plan
        )
        self._write_sampling_thresholds(
            row_index=row_index, request=request
        )

    def _write_observation(
        self, *, row_index: int, observation: Observation
    ) -> None:
        for token_index, token in enumerate(observation.tokens):
            self._write_observation_token(
                row_index=row_index,
                token_index=token_index,
                token=token,
            )

    def _write_observation_token(
        self,
        *,
        row_index: int,
        token_index: int,
        token: ObservationToken,
    ) -> None:
        self._write_component_row(
            row_index=row_index,
            token_index=token_index,
            values=component_ids(token),
        )
        numeric = numeric_feature_values(token)
        self._write_numeric_row(
            row_index=row_index,
            token_index=token_index,
            values=numeric,
        )

    def _write_component_row(
        self,
        *,
        row_index: int,
        token_index: int,
        values: TokenComponentIds,
    ) -> None:
        base_offset = (
            _row_offset(self.layout.component_ids, row_index)
            + token_index * ACTION_OBSERVATION_COMPONENT_BYTES
        )
        _pack_i64(self.data, base_offset, values.token_type)
        _pack_i64(self.data, base_offset + I64.size, values.segment)
        _pack_i64(self.data, base_offset + I64.size * 2, values.field)
        _pack_i64(self.data, base_offset + I64.size * 3, values.value)
        _pack_i64(self.data, base_offset + I64.size * 4, values.suit)
        _pack_i64(self.data, base_offset + I64.size * 5, values.rank)
        _pack_i64(self.data, base_offset + I64.size * 6, values.points)
        _pack_i64(self.data, base_offset + I64.size * 7, values.color)
        _pack_i64(self.data, base_offset + I64.size * 8, values.role)
        _pack_i64(
            self.data, base_offset + I64.size * 9, values.trick_age
        )
        _pack_i64(
            self.data, base_offset + I64.size * 10, values.trick_state
        )
        _pack_i64(
            self.data, base_offset + I64.size * 11, values.play_order
        )
        _pack_i64(self.data, base_offset + I64.size * 12, values.count)
        _pack_i64(
            self.data, base_offset + I64.size * 13, values.play_width
        )
        _pack_i64(
            self.data, base_offset + I64.size * 14, values.event_age
        )

    def _write_numeric_row(
        self,
        *,
        row_index: int,
        token_index: int,
        values: NumericFeatureValues,
    ) -> None:
        value_offset = (
            _row_offset(self.layout.numeric_values, row_index)
            + token_index * NUMERIC_FEATURE_BYTES
        )
        mask_offset = (
            _row_offset(self.layout.numeric_masks, row_index)
            + token_index * NUMERIC_FEATURE_BYTES
        )
        for index, value in enumerate(values.values):
            _pack_f32(self.data, value_offset + index * F32.size, value)
        for index, value in enumerate(values.masks):
            _pack_f32(self.data, mask_offset + index * F32.size, value)

    def _write_action_plan(
        self, *, row_index: int, plan: ActionPlanFrame
    ) -> None:
        self._write_scalar_i64(
            self.layout.kind_codes, row_index, plan.kind_code
        )
        self._write_selection_columns(row_index=row_index, plan=plan)
        self._write_trace_set_columns(row_index=row_index, plan=plan)

    def _write_selection_columns(
        self, *, row_index: int, plan: ActionPlanFrame
    ) -> None:
        self._write_scalar_i64(
            self.layout.min_select, row_index, plan.min_select
        )
        self._write_scalar_i64(
            self.layout.max_select, row_index, plan.max_select
        )
        self._write_scalar_i64(
            self.layout.exact_select, row_index, plan.exact_select
        )
        self._write_scalar_i64(
            self.layout.required_same_suit_count,
            row_index,
            plan.required_same_suit_count,
        )
        self._write_scalar_i64(
            self.layout.pair_floor, row_index, plan.pair_floor
        )
        self._write_scalar_bool(
            self.layout.has_tractor, row_index, plan.has_tractor
        )
        self._write_i64_vector(
            column=self.layout.available_counts,
            row_index=row_index,
            values=plan.available_counts,
        )
        self._write_i64_vector(
            column=self.layout.effective_suits,
            row_index=row_index,
            values=plan.effective_suits,
        )
        self._write_bool_vector(
            column=self.layout.same_suit_mask,
            row_index=row_index,
            values=plan.same_suit_mask,
        )
        self._write_bool_vector(
            column=self.layout.off_suit_mask,
            row_index=row_index,
            values=plan.off_suit_mask,
        )
        self._write_bool_vector(
            column=self.layout.pair_face_mask,
            row_index=row_index,
            values=plan.pair_face_mask,
        )
        self._write_pair_plan_columns(row_index=row_index, plan=plan)

    def _write_trace_set_columns(
        self, *, row_index: int, plan: ActionPlanFrame
    ) -> None:
        token_row_base = _row_offset(
            self.layout.trace_tokens, row_index
        )
        token_mask_base = _row_offset(
            self.layout.trace_token_mask, row_index
        )
        length_base = _row_offset(self.layout.trace_lengths, row_index)
        row_mask_base = _row_offset(
            self.layout.trace_row_mask, row_index
        )
        for trace_index, trace in enumerate(plan.trace_tokens):
            _pack_i64(
                self.data,
                length_base + trace_index * I64.size,
                len(trace),
            )
            self.data[row_mask_base + trace_index] = 1
            trace_token_base = (
                token_row_base
                + trace_index
                * self.layout.padded_generation_steps
                * I64.size
            )
            trace_token_mask_base = (
                token_mask_base
                + trace_index * self.layout.padded_generation_steps
            )
            for step_index, token_id in enumerate(trace):
                _pack_i64(
                    self.data,
                    trace_token_base + step_index * I64.size,
                    token_id,
                )
                self.data[trace_token_mask_base + step_index] = 1

    def _write_pair_plan_columns(
        self, *, row_index: int, plan: ActionPlanFrame
    ) -> None:
        mask_base = _row_offset(self.layout.pair_plan_masks, row_index)
        row_mask_base = _row_offset(
            self.layout.pair_plan_row_mask, row_index
        )
        for plan_index, plan_mask in enumerate(plan.pair_plan_masks):
            self.data[row_mask_base + plan_index] = 1
            plan_base = mask_base + plan_index * ACTION_FACE_COUNT
            for face_index, value in enumerate(plan_mask):
                self.data[plan_base + face_index] = 1 if value else 0

    def _write_sampling_thresholds(
        self, *, row_index: int, request: _PlannedPolicyRequest
    ) -> None:
        base_offset = _row_offset(
            self.layout.sampling_thresholds, row_index
        )
        for argument_index in range(request.generation_step_count):
            threshold = policy_choice_threshold(
                key=request.decision_key,
                argument_index=argument_index,
            )
            _pack_f64(
                self.data,
                base_offset + argument_index * F64.size,
                threshold,
            )

    def _write_scalar_i64(
        self, column: ColumnLayout, row_index: int, value: int
    ) -> None:
        _pack_i64(self.data, _row_offset(column, row_index), value)

    def _write_scalar_bool(
        self, column: ColumnLayout, row_index: int, value: bool
    ) -> None:
        self.data[_row_offset(column, row_index)] = 1 if value else 0

    def _write_i64_vector(
        self,
        *,
        column: ColumnLayout,
        row_index: int,
        values: tuple[int, ...],
    ) -> None:
        base_offset = _row_offset(column, row_index)
        for index, value in enumerate(values):
            _pack_i64(self.data, base_offset + index * I64.size, value)

    def _write_bool_vector(
        self,
        *,
        column: ColumnLayout,
        row_index: int,
        values: tuple[bool, ...],
    ) -> None:
        base_offset = _row_offset(column, row_index)
        for index, value in enumerate(values):
            self.data[base_offset + index] = 1 if value else 0


def _plan_request(
    *,
    request: PolicyRequestInput,
    max_observation_tokens: int,
) -> _result.Ok[_PlannedPolicyRequest] | _result.Rejected:
    action_plan = compile_legal_action_frame(request.legal_actions)
    generation_step_count = action_plan_generation_step_count(
        action_plan
    )
    observation_token_count = max(len(request.observation.tokens), 1)
    validation_result = _validate_request(
        observation_token_count=observation_token_count,
        action_plan=action_plan,
        generation_step_count=generation_step_count,
        max_observation_tokens=max_observation_tokens,
    )
    if isinstance(validation_result, Rejected):
        return validation_result
    return Ok(
        value=_PlannedPolicyRequest(
            route=request.route,
            policy_version=request.decision_key.policy_version,
            decision_key=request.decision_key,
            observation=request.observation,
            action_plan=action_plan,
            generation_step_count=generation_step_count,
            observation_token_count=observation_token_count,
        )
    )


def _validate_request(
    *,
    observation_token_count: int,
    action_plan: ActionPlanFrame,
    generation_step_count: int,
    max_observation_tokens: int,
) -> Ok[None] | Rejected:
    if observation_token_count > max_observation_tokens:
        return Rejected(
            reason="policy request observation exceeds token budget"
        )
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


def _row_offset(column: ColumnLayout, row_index: int) -> int:
    return column.offset + row_index * column.row_bytes


def _pack_i64(data: bytearray, offset: int, value: int) -> None:
    I64.pack_into(data, offset, value)


def _pack_f32(data: bytearray, offset: int, value: float) -> None:
    F32.pack_into(data, offset, value)


def _pack_f64(data: bytearray, offset: int, value: float) -> None:
    F64.pack_into(data, offset, value)


ACTION_OBSERVATION_COMPONENT_BYTES = (
    OBSERVATION_COMPONENT_COUNT * I64.size
)
NUMERIC_FEATURE_BYTES = NUMERIC_FEATURE_COUNT * F32.size
