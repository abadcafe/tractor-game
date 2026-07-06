"""Policy inference request wire schema and builder."""

from __future__ import annotations

import struct
from dataclasses import dataclass

from server import result as _result
from server.result import Ok, Rejected
from server.training.feature_schema import NUMERIC_FEATURE_COUNT
from server.training.legal_actions import LegalActionIndex
from server.training.observation import Observation
from server.training.packed_observation import (
    OBSERVATION_COMPONENT_COUNT,
    PackedObservation,
    pack_observation,
)
from server.training.sampling import (
    PolicyDecisionKey,
    policy_choice_threshold,
)
from server.training.semantic_action_plan import (
    ActionPlanFrame,
    compile_legal_action_frame,
)
from server.training.semantic_action_plan.spec import ACTION_FACE_COUNT
from server.training.semantic_actions.codec import SEMANTIC_CODEC

from .types import (
    PolicyRequestMetadata,
    PolicyRequestRoute,
    PolicyRequestWire,
)

WIRE_REQUEST_MAGIC = 0x5452504F4C495231
WIRE_MAX_TRACE_COUNT = 128
WIRE_MAX_PAIR_PLAN_COUNT = 64

_I64 = struct.Struct("<q")
_F32 = struct.Struct("<f")
_F64 = struct.Struct("<d")

_HEADER_WORD_COUNT = 20
REQUEST_HEADER_WORD_COUNT = _HEADER_WORD_COUNT
REQUEST_HEADER_BYTES = _HEADER_WORD_COUNT * _I64.size

_MAGIC_INDEX = 0
_TOTAL_BYTES_INDEX = 1
_WORKER_INDEX_INDEX = 2
_REQUEST_ID_INDEX = 3
_TOKEN_COUNT_INDEX = 4
_TRACE_COUNT_INDEX = 5
_TRACE_STEPS_INDEX = 6
_PAIR_PLAN_COUNT_INDEX = 7
_BASE_SEED_INDEX = 8
_POLICY_VERSION_INDEX = 9
_EPISODE_ID_INDEX = 10
_PLAYER_INDEX_INDEX = 11
_DECISION_INDEX_INDEX = 12
_KIND_CODE_INDEX = 13
_MIN_SELECT_INDEX = 14
_MAX_SELECT_INDEX = 15
_EXACT_SELECT_INDEX = 16
_REQUIRED_SAME_SUIT_COUNT_INDEX = 17
_PAIR_FLOOR_INDEX = 18
_HAS_TRACTOR_INDEX = 19
REQUEST_TOKEN_COUNT_INDEX = _TOKEN_COUNT_INDEX
REQUEST_TRACE_COUNT_INDEX = _TRACE_COUNT_INDEX
REQUEST_TRACE_STEPS_INDEX = _TRACE_STEPS_INDEX
REQUEST_PAIR_PLAN_COUNT_INDEX = _PAIR_PLAN_COUNT_INDEX
REQUEST_POLICY_VERSION_INDEX = _POLICY_VERSION_INDEX
REQUEST_KIND_CODE_INDEX = _KIND_CODE_INDEX
REQUEST_MIN_SELECT_INDEX = _MIN_SELECT_INDEX
REQUEST_MAX_SELECT_INDEX = _MAX_SELECT_INDEX
REQUEST_EXACT_SELECT_INDEX = _EXACT_SELECT_INDEX
REQUEST_REQUIRED_SAME_SUIT_COUNT_INDEX = _REQUIRED_SAME_SUIT_COUNT_INDEX
REQUEST_PAIR_FLOOR_INDEX = _PAIR_FLOOR_INDEX
REQUEST_HAS_TRACTOR_INDEX = _HAS_TRACTOR_INDEX


@dataclass(frozen=True, slots=True)
class RequestSectionOffsets:
    """Byte offsets for one compact request body."""

    component_ids: int
    numeric_values: int
    numeric_masks: int
    available_counts: int
    effective_suits: int
    same_suit_mask: int
    off_suit_mask: int
    pair_face_mask: int
    trace_tokens: int
    trace_token_mask: int
    trace_lengths: int
    trace_row_mask: int
    pair_plan_masks: int
    pair_plan_row_mask: int
    sampling_thresholds: int
    total_bytes: int


type WireBuffer = bytes | bytearray | memoryview


def build_policy_request_wire(
    *,
    worker_index: int,
    request_id: int,
    observation: Observation,
    legal_actions: LegalActionIndex,
    decision_key: PolicyDecisionKey,
) -> _result.Ok[PolicyRequestWire] | _result.Rejected:
    """Build one compact tensor-oriented request wire message."""
    assert worker_index >= 0
    assert request_id >= 0
    packed = pack_observation(observation)
    action_plan = compile_legal_action_frame(legal_actions)
    validate_result = _validate_wire_sizes(
        packed=packed,
        action_plan=action_plan,
    )
    if isinstance(validate_result, Rejected):
        return validate_result
    trace_count = _trace_count(action_plan)
    trace_steps = _trace_steps(action_plan)
    pair_plan_count = _pair_plan_count(action_plan)
    offsets = request_section_offsets(
        token_count=packed.token_count(),
        trace_count=trace_count,
        trace_steps=trace_steps,
        pair_plan_count=pair_plan_count,
    )
    writer = _WireWriter(bytearray(offsets.total_bytes))
    _write_header(
        writer=writer,
        total_bytes=offsets.total_bytes,
        worker_index=worker_index,
        request_id=request_id,
        token_count=packed.token_count(),
        trace_count=trace_count,
        trace_steps=trace_steps,
        pair_plan_count=pair_plan_count,
        decision_key=decision_key,
        action_plan=action_plan,
    )
    _write_observation_sections(
        writer=writer, offsets=offsets, packed=packed
    )
    _write_action_plan_sections(
        writer=writer,
        offsets=offsets,
        action_plan=action_plan,
        trace_count=trace_count,
        trace_steps=trace_steps,
        pair_plan_count=pair_plan_count,
    )
    _write_sampling_thresholds(
        writer=writer,
        offsets=offsets,
        decision_key=decision_key,
    )
    return Ok(value=PolicyRequestWire(data=writer.bytes()))


def decode_policy_request_metadata(
    data: WireBuffer,
) -> _result.Ok[PolicyRequestMetadata] | _result.Rejected:
    """Decode the request header without unpacking payload body."""
    if len(data) < REQUEST_HEADER_BYTES:
        return Rejected(
            reason="policy request wire header is truncated"
        )
    if _read_i64(data, _MAGIC_INDEX) != WIRE_REQUEST_MAGIC:
        return Rejected(reason="policy request wire schema is invalid")
    total_bytes = _read_i64(data, _TOTAL_BYTES_INDEX)
    if total_bytes != len(data):
        return Rejected(reason="policy request wire length mismatch")
    worker_index = _read_i64(data, _WORKER_INDEX_INDEX)
    request_id = _read_i64(data, _REQUEST_ID_INDEX)
    token_count = _read_i64(data, _TOKEN_COUNT_INDEX)
    trace_count = _read_i64(data, _TRACE_COUNT_INDEX)
    trace_steps = _read_i64(data, _TRACE_STEPS_INDEX)
    pair_plan_count = _read_i64(data, _PAIR_PLAN_COUNT_INDEX)
    policy_version = _read_i64(data, _POLICY_VERSION_INDEX)
    if worker_index < 0 or request_id < 0:
        return Rejected(reason="policy request route is invalid")
    if token_count <= 0:
        return Rejected(reason="policy request token count is invalid")
    if trace_count <= 0 or trace_count > WIRE_MAX_TRACE_COUNT:
        return Rejected(reason="policy request trace count is invalid")
    if (
        trace_steps <= 0
        or trace_steps > SEMANTIC_CODEC.max_argument_tokens
    ):
        return Rejected(reason="policy request trace width is invalid")
    if (
        pair_plan_count <= 0
        or pair_plan_count > WIRE_MAX_PAIR_PLAN_COUNT
    ):
        return Rejected(
            reason="policy request pair plan count is invalid"
        )
    expected = request_section_offsets(
        token_count=token_count,
        trace_count=trace_count,
        trace_steps=trace_steps,
        pair_plan_count=pair_plan_count,
    ).total_bytes
    if total_bytes != expected:
        return Rejected(reason="policy request wire layout mismatch")
    return Ok(
        value=PolicyRequestMetadata(
            route=PolicyRequestRoute(
                worker_index=worker_index,
                request_id=request_id,
            ),
            byte_count=total_bytes,
            token_count=token_count,
            trace_count=trace_count,
            trace_steps=trace_steps,
            pair_plan_count=pair_plan_count,
            policy_version=policy_version,
        )
    )


def max_policy_request_wire_bytes(
    *, max_observation_tokens: int
) -> int:
    """Return maximum compact wire bytes accepted by a model rank."""
    assert max_observation_tokens > 0
    return request_section_offsets(
        token_count=max_observation_tokens,
        trace_count=WIRE_MAX_TRACE_COUNT,
        trace_steps=SEMANTIC_CODEC.max_argument_tokens,
        pair_plan_count=WIRE_MAX_PAIR_PLAN_COUNT,
    ).total_bytes


def request_section_offsets(
    *,
    token_count: int,
    trace_count: int,
    trace_steps: int,
    pair_plan_count: int,
) -> RequestSectionOffsets:
    """Return compact section offsets for one request shape."""
    assert token_count > 0
    assert trace_count > 0
    assert trace_steps > 0
    assert pair_plan_count > 0
    offset = REQUEST_HEADER_BYTES
    offset = _align(offset, _I64.size)
    component_ids = offset
    offset += token_count * OBSERVATION_COMPONENT_COUNT * _I64.size
    offset = _align(offset, _F32.size)
    numeric_values = offset
    offset += token_count * NUMERIC_FEATURE_COUNT * _F32.size
    offset = _align(offset, _F32.size)
    numeric_masks = offset
    offset += token_count * NUMERIC_FEATURE_COUNT * _F32.size
    offset = _align(offset, _I64.size)
    available_counts = offset
    offset += ACTION_FACE_COUNT * _I64.size
    effective_suits = offset
    offset += ACTION_FACE_COUNT * _I64.size
    same_suit_mask = offset
    offset += ACTION_FACE_COUNT
    off_suit_mask = offset
    offset += ACTION_FACE_COUNT
    pair_face_mask = offset
    offset += ACTION_FACE_COUNT
    offset = _align(offset, _I64.size)
    trace_tokens = offset
    offset += trace_count * trace_steps * _I64.size
    trace_token_mask = offset
    offset += trace_count * trace_steps
    offset = _align(offset, _I64.size)
    trace_lengths = offset
    offset += trace_count * _I64.size
    trace_row_mask = offset
    offset += trace_count
    pair_plan_masks = offset
    offset += pair_plan_count * ACTION_FACE_COUNT
    pair_plan_row_mask = offset
    offset += pair_plan_count
    offset = _align(offset, _F64.size)
    sampling_thresholds = offset
    offset += SEMANTIC_CODEC.max_argument_tokens * _F64.size
    return RequestSectionOffsets(
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
        total_bytes=offset,
    )


class _WireWriter:
    def __init__(self, data: bytearray) -> None:
        self._data = data

    def bytes(self) -> bytes:
        return bytes(self._data)

    def write_i64_at(self, offset: int, value: int) -> None:
        _I64.pack_into(self._data, offset, value)

    def write_f32_at(self, offset: int, value: float) -> None:
        _F32.pack_into(self._data, offset, value)

    def write_f64_at(self, offset: int, value: float) -> None:
        _F64.pack_into(self._data, offset, value)

    def write_bool_at(self, offset: int, value: bool) -> None:
        self._data[offset] = 1 if value else 0


def _write_header(
    *,
    writer: _WireWriter,
    total_bytes: int,
    worker_index: int,
    request_id: int,
    token_count: int,
    trace_count: int,
    trace_steps: int,
    pair_plan_count: int,
    decision_key: PolicyDecisionKey,
    action_plan: ActionPlanFrame,
) -> None:
    words = (
        WIRE_REQUEST_MAGIC,
        total_bytes,
        worker_index,
        request_id,
        token_count,
        trace_count,
        trace_steps,
        pair_plan_count,
        decision_key.base_seed,
        decision_key.policy_version,
        decision_key.episode_id,
        decision_key.player_index,
        decision_key.decision_index,
        action_plan.kind_code,
        action_plan.min_select,
        action_plan.max_select,
        action_plan.exact_select,
        action_plan.required_same_suit_count,
        action_plan.pair_floor,
        1 if action_plan.has_tractor else 0,
    )
    for index, word in enumerate(words):
        writer.write_i64_at(index * _I64.size, word)


def _write_observation_sections(
    *,
    writer: _WireWriter,
    offsets: RequestSectionOffsets,
    packed: PackedObservation,
) -> None:
    component_offset = offsets.component_ids
    numeric_value_offset = offsets.numeric_values
    numeric_mask_offset = offsets.numeric_masks
    for row_index, row in enumerate(packed.component_rows):
        for column_index, value in enumerate(row):
            item_offset = (
                component_offset
                + (
                    row_index * OBSERVATION_COMPONENT_COUNT
                    + column_index
                )
                * _I64.size
            )
            writer.write_i64_at(item_offset, value)
    for row_index, row in enumerate(packed.numeric_value_rows):
        for column_index, value in enumerate(row):
            item_offset = (
                numeric_value_offset
                + (row_index * NUMERIC_FEATURE_COUNT + column_index)
                * _F32.size
            )
            writer.write_f32_at(item_offset, value)
    for row_index, row in enumerate(packed.numeric_mask_rows):
        for column_index, value in enumerate(row):
            item_offset = (
                numeric_mask_offset
                + (row_index * NUMERIC_FEATURE_COUNT + column_index)
                * _F32.size
            )
            writer.write_f32_at(item_offset, value)


def _write_action_plan_sections(
    *,
    writer: _WireWriter,
    offsets: RequestSectionOffsets,
    action_plan: ActionPlanFrame,
    trace_count: int,
    trace_steps: int,
    pair_plan_count: int,
) -> None:
    _write_i64_tuple(
        writer, offsets.available_counts, action_plan.available_counts
    )
    _write_i64_tuple(
        writer, offsets.effective_suits, action_plan.effective_suits
    )
    _write_bool_tuple(
        writer, offsets.same_suit_mask, action_plan.same_suit_mask
    )
    _write_bool_tuple(
        writer, offsets.off_suit_mask, action_plan.off_suit_mask
    )
    _write_bool_tuple(
        writer, offsets.pair_face_mask, action_plan.pair_face_mask
    )
    _write_trace_sections(
        writer=writer,
        offsets=offsets,
        action_plan=action_plan,
        trace_count=trace_count,
        trace_steps=trace_steps,
    )
    _write_pair_plan_sections(
        writer=writer,
        offsets=offsets,
        action_plan=action_plan,
        pair_plan_count=pair_plan_count,
    )


def _write_trace_sections(
    *,
    writer: _WireWriter,
    offsets: RequestSectionOffsets,
    action_plan: ActionPlanFrame,
    trace_count: int,
    trace_steps: int,
) -> None:
    for trace_index in range(trace_count):
        trace = (
            action_plan.trace_tokens[trace_index]
            if trace_index < len(action_plan.trace_tokens)
            else ()
        )
        writer.write_i64_at(
            offsets.trace_lengths + trace_index * _I64.size,
            len(trace),
        )
        writer.write_bool_at(
            offsets.trace_row_mask + trace_index,
            trace_index < len(action_plan.trace_tokens),
        )
        for step_index in range(trace_steps):
            flat_index = trace_index * trace_steps + step_index
            token_id = (
                trace[step_index] if step_index < len(trace) else 0
            )
            writer.write_i64_at(
                offsets.trace_tokens + flat_index * _I64.size,
                token_id,
            )
            writer.write_bool_at(
                offsets.trace_token_mask + flat_index,
                step_index < len(trace),
            )


def _write_pair_plan_sections(
    *,
    writer: _WireWriter,
    offsets: RequestSectionOffsets,
    action_plan: ActionPlanFrame,
    pair_plan_count: int,
) -> None:
    for plan_index in range(pair_plan_count):
        row = (
            action_plan.pair_plan_masks[plan_index]
            if plan_index < len(action_plan.pair_plan_masks)
            else tuple(False for _ in range(ACTION_FACE_COUNT))
        )
        writer.write_bool_at(
            offsets.pair_plan_row_mask + plan_index,
            plan_index < len(action_plan.pair_plan_masks),
        )
        for face_index, value in enumerate(row):
            writer.write_bool_at(
                offsets.pair_plan_masks
                + plan_index * ACTION_FACE_COUNT
                + face_index,
                value,
            )


def _write_sampling_thresholds(
    *,
    writer: _WireWriter,
    offsets: RequestSectionOffsets,
    decision_key: PolicyDecisionKey,
) -> None:
    for argument_index in range(SEMANTIC_CODEC.max_argument_tokens):
        writer.write_f64_at(
            offsets.sampling_thresholds + argument_index * _F64.size,
            policy_choice_threshold(
                key=decision_key,
                argument_index=argument_index,
            ),
        )


def _write_i64_tuple(
    writer: _WireWriter, offset: int, values: tuple[int, ...]
) -> None:
    for index, value in enumerate(values):
        writer.write_i64_at(offset + index * _I64.size, value)


def _write_bool_tuple(
    writer: _WireWriter, offset: int, values: tuple[bool, ...]
) -> None:
    for index, value in enumerate(values):
        writer.write_bool_at(offset + index, value)


def _validate_wire_sizes(
    *, packed: PackedObservation, action_plan: ActionPlanFrame
) -> Ok[None] | Rejected:
    trace_count = _trace_count(action_plan)
    if trace_count > WIRE_MAX_TRACE_COUNT:
        return Rejected(reason="policy request has too many traces")
    pair_plan_count = _pair_plan_count(action_plan)
    if pair_plan_count > WIRE_MAX_PAIR_PLAN_COUNT:
        return Rejected(reason="policy request has too many pair plans")
    if packed.token_count() <= 0:
        return Rejected(reason="policy request observation is empty")
    if any(
        len(row) != NUMERIC_FEATURE_COUNT
        for row in packed.numeric_value_rows
    ):
        return Rejected(
            reason="policy request numeric value width is invalid"
        )
    if any(
        len(row) != NUMERIC_FEATURE_COUNT
        for row in packed.numeric_mask_rows
    ):
        return Rejected(
            reason="policy request numeric mask width is invalid"
        )
    return Ok(value=None)


def _trace_count(action_plan: ActionPlanFrame) -> int:
    return max(len(action_plan.trace_tokens), 1)


def _trace_steps(action_plan: ActionPlanFrame) -> int:
    if not action_plan.trace_tokens:
        return 1
    return max(len(trace) for trace in action_plan.trace_tokens)


def _pair_plan_count(action_plan: ActionPlanFrame) -> int:
    return max(len(action_plan.pair_plan_masks), 1)


def _read_i64(data: WireBuffer, word_index: int) -> int:
    values = _I64.unpack_from(data, word_index * _I64.size)
    return int(values[0])


def _align(offset: int, alignment: int) -> int:
    remainder = offset % alignment
    if remainder == 0:
        return offset
    return offset + alignment - remainder
