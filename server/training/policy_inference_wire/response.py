"""Policy inference response wire codec."""

from __future__ import annotations

import struct

from server import result as _result
from server.result import Ok, Rejected
from server.training.legal_actions import LegalActionIndex
from server.training.policy import PolicyDecision
from server.training.policy_sampling import (
    DecisionHandle,
    ModelRankPolicyDecision,
)
from server.training.semantic_action_plan import (
    semantic_trace_from_token_ids,
)

from .types import (
    CompletedPolicyResponse,
    PolicyRequestRoute,
    PolicyResponse,
    PolicyResponseWire,
    RejectedPolicyResponse,
)

WIRE_RESPONSE_MAGIC = 0x5452504F4C4F5231

_I64 = struct.Struct("<q")
_STATUS_COMPLETED = 1
_STATUS_REJECTED = 2
_HEADER_WORD_COUNT = 12
_HEADER_BYTES = _HEADER_WORD_COUNT * _I64.size

_MAGIC_INDEX = 0
_TOTAL_BYTES_INDEX = 1
_WORKER_INDEX_INDEX = 2
_REQUEST_ID_INDEX = 3
_STATUS_INDEX = 4
_MODEL_RANK_INDEX = 5
_POLICY_VERSION_INDEX = 6
_SLOT_INDEX_INDEX = 7
_SLOT_GENERATION_INDEX = 8
_CHOICE_COUNT_INDEX = 9
_TRACE_COUNT_INDEX = 10
_REASON_BYTES_INDEX = 11


def build_completed_policy_response_wire(
    *,
    route: PolicyRequestRoute,
    decision: ModelRankPolicyDecision,
) -> PolicyResponseWire:
    """Build one successful response wire message."""
    trace_count = len(decision.trace_token_ids)
    total_bytes = _HEADER_BYTES + trace_count * _I64.size
    data = bytearray(total_bytes)
    _write_header(
        data=data,
        total_bytes=total_bytes,
        route=route,
        status=_STATUS_COMPLETED,
        model_rank_index=decision.decision_handle.model_rank_index,
        policy_version=decision.decision_handle.policy_version,
        slot_index=decision.decision_handle.slot_index,
        slot_generation=decision.decision_handle.slot_generation,
        choice_count=decision.choice_count,
        trace_count=trace_count,
        reason_bytes=0,
    )
    for index, token_id in enumerate(decision.trace_token_ids):
        _I64.pack_into(
            data, _HEADER_BYTES + index * _I64.size, token_id
        )
    return PolicyResponseWire(data=bytes(data))


def build_rejected_policy_response_wire(
    *, route: PolicyRequestRoute, reason: str
) -> PolicyResponseWire:
    """Build one rejected response wire message."""
    assert reason
    encoded = reason.encode("utf-8")
    total_bytes = _HEADER_BYTES + len(encoded)
    data = bytearray(total_bytes)
    _write_header(
        data=data,
        total_bytes=total_bytes,
        route=route,
        status=_STATUS_REJECTED,
        model_rank_index=0,
        policy_version=0,
        slot_index=0,
        slot_generation=0,
        choice_count=0,
        trace_count=0,
        reason_bytes=len(encoded),
    )
    data[_HEADER_BYTES:] = encoded
    return PolicyResponseWire(data=bytes(data))


def decode_policy_response_wire(
    data: bytes,
) -> _result.Ok[PolicyResponse] | _result.Rejected:
    """Decode one response wire message."""
    if len(data) < _HEADER_BYTES:
        return Rejected(reason="policy response wire is truncated")
    if _read_i64(data, _MAGIC_INDEX) != WIRE_RESPONSE_MAGIC:
        return Rejected(reason="policy response wire schema is invalid")
    total_bytes = _read_i64(data, _TOTAL_BYTES_INDEX)
    if total_bytes != len(data):
        return Rejected(reason="policy response wire length mismatch")
    route = PolicyRequestRoute(
        worker_index=_read_i64(data, _WORKER_INDEX_INDEX),
        request_id=_read_i64(data, _REQUEST_ID_INDEX),
    )
    status = _read_i64(data, _STATUS_INDEX)
    if status == _STATUS_REJECTED:
        reason_bytes = _read_i64(data, _REASON_BYTES_INDEX)
        if reason_bytes < 0 or _HEADER_BYTES + reason_bytes != len(
            data
        ):
            return Rejected(
                reason="policy response reason length is invalid"
            )
        try:
            reason = data[_HEADER_BYTES:].decode("utf-8")
        except UnicodeDecodeError:
            return Rejected(
                reason="policy response reason is invalid UTF-8"
            )
        return Ok(
            value=RejectedPolicyResponse(route=route, reason=reason)
        )
    if status != _STATUS_COMPLETED:
        return Rejected(reason="policy response status is invalid")
    trace_count = _read_i64(data, _TRACE_COUNT_INDEX)
    if trace_count <= 0:
        return Rejected(reason="policy response trace is empty")
    expected_bytes = _HEADER_BYTES + trace_count * _I64.size
    if expected_bytes != len(data):
        return Rejected(reason="policy response trace length mismatch")
    trace_token_ids = tuple(
        _read_i64_at(data, _HEADER_BYTES + index * _I64.size)
        for index in range(trace_count)
    )
    return Ok(
        value=CompletedPolicyResponse(
            route=route,
            trace_token_ids=trace_token_ids,
            decision_handle_model_rank=_read_i64(
                data, _MODEL_RANK_INDEX
            ),
            decision_handle_policy_version=_read_i64(
                data, _POLICY_VERSION_INDEX
            ),
            decision_handle_slot_index=_read_i64(
                data, _SLOT_INDEX_INDEX
            ),
            decision_handle_slot_generation=_read_i64(
                data, _SLOT_GENERATION_INDEX
            ),
            choice_count=_read_i64(data, _CHOICE_COUNT_INDEX),
        )
    )


def decode_policy_response(
    *,
    legal_actions: LegalActionIndex,
    response: PolicyResponse,
) -> Ok[PolicyDecision] | Rejected:
    """Decode a response through the worker-side rule index."""
    if isinstance(response, RejectedPolicyResponse):
        return Rejected(reason=response.reason)
    trace_result = semantic_trace_from_token_ids(
        response.trace_token_ids
    )
    if isinstance(trace_result, Rejected):
        return trace_result
    decoded = legal_actions.decode(trace_result.value)
    if isinstance(decoded, Rejected):
        return decoded
    return Ok(
        value=PolicyDecision(
            action=decoded.value,
            decision_handle=DecisionHandle(
                model_rank_index=response.decision_handle_model_rank,
                policy_version=response.decision_handle_policy_version,
                slot_index=response.decision_handle_slot_index,
                slot_generation=(
                    response.decision_handle_slot_generation
                ),
            ),
            choice_count=response.choice_count,
        )
    )


def _write_header(
    *,
    data: bytearray,
    total_bytes: int,
    route: PolicyRequestRoute,
    status: int,
    model_rank_index: int,
    policy_version: int,
    slot_index: int,
    slot_generation: int,
    choice_count: int,
    trace_count: int,
    reason_bytes: int,
) -> None:
    words = (
        WIRE_RESPONSE_MAGIC,
        total_bytes,
        route.worker_index,
        route.request_id,
        status,
        model_rank_index,
        policy_version,
        slot_index,
        slot_generation,
        choice_count,
        trace_count,
        reason_bytes,
    )
    for index, word in enumerate(words):
        _I64.pack_into(data, index * _I64.size, word)


def _read_i64(data: bytes, word_index: int) -> int:
    return _read_i64_at(data, word_index * _I64.size)


def _read_i64_at(data: bytes, offset: int) -> int:
    values = _I64.unpack_from(data, offset)
    return int(values[0])
