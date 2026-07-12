"""Columnar policy inference response wire codec."""

from __future__ import annotations

import struct

from server.foundation import result as _result
from server.foundation.result import Ok, Rejected
from server.training.legal_actions import LegalActionIndex
from server.training.policy import PolicyDecision
from server.training.policy_inference_batch.response_types import (
    CompletedPolicyResponse,
    PolicyRequestRoute,
    PolicyResponse,
    PolicyResponseBatchWire,
    RejectedPolicyResponse,
)
from server.training.policy_sampling import (
    CompactPolicyDecisionBatch,
    CompactTraceTokenIds,
    DecisionHandle,
)
from server.training.semantic_action_plan import (
    semantic_trace_from_token_ids,
)

WIRE_RESPONSE_BATCH_MAGIC = 0x5452504F4C4F4332

_I64 = struct.Struct("<q")
_STATUS_COMPLETED = 1
_STATUS_REJECTED = 2
_HEADER_WORD_COUNT = 5
_HEADER_BYTES = _HEADER_WORD_COUNT * _I64.size

_MAGIC_INDEX = 0
_TOTAL_BYTES_INDEX = 1
_ROW_COUNT_INDEX = 2
_MAX_TRACE_COUNT_INDEX = 3
_REASON_BYTES_INDEX = 4


def encode_policy_response_batch_wire(
    responses: tuple[PolicyResponse, ...],
) -> Ok[PolicyResponseBatchWire] | Rejected:
    """Encode one columnar response batch frame."""
    if not responses:
        return Rejected(reason="policy response batch is empty")
    layout = _response_layout(
        row_count=len(responses),
        max_trace_count=_max_trace_count(responses),
        reason_byte_count=_reason_byte_count(responses),
    )
    data = bytearray(layout.total_bytes)
    _write_header(
        data=data,
        row_count=len(responses),
        max_trace_count=layout.max_trace_count,
        reason_byte_count=layout.reason_byte_count,
    )
    trace_base = layout.trace_tokens_offset
    reason_offset = 0
    for row_index, response in enumerate(responses):
        _write_i64_column(
            data,
            layout.worker_indices_offset,
            row_index,
            response.route.worker_index,
        )
        _write_i64_column(
            data,
            layout.request_ids_offset,
            row_index,
            response.route.request_id,
        )
        if isinstance(response, RejectedPolicyResponse):
            reason_bytes = response.reason.encode("utf-8")
            _write_rejected_row(
                data=data,
                layout=layout,
                row_index=row_index,
                reason_offset=reason_offset,
                reason_length=len(reason_bytes),
            )
            start = layout.reason_bytes_offset + reason_offset
            data[start : start + len(reason_bytes)] = reason_bytes
            reason_offset += len(reason_bytes)
            continue
        _write_completed_row(
            data=data,
            layout=layout,
            row_index=row_index,
            response=response,
        )
        trace_start = (
            trace_base + row_index * layout.max_trace_count * _I64.size
        )
        trace_bytes = response.trace_token_ids.encoded_i64
        data[trace_start : trace_start + len(trace_bytes)] = trace_bytes
    return Ok(value=PolicyResponseBatchWire(data=bytes(data)))


def build_completed_policy_responses(
    *,
    routes: tuple[PolicyRequestRoute, ...],
    decisions: CompactPolicyDecisionBatch,
) -> Ok[tuple[CompletedPolicyResponse, ...]] | Rejected:
    """Build in-process response rows from compact decisions."""
    if len(routes) != decisions.row_count():
        return Rejected(reason="policy response batch route mismatch")
    if not routes:
        return Rejected(reason="policy response batch is empty")
    responses: list[CompletedPolicyResponse] = []
    for row_index, route in enumerate(routes):
        responses.append(
            CompletedPolicyResponse(
                route=route,
                trace_token_ids=(
                    decisions.trace_token_batch.compact_row(row_index)
                ),
                decision_handle_model_rank=decisions.model_rank_index,
                decision_handle_policy_version=(
                    decisions.policy_versions[row_index]
                ),
                decision_handle_row_index=decisions.row_indices[
                    row_index
                ],
                choice_count=decisions.choice_counts[row_index],
            )
        )
    return Ok(value=tuple(responses))


def build_rejected_policy_responses(
    *, routes: tuple[PolicyRequestRoute, ...], reason: str
) -> Ok[tuple[RejectedPolicyResponse, ...]] | Rejected:
    """Build in-process rejected response rows."""
    assert reason
    if not routes:
        return Rejected(reason="policy response batch is empty")
    return Ok(
        value=tuple(
            RejectedPolicyResponse(route=route, reason=reason)
            for route in routes
        )
    )


def decode_policy_response_batch_wire(
    data: bytes,
) -> _result.Ok[tuple[PolicyResponse, ...]] | _result.Rejected:
    """Decode one response batch frame into response rows."""
    shape_result = _decode_shape(data)
    if isinstance(shape_result, Rejected):
        return shape_result
    layout = shape_result.value
    responses: list[PolicyResponse] = []
    for row_index in range(layout.row_count):
        route_result = _decode_route(
            data=data, layout=layout, row_index=row_index
        )
        if isinstance(route_result, Rejected):
            return route_result
        status = _read_i64_column(
            data, layout.statuses_offset, row_index
        )
        if status == _STATUS_REJECTED:
            rejected_result = _decode_rejected_response(
                data=data,
                layout=layout,
                row_index=row_index,
                route=route_result.value,
            )
            if isinstance(rejected_result, Rejected):
                return rejected_result
            responses.append(rejected_result.value)
            continue
        if status != _STATUS_COMPLETED:
            return Rejected(reason="policy response status is invalid")
        completed_result = _decode_completed_response(
            data=data,
            layout=layout,
            row_index=row_index,
            route=route_result.value,
        )
        if isinstance(completed_result, Rejected):
            return completed_result
        responses.append(completed_result.value)
    return Ok(value=tuple(responses))


def decode_policy_response(
    *,
    legal_actions: LegalActionIndex,
    response: PolicyResponse,
) -> Ok[PolicyDecision] | Rejected:
    """Decode a response through the worker-side rule index."""
    if isinstance(response, RejectedPolicyResponse):
        return Rejected(reason=response.reason)
    trace_result = semantic_trace_from_token_ids(
        response.trace_token_ids.to_tuple()
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
                row_index=response.decision_handle_row_index,
            ),
            choice_count=response.choice_count,
        )
    )


class _ResponseLayout:
    """Byte offsets for one decoded columnar response frame."""

    def __init__(
        self,
        *,
        row_count: int,
        max_trace_count: int,
        reason_byte_count: int,
    ) -> None:
        assert row_count > 0
        assert max_trace_count >= 0
        assert reason_byte_count >= 0
        self.row_count = row_count
        self.max_trace_count = max_trace_count
        self.reason_byte_count = reason_byte_count
        offset = _HEADER_BYTES
        self.worker_indices_offset = offset
        offset += row_count * _I64.size
        self.request_ids_offset = offset
        offset += row_count * _I64.size
        self.statuses_offset = offset
        offset += row_count * _I64.size
        self.model_rank_indices_offset = offset
        offset += row_count * _I64.size
        self.policy_versions_offset = offset
        offset += row_count * _I64.size
        self.row_indices_offset = offset
        offset += row_count * _I64.size
        self.choice_counts_offset = offset
        offset += row_count * _I64.size
        self.trace_counts_offset = offset
        offset += row_count * _I64.size
        self.reason_offsets_offset = offset
        offset += row_count * _I64.size
        self.reason_lengths_offset = offset
        offset += row_count * _I64.size
        self.trace_tokens_offset = offset
        offset += row_count * max_trace_count * _I64.size
        self.reason_bytes_offset = offset
        self.total_bytes = offset + reason_byte_count


def _max_trace_count(responses: tuple[PolicyResponse, ...]) -> int:
    return max(
        (
            len(response.trace_token_ids)
            for response in responses
            if isinstance(response, CompletedPolicyResponse)
        ),
        default=0,
    )


def _reason_byte_count(responses: tuple[PolicyResponse, ...]) -> int:
    return sum(
        len(response.reason.encode("utf-8"))
        for response in responses
        if isinstance(response, RejectedPolicyResponse)
    )


def _response_layout(
    *, row_count: int, max_trace_count: int, reason_byte_count: int
) -> _ResponseLayout:
    return _ResponseLayout(
        row_count=row_count,
        max_trace_count=max_trace_count,
        reason_byte_count=reason_byte_count,
    )


def _write_header(
    *,
    data: bytearray,
    row_count: int,
    max_trace_count: int,
    reason_byte_count: int,
) -> None:
    words = (
        WIRE_RESPONSE_BATCH_MAGIC,
        len(data),
        row_count,
        max_trace_count,
        reason_byte_count,
    )
    for index, word in enumerate(words):
        _I64.pack_into(data, index * _I64.size, word)


def _write_completed_row(
    *,
    data: bytearray,
    layout: _ResponseLayout,
    row_index: int,
    response: CompletedPolicyResponse,
) -> None:
    _write_i64_column(
        data, layout.statuses_offset, row_index, _STATUS_COMPLETED
    )
    _write_i64_column(
        data,
        layout.model_rank_indices_offset,
        row_index,
        response.decision_handle_model_rank,
    )
    _write_i64_column(
        data,
        layout.policy_versions_offset,
        row_index,
        response.decision_handle_policy_version,
    )
    _write_i64_column(
        data,
        layout.row_indices_offset,
        row_index,
        response.decision_handle_row_index,
    )
    _write_i64_column(
        data,
        layout.choice_counts_offset,
        row_index,
        response.choice_count,
    )
    _write_i64_column(
        data,
        layout.trace_counts_offset,
        row_index,
        len(response.trace_token_ids),
    )


def _write_rejected_row(
    *,
    data: bytearray,
    layout: _ResponseLayout,
    row_index: int,
    reason_offset: int,
    reason_length: int,
) -> None:
    _write_i64_column(
        data, layout.statuses_offset, row_index, _STATUS_REJECTED
    )
    _write_i64_column(
        data, layout.reason_offsets_offset, row_index, reason_offset
    )
    _write_i64_column(
        data, layout.reason_lengths_offset, row_index, reason_length
    )


def _decode_shape(data: bytes) -> Ok[_ResponseLayout] | Rejected:
    if len(data) < _HEADER_BYTES:
        return Rejected(
            reason="policy response batch wire is truncated"
        )
    if _read_i64(data, _MAGIC_INDEX) != WIRE_RESPONSE_BATCH_MAGIC:
        return Rejected(
            reason="policy response batch wire schema is invalid"
        )
    total_bytes = _read_i64(data, _TOTAL_BYTES_INDEX)
    if total_bytes != len(data):
        return Rejected(
            reason="policy response batch wire length mismatch"
        )
    row_count = _read_i64(data, _ROW_COUNT_INDEX)
    max_trace_count = _read_i64(data, _MAX_TRACE_COUNT_INDEX)
    reason_byte_count = _read_i64(data, _REASON_BYTES_INDEX)
    if row_count <= 0:
        return Rejected(reason="policy response batch is empty")
    if max_trace_count < 0 or reason_byte_count < 0:
        return Rejected(reason="policy response batch shape is invalid")
    layout = _response_layout(
        row_count=row_count,
        max_trace_count=max_trace_count,
        reason_byte_count=reason_byte_count,
    )
    if layout.total_bytes != len(data):
        return Rejected(
            reason="policy response batch wire length mismatch"
        )
    return Ok(value=layout)


def _decode_route(
    *, data: bytes, layout: _ResponseLayout, row_index: int
) -> Ok[PolicyRequestRoute] | Rejected:
    worker_index = _read_i64_column(
        data, layout.worker_indices_offset, row_index
    )
    request_id = _read_i64_column(
        data, layout.request_ids_offset, row_index
    )
    if worker_index < 0 or request_id < 0:
        return Rejected(reason="policy response route is invalid")
    return Ok(
        value=PolicyRequestRoute(
            worker_index=worker_index,
            request_id=request_id,
        )
    )


def _decode_rejected_response(
    *,
    data: bytes,
    layout: _ResponseLayout,
    row_index: int,
    route: PolicyRequestRoute,
) -> Ok[PolicyResponse] | Rejected:
    reason_offset = _read_i64_column(
        data, layout.reason_offsets_offset, row_index
    )
    reason_length = _read_i64_column(
        data, layout.reason_lengths_offset, row_index
    )
    if reason_offset < 0 or reason_length <= 0:
        return Rejected(
            reason="policy response reason length is invalid"
        )
    start = layout.reason_bytes_offset + reason_offset
    end = start + reason_length
    if end > layout.total_bytes:
        return Rejected(
            reason="policy response reason length is invalid"
        )
    try:
        reason = data[start:end].decode("utf-8")
    except UnicodeDecodeError:
        return Rejected(
            reason="policy response reason is invalid UTF-8"
        )
    return Ok(value=RejectedPolicyResponse(route=route, reason=reason))


def _decode_completed_response(
    *,
    data: bytes,
    layout: _ResponseLayout,
    row_index: int,
    route: PolicyRequestRoute,
) -> Ok[PolicyResponse] | Rejected:
    trace_count = _read_i64_column(
        data, layout.trace_counts_offset, row_index
    )
    if trace_count <= 0 or trace_count > layout.max_trace_count:
        return Rejected(reason="policy response trace length mismatch")
    model_rank_index = _read_i64_column(
        data, layout.model_rank_indices_offset, row_index
    )
    policy_version = _read_i64_column(
        data, layout.policy_versions_offset, row_index
    )
    row_index_value = _read_i64_column(
        data, layout.row_indices_offset, row_index
    )
    choice_count = _read_i64_column(
        data, layout.choice_counts_offset, row_index
    )
    if (
        model_rank_index < 0
        or policy_version < 0
        or row_index_value < 0
        or choice_count <= 0
    ):
        return Rejected(reason="policy response handle is invalid")
    trace_start = (
        layout.trace_tokens_offset
        + row_index * layout.max_trace_count * _I64.size
    )
    trace_token_ids = CompactTraceTokenIds.from_i64_bytes(
        data=data[trace_start : trace_start + trace_count * _I64.size],
        count=trace_count,
    )
    return Ok(
        value=CompletedPolicyResponse(
            route=route,
            trace_token_ids=trace_token_ids,
            decision_handle_model_rank=model_rank_index,
            decision_handle_policy_version=policy_version,
            decision_handle_row_index=row_index_value,
            choice_count=choice_count,
        )
    )


def _write_i64_column(
    data: bytearray, column_offset: int, row_index: int, value: int
) -> None:
    _I64.pack_into(data, column_offset + row_index * _I64.size, value)


def _read_i64(data: bytes, word_index: int) -> int:
    return _read_i64_at(data, word_index * _I64.size)


def _read_i64_column(
    data: bytes, column_offset: int, row_index: int
) -> int:
    return _read_i64_at(data, column_offset + row_index * _I64.size)


def _read_i64_at(data: bytes, offset: int) -> int:
    values = _I64.unpack_from(data, offset)
    return int(values[0])
