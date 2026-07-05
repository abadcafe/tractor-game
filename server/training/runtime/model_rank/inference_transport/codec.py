"""Binary codec for inference transport envelopes."""

from __future__ import annotations

from server import result as _result
from server.result import Ok, Rejected
from server.training.policy_request_frame import (
    decode_policy_response_frame,
    encode_policy_response_frame,
)

from .messages import (
    PolicyInferenceRequestControl,
    PolicyInferenceResponseEnvelope,
)

_HEADER_BYTES = 16


def encode_request_envelope(
    envelope: PolicyInferenceRequestControl,
) -> bytes:
    """Encode one shared-memory request control frame."""
    return (
        envelope.worker_index.to_bytes(8, "little", signed=True)
        + envelope.request_id.to_bytes(8, "little", signed=True)
        + envelope.byte_count.to_bytes(8, "little", signed=True)
        + _encode_text(envelope.slot_name)
    )


def decode_request_envelope(
    data: bytes,
) -> _result.Ok[PolicyInferenceRequestControl] | _result.Rejected:
    """Decode one shared-memory request control frame."""
    header_result = _decode_header(data)
    if isinstance(header_result, Rejected):
        return header_result
    reader = _ControlReader(data=data, offset=_HEADER_BYTES)
    byte_count_result = reader.read_i64()
    if isinstance(byte_count_result, Rejected):
        return byte_count_result
    slot_name_result = reader.read_text()
    if isinstance(slot_name_result, Rejected):
        return slot_name_result
    finish_result = reader.finish()
    if isinstance(finish_result, Rejected):
        return finish_result
    return Ok(
        value=PolicyInferenceRequestControl(
            worker_index=header_result.value.worker_index,
            request_id=header_result.value.request_id,
            byte_count=byte_count_result.value,
            slot_name=slot_name_result.value,
        )
    )


def encode_response_envelope(
    envelope: PolicyInferenceResponseEnvelope,
) -> bytes:
    """Encode one response envelope."""
    return (
        envelope.worker_index.to_bytes(8, "little", signed=True)
        + envelope.request_id.to_bytes(8, "little", signed=True)
        + encode_policy_response_frame(envelope.frame)
    )


def decode_response_envelope(
    data: bytes,
) -> _result.Ok[PolicyInferenceResponseEnvelope] | _result.Rejected:
    """Decode one response envelope."""
    header_result = _decode_header(data)
    if isinstance(header_result, Rejected):
        return header_result
    payload_result = decode_policy_response_frame(data[_HEADER_BYTES:])
    if isinstance(payload_result, Rejected):
        return payload_result
    return Ok(
        value=PolicyInferenceResponseEnvelope(
            worker_index=header_result.value.worker_index,
            request_id=header_result.value.request_id,
            frame=payload_result.value,
        )
    )


class _Header:
    def __init__(self, *, worker_index: int, request_id: int) -> None:
        self.worker_index = worker_index
        self.request_id = request_id


def _decode_header(data: bytes) -> Ok[_Header] | Rejected:
    if len(data) < _HEADER_BYTES:
        return Rejected(reason="inference transport frame is truncated")
    worker_index = int.from_bytes(data[0:8], "little", signed=True)
    request_id = int.from_bytes(data[8:16], "little", signed=True)
    if worker_index < 0 or request_id < 0:
        return Rejected(reason="inference transport header is invalid")
    return Ok(
        value=_Header(worker_index=worker_index, request_id=request_id)
    )


def _encode_text(value: str) -> bytes:
    data = value.encode("utf-8")
    return len(data).to_bytes(8, "little", signed=True) + data


class _ControlReader:
    def __init__(self, *, data: bytes, offset: int) -> None:
        self._data = data
        self._offset = offset

    def finish(self) -> Ok[None] | Rejected:
        if self._offset == len(self._data):
            return Ok(value=None)
        return Rejected(
            reason="inference transport frame has trailing bytes"
        )

    def read_i64(self) -> Ok[int] | Rejected:
        if not self._has(8):
            return Rejected(
                reason="inference transport frame is truncated"
            )
        value = int.from_bytes(
            self._data[self._offset : self._offset + 8],
            "little",
            signed=True,
        )
        self._offset += 8
        return Ok(value=value)

    def read_text(self) -> Ok[str] | Rejected:
        length_result = self.read_i64()
        if isinstance(length_result, Rejected):
            return length_result
        length = length_result.value
        if length <= 0:
            return Rejected(
                reason="inference transport text length is invalid"
            )
        if not self._has(length):
            return Rejected(
                reason="inference transport frame is truncated"
            )
        try:
            value = self._data[
                self._offset : self._offset + length
            ].decode("utf-8")
        except UnicodeDecodeError:
            return Rejected(
                reason="inference transport text is invalid UTF-8"
            )
        self._offset += length
        return Ok(value=value)

    def _has(self, size: int) -> bool:
        assert size >= 0
        return self._offset + size <= len(self._data)
