"""Connection-backed binary inference transport."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.shared_memory import SharedMemory

from server import result as _result
from server.result import Ok, Rejected
from server.training.policy_request_frame import (
    decode_policy_request_frame,
    encode_policy_request_frame,
)

from .codec import (
    decode_request_envelope,
    decode_response_envelope,
    encode_request_envelope,
    encode_response_envelope,
)
from .messages import (
    PolicyInferenceRequest,
    PolicyInferenceRequestBatch,
    PolicyInferenceRequestControl,
    PolicyInferenceResponseEnvelope,
)


@dataclass(frozen=True, slots=True)
class SharedMemoryPolicyRequestSender:
    """Worker-side request sender over a shared-memory slot."""

    connection: Connection
    slot_name: str
    slot_size: int

    def send(
        self, envelope: PolicyInferenceRequest
    ) -> Ok[None] | Rejected:
        payload = encode_policy_request_frame(envelope.frame)
        if len(payload) > self.slot_size:
            return Rejected(
                reason=(
                    "model-rank inference request exceeds shared "
                    f"memory slot: {len(payload)} > {self.slot_size}"
                )
            )
        slot_result = _open_shared_memory_slot(self.slot_name)
        if isinstance(slot_result, Rejected):
            return slot_result
        slot = slot_result.value
        try:
            buffer = _shared_memory_buffer(slot)
            buffer[: len(payload)] = payload
            del buffer
            self.connection.send_bytes(
                encode_request_envelope(
                    PolicyInferenceRequestControl(
                        worker_index=envelope.worker_index,
                        request_id=envelope.request_id,
                        byte_count=len(payload),
                        slot_name=self.slot_name,
                    )
                )
            )
        except OSError as exc:
            reason = f"model-rank inference request send failed: {exc}"
            return Rejected(reason=reason)
        finally:
            slot.close()
        return Ok(value=None)


@dataclass(frozen=True, slots=True)
class ConnectionPolicyResponseReceiver:
    """Worker-side response receiver over a binary connection."""

    connection: Connection

    def receive(
        self, *, timeout_seconds: float
    ) -> _result.Ok[PolicyInferenceResponseEnvelope] | _result.Rejected:
        if not self.connection.poll(timeout_seconds):
            return Rejected(
                reason="model rank policy inference timed out"
            )
        return _receive_response(self.connection)


def receive_policy_request_batch(
    *,
    receivers: tuple[SharedMemoryPolicyRequestReceiver, ...],
    batch_size: int,
    wait_seconds: float,
) -> _result.Ok[PolicyInferenceRequestBatch | None] | _result.Rejected:
    """Receive at most one model-rank inference batch."""
    assert batch_size > 0
    if not receivers:
        return Ok(value=None)
    ready = tuple(
        item
        for item in wait(
            tuple(receiver.connection for receiver in receivers),
            timeout=wait_seconds,
        )
        if isinstance(item, Connection)
    )
    if not ready:
        return Ok(value=None)
    requests: list[PolicyInferenceRequest] = []
    for receiver in _ready_receivers(receivers=receivers, ready=ready):
        if len(requests) >= batch_size:
            break
        request_result = receiver.receive_request()
        if isinstance(request_result, Rejected):
            return request_result
        requests.append(request_result.value)
    while len(requests) < batch_size:
        drained = False
        for receiver in receivers:
            if len(requests) >= batch_size:
                break
            if not receiver.connection.poll():
                continue
            request_result = receiver.receive_request()
            if isinstance(request_result, Rejected):
                return request_result
            requests.append(request_result.value)
            drained = True
        if not drained:
            break
    if not requests:
        return Ok(value=None)
    return Ok(value=PolicyInferenceRequestBatch(tuple(requests)))


def send_policy_response(
    *,
    sender: Connection,
    envelope: PolicyInferenceResponseEnvelope,
) -> Ok[None] | Rejected:
    """Send one model-rank inference response."""
    try:
        sender.send_bytes(encode_response_envelope(envelope))
    except OSError as exc:
        return Rejected(
            reason=f"model-rank inference response send failed: {exc}"
        )
    return Ok(value=None)


def _receive_request(
    connection: Connection,
) -> _result.Ok[PolicyInferenceRequest] | _result.Rejected:
    try:
        data = connection.recv_bytes()
    except (EOFError, OSError) as exc:
        return Rejected(
            reason=f"model-rank inference request receive failed: {exc}"
        )
    control_result = decode_request_envelope(data)
    if isinstance(control_result, Rejected):
        return control_result
    return _request_from_control(control_result.value)


def _receive_response(
    connection: Connection,
) -> _result.Ok[PolicyInferenceResponseEnvelope] | _result.Rejected:
    try:
        data = connection.recv_bytes()
    except (EOFError, OSError) as exc:
        reason = f"model-rank inference response receive failed: {exc}"
        return Rejected(reason=reason)
    return decode_response_envelope(data)


@dataclass(frozen=True, slots=True)
class SharedMemoryPolicyRequestReceiver:
    """Model-rank request receiver backed by one shared-memory slot."""

    connection: Connection

    def receive_request(
        self,
    ) -> _result.Ok[PolicyInferenceRequest] | _result.Rejected:
        return _receive_request(self.connection)


def _request_from_control(
    control: PolicyInferenceRequestControl,
) -> _result.Ok[PolicyInferenceRequest] | _result.Rejected:
    slot_result = _open_shared_memory_slot(control.slot_name)
    if isinstance(slot_result, Rejected):
        return slot_result
    slot = slot_result.value
    try:
        buffer = _shared_memory_buffer(slot)
        if control.byte_count > len(buffer):
            del buffer
            return Rejected(
                reason=(
                    "model-rank inference request slot length is "
                    "invalid"
                )
            )
        payload = bytes(buffer[: control.byte_count])
        del buffer
    finally:
        slot.close()
    frame_result = decode_policy_request_frame(payload)
    if isinstance(frame_result, Rejected):
        return frame_result
    return Ok(
        value=PolicyInferenceRequest(
            worker_index=control.worker_index,
            request_id=control.request_id,
            frame=frame_result.value,
            byte_count=control.byte_count,
        )
    )


def _open_shared_memory_slot(
    slot_name: str,
) -> Ok[SharedMemory] | Rejected:
    try:
        return Ok(value=SharedMemory(name=slot_name, track=False))
    except OSError as exc:
        return Rejected(
            reason=(
                f"model-rank inference shared memory open failed: {exc}"
            )
        )


def _shared_memory_buffer(slot: SharedMemory) -> memoryview[int]:
    buffer = slot.buf
    assert buffer is not None
    return buffer


def _ready_receivers(
    *,
    receivers: tuple[SharedMemoryPolicyRequestReceiver, ...],
    ready: tuple[Connection, ...],
) -> tuple[SharedMemoryPolicyRequestReceiver, ...]:
    return tuple(
        receiver
        for receiver in receivers
        if any(receiver.connection == item for item in ready)
    )
