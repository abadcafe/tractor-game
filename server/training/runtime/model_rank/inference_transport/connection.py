"""Connection-backed raw inference transport."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.context import BufferTooShort

from server import result as _result
from server.result import Ok, Rejected
from server.training.policy_inference_wire import (
    PolicyRequestWire,
    PolicyResponseWire,
)


@dataclass(frozen=True, slots=True)
class ConnectionPolicyRequestSender:
    """Worker-side request sender over a binary connection."""

    connection: Connection

    def send(self, request: PolicyRequestWire) -> Ok[None] | Rejected:
        """Send one raw inference request."""
        try:
            self.connection.send_bytes(request.data)
        except OSError as exc:
            return Rejected(
                reason=(
                    f"model-rank inference request send failed: {exc}"
                )
            )
        return Ok(value=None)


@dataclass(frozen=True, slots=True)
class ConnectionPolicyRequestReceiver:
    """Model-rank request receiver over a binary connection."""

    connection: Connection

    def receive_bytes_into(
        self, buffer: memoryview
    ) -> _result.Ok[int] | _result.Rejected:
        """Receive one request directly into a caller-owned slot."""
        try:
            byte_count = self.connection.recv_bytes_into(buffer)
        except BufferTooShort:
            return Rejected(reason="policy request wire exceeds slot")
        except (EOFError, OSError) as exc:
            return Rejected(
                reason=(
                    "model-rank inference request receive failed: "
                    f"{exc}"
                )
            )
        return Ok(value=byte_count)


@dataclass(frozen=True, slots=True)
class ConnectionPolicyResponseReceiver:
    """Worker-side response receiver over a binary connection."""

    connection: Connection

    def receive(
        self, *, timeout_seconds: float
    ) -> _result.Ok[PolicyResponseWire] | _result.Rejected:
        """Receive one raw inference response."""
        if not self.connection.poll(timeout_seconds):
            return Rejected(
                reason="model rank policy inference timed out"
            )
        return _receive_response(self.connection)


def send_policy_response(
    *,
    sender: Connection,
    response: PolicyResponseWire,
) -> Ok[None] | Rejected:
    """Send one raw inference response."""
    try:
        sender.send_bytes(response.data)
    except OSError as exc:
        return Rejected(
            reason=f"model-rank inference response send failed: {exc}"
        )
    return Ok(value=None)


def _receive_response(
    connection: Connection,
) -> _result.Ok[PolicyResponseWire] | _result.Rejected:
    try:
        data = connection.recv_bytes()
    except (EOFError, OSError) as exc:
        reason = f"model-rank inference response receive failed: {exc}"
        return Rejected(reason=reason)
    return Ok(value=PolicyResponseWire(data=data))
