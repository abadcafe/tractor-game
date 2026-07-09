"""Connection-backed raw inference transport."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.context import BufferTooShort

from server import result as _result
from server.result import Ok, Rejected
from server.training.policy_inference_batch import (
    PolicyResponseBatchWire,
)
from server.training.policy_inference_batch.types import (
    PolicyRequestWireFrame,
)


@dataclass(frozen=True, slots=True)
class ConnectionPolicyRequestSender:
    """Worker-side request sender over a binary connection."""

    connection: Connection

    def send(
        self, request: PolicyRequestWireFrame
    ) -> Ok[None] | Rejected:
        """Send one raw inference request batch frame."""
        try:
            self.connection.send_bytes(request.view())
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

    def receive_batch_bytes_into(
        self, buffer: memoryview
    ) -> _result.Ok[int] | _result.Rejected:
        """Receive one request batch into a caller-owned slot."""
        try:
            byte_count = self.connection.recv_bytes_into(buffer)
        except BufferTooShort:
            return Rejected(
                reason="policy request batch wire exceeds slot"
            )
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
    ) -> _result.Ok[PolicyResponseBatchWire] | _result.Rejected:
        """Receive one raw inference response batch frame."""
        if not self.connection.poll(timeout_seconds):
            return Rejected(
                reason="model rank policy inference timed out"
            )
        return _receive_response(self.connection)


@dataclass(frozen=True, slots=True)
class ConnectionPolicyResponseSender:
    """Model-rank response sender for one assigned worker."""

    worker_index: int
    connection: Connection

    def __post_init__(self) -> None:
        assert self.worker_index >= 0


def send_policy_response_batch(
    *,
    sender: ConnectionPolicyResponseSender,
    response: PolicyResponseBatchWire,
) -> Ok[None] | Rejected:
    """Send one raw inference response batch frame."""
    try:
        sender.connection.send_bytes(response.data)
    except OSError as exc:
        return Rejected(
            reason=f"model-rank inference response send failed: {exc}"
        )
    return Ok(value=None)


def _receive_response(
    connection: Connection,
) -> _result.Ok[PolicyResponseBatchWire] | _result.Rejected:
    try:
        data = connection.recv_bytes()
    except (EOFError, OSError) as exc:
        reason = f"model-rank inference response receive failed: {exc}"
        return Rejected(reason=reason)
    return Ok(value=PolicyResponseBatchWire(data=data))
