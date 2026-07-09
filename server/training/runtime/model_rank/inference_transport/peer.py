"""Full-duplex async policy inference peer."""

from __future__ import annotations

from dataclasses import dataclass

from server import result as _result
from server.result import Ok, Rejected
from server.training.policy_inference_batch import (
    PolicyResponseBatchWire,
)
from server.training.policy_inference_batch.types import (
    PolicyRequestWireFrame,
)
from server.training.runtime.async_ipc import AsyncFrameEndpoint


@dataclass(frozen=True, slots=True)
class AsyncPolicyPeer:
    """One worker/model-rank inference data-plane peer."""

    worker_index: int
    endpoint: AsyncFrameEndpoint

    def __post_init__(self) -> None:
        assert self.worker_index >= 0

    async def send_request(
        self, request: PolicyRequestWireFrame
    ) -> Ok[None] | Rejected:
        """Send one worker request batch frame."""
        send_result = await self.endpoint.send_frame(request.view())
        if isinstance(send_result, Rejected):
            return Rejected(
                reason=(
                    "model-rank inference request send failed: "
                    f"{send_result.reason}"
                )
            )
        return Ok(value=None)

    async def receive_request_into(
        self, buffer: memoryview
    ) -> _result.Ok[int] | _result.Rejected:
        """Receive one request batch into caller staging memory."""
        receive_result = await self.endpoint.recv_frame_into(buffer)
        if isinstance(receive_result, Rejected):
            return Rejected(
                reason=(
                    "model-rank inference request receive failed: "
                    f"{receive_result.reason}"
                )
            )
        return receive_result

    async def send_response(
        self, response: PolicyResponseBatchWire
    ) -> Ok[None] | Rejected:
        """Send one model-rank response batch frame."""
        send_result = await self.endpoint.send_frame(response.data)
        if isinstance(send_result, Rejected):
            return Rejected(
                reason=(
                    "model-rank inference response send failed: "
                    f"{send_result.reason}"
                )
            )
        return Ok(value=None)

    async def receive_response(
        self, *, timeout_seconds: float
    ) -> _result.Ok[PolicyResponseBatchWire] | _result.Rejected:
        """Receive one response batch frame."""
        frame_result = await self.endpoint.recv_frame(
            timeout_seconds=timeout_seconds
        )
        if isinstance(frame_result, Rejected):
            return Rejected(
                reason=(
                    "model-rank inference response receive failed: "
                    f"{frame_result.reason}"
                )
            )
        return Ok(
            value=PolicyResponseBatchWire(data=frame_result.value)
        )

    def close(self) -> None:
        """Close this policy data-plane peer."""
        self.endpoint.close()
