"""Framed policy client backed by a model-rank process."""

from __future__ import annotations

import asyncio

from server.result import Ok, Rejected
from server.training.legal_actions import LegalActionIndex
from server.training.observation import Observation
from server.training.policy import PolicyDecision
from server.training.policy_inference_wire import (
    CompletedPolicyResponse,
    RejectedPolicyResponse,
    build_policy_request_wire,
    decode_policy_response,
    decode_policy_response_wire,
)
from server.training.runtime.model_rank.inference_transport import (
    ConnectionPolicyRequestSender,
    ConnectionPolicyResponseReceiver,
)
from server.training.sampling import PolicyDecisionKey


class FramedPolicyClient:
    """Worker-side facade for delegated model-rank inference."""

    def __init__(
        self,
        *,
        worker_index: int,
        request_sender: ConnectionPolicyRequestSender,
        response_receiver: ConnectionPolicyResponseReceiver,
        timeout_seconds: float,
    ) -> None:
        self._worker_index = worker_index
        self._request_sender = request_sender
        self._response_receiver = response_receiver
        self._timeout_seconds = timeout_seconds
        self._next_request_id = 0

    async def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
        decision_key: PolicyDecisionKey,
    ) -> Ok[PolicyDecision] | Rejected:
        request_id = self._next_request_id
        self._next_request_id += 1
        request_result = build_policy_request_wire(
            worker_index=self._worker_index,
            request_id=request_id,
            observation=observation,
            legal_actions=legal_actions,
            decision_key=decision_key,
        )
        if isinstance(request_result, Rejected):
            return request_result
        send_result = self._request_sender.send(request_result.value)
        if isinstance(send_result, Rejected):
            return send_result
        response_result = await asyncio.to_thread(
            self._response_receiver.receive,
            timeout_seconds=self._timeout_seconds,
        )
        if isinstance(response_result, Rejected):
            return response_result
        response = decode_policy_response_wire(
            response_result.value.data
        )
        if isinstance(response, Rejected):
            return response
        route = response.value.route
        if route.worker_index != self._worker_index:
            return Rejected(
                reason="model rank inference response worker mismatch"
            )
        if route.request_id != request_id:
            return Rejected(
                reason="model rank inference response request mismatch"
            )
        if isinstance(response.value, RejectedPolicyResponse):
            return Rejected(reason=response.value.reason)
        assert isinstance(response.value, CompletedPolicyResponse)
        return decode_policy_response(
            legal_actions=legal_actions,
            response=response.value,
        )
