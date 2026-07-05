"""Framed policy client backed by a model-rank process."""

from __future__ import annotations

from server.result import Ok, Rejected
from server.training.legal_actions import LegalActionIndex
from server.training.observation import Observation
from server.training.policy import PolicyDecision
from server.training.policy_request_frame import (
    build_policy_request_frame,
    decode_policy_response,
)
from server.training.runtime.model_rank.inference_transport import (
    ConnectionPolicyResponseReceiver,
    PolicyInferenceRequest,
    SharedMemoryPolicyRequestSender,
)
from server.training.sampling import PolicyDecisionKey


class FramedPolicyClient:
    """Synchronous worker-side facade for framed model inference."""

    def __init__(
        self,
        *,
        worker_index: int,
        request_sender: SharedMemoryPolicyRequestSender,
        response_receiver: ConnectionPolicyResponseReceiver,
        timeout_seconds: float,
    ) -> None:
        self._worker_index = worker_index
        self._request_sender = request_sender
        self._response_receiver = response_receiver
        self._timeout_seconds = timeout_seconds
        self._next_request_id = 0

    def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
        decision_key: PolicyDecisionKey,
    ) -> Ok[PolicyDecision] | Rejected:
        frame_result = build_policy_request_frame(
            observation=observation,
            legal_actions=legal_actions,
            decision_key=decision_key,
        )
        if isinstance(frame_result, Rejected):
            return frame_result
        request_id = self._next_request_id
        self._next_request_id += 1
        send_result = self._request_sender.send(
            PolicyInferenceRequest(
                worker_index=self._worker_index,
                request_id=request_id,
                frame=frame_result.value,
            )
        )
        if isinstance(send_result, Rejected):
            return send_result
        response_result = self._response_receiver.receive(
            timeout_seconds=self._timeout_seconds
        )
        if isinstance(response_result, Rejected):
            return response_result
        response = response_result.value
        if response.worker_index != self._worker_index:
            return Rejected(
                reason="model rank inference response worker mismatch"
            )
        if response.request_id != request_id:
            return Rejected(
                reason="model rank inference response request mismatch"
            )
        return decode_policy_response(
            legal_actions=legal_actions,
            response=response.frame,
        )
