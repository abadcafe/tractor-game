"""Framed policy client backed by a model-rank process."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

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

type PolicyDecisionResult = Ok[PolicyDecision] | Rejected


@dataclass(slots=True)
class _PendingPolicyRequest:
    legal_actions: LegalActionIndex
    future: asyncio.Future[PolicyDecisionResult]


class FramedPolicyClient:
    """Worker-side facade for delegated model-rank inference."""

    def __init__(
        self,
        *,
        worker_index: int,
        max_observation_tokens: int,
        request_sender: ConnectionPolicyRequestSender,
        response_receiver: ConnectionPolicyResponseReceiver,
        timeout_seconds: float,
    ) -> None:
        self._worker_index = worker_index
        self._max_observation_tokens = max_observation_tokens
        self._request_sender = request_sender
        self._response_receiver = response_receiver
        self._timeout_seconds = timeout_seconds
        self._next_request_id = 0
        self._pending: dict[int, _PendingPolicyRequest] = {}
        self._reader_task: asyncio.Task[None] | None = None

    async def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
        decision_key: PolicyDecisionKey,
    ) -> PolicyDecisionResult:
        loop = asyncio.get_running_loop()
        request_id = self._next_request_id
        self._next_request_id += 1
        future: asyncio.Future[PolicyDecisionResult] = (
            loop.create_future()
        )
        request_result = build_policy_request_wire(
            max_observation_tokens=self._max_observation_tokens,
            worker_index=self._worker_index,
            request_id=request_id,
            observation=observation,
            legal_actions=legal_actions,
            decision_key=decision_key,
        )
        if isinstance(request_result, Rejected):
            return request_result
        self._pending[request_id] = _PendingPolicyRequest(
            legal_actions=legal_actions,
            future=future,
        )
        self._ensure_reader_task()
        send_result = await asyncio.to_thread(
            self._request_sender.send, request_result.value
        )
        if isinstance(send_result, Rejected):
            self._pending.pop(request_id, None)
            return send_result
        try:
            return await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self._timeout_seconds,
            )
        except asyncio.CancelledError:
            self._pending.pop(request_id, None)
            future.cancel()
            raise
        except TimeoutError:
            self._pending.pop(request_id, None)
            future.cancel()
            return Rejected(
                reason="model rank policy inference timed out"
            )
        finally:
            if future.cancelled():
                self._pending.pop(request_id, None)

    def _ensure_reader_task(self) -> None:
        task = self._reader_task
        if task is not None and not task.done():
            return
        self._reader_task = asyncio.create_task(self._read_responses())

    async def _read_responses(self) -> None:
        while self._pending:
            response_result = await asyncio.to_thread(
                self._response_receiver.receive,
                timeout_seconds=self._timeout_seconds,
            )
            if isinstance(response_result, Rejected):
                self._reject_all(response_result.reason)
                return
            response = decode_policy_response_wire(
                response_result.value.data
            )
            if isinstance(response, Rejected):
                self._reject_all(response.reason)
                return
            route = response.value.route
            if route.worker_index != self._worker_index:
                self._reject_all(
                    "model rank inference response worker mismatch"
                )
                return
            pending = self._pending.pop(route.request_id, None)
            if pending is None:
                continue
            if pending.future.done():
                continue
            if isinstance(response.value, RejectedPolicyResponse):
                pending.future.set_result(
                    Rejected(reason=response.value.reason)
                )
                continue
            assert isinstance(response.value, CompletedPolicyResponse)
            decoded = decode_policy_response(
                legal_actions=pending.legal_actions,
                response=response.value,
            )
            pending.future.set_result(decoded)

    def _reject_all(self, reason: str) -> None:
        pending_items = tuple(self._pending.values())
        self._pending.clear()
        rejection = Rejected(reason=reason)
        for pending in pending_items:
            if pending.future.done():
                continue
            pending.future.set_result(rejection)
