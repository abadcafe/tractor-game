"""Unified batched policy client for local and process model ranks."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Protocol

import torch

from server.result import Ok, Rejected
from server.training.legal_actions import LegalActionIndex
from server.training.observation import Observation
from server.training.policy import PolicyDecision
from server.training.policy_inference_batch import (
    CompletedPolicyResponse,
    DevicePolicyRequestBatch,
    PolicyRequestBatch,
    PolicyRequestBatchBuilder,
    PolicyRequestInput,
    PolicyRequestRoute,
    PolicyResponse,
    PolicyResponseBatchWire,
    RejectedPolicyResponse,
    decode_policy_response,
    decode_policy_response_batch_wire,
    materialize_policy_request_batch,
)
from server.training.policy_inference_batch.types import (
    PolicyRequestWireFrame,
)
from server.training.policy_sampling import ModelRankPolicyDecision
from server.training.sampling import PolicyDecisionKey

type PolicyDecisionResult = Ok[PolicyDecision] | Rejected
type ModelRankDecisionResult = Ok[ModelRankPolicyDecision] | Rejected


class PolicyBatchTransport(Protocol):
    """Transport one request frame through a model-rank boundary."""

    def submit_batch(
        self, *, batch: PolicyRequestBatch
    ) -> Ok[None] | Rejected: ...

    def receive(
        self, *, timeout_seconds: float
    ) -> Ok[tuple[PolicyResponse, ...]] | Rejected: ...


class PolicyRequestSender(Protocol):
    """Send raw policy request batch frames for this worker."""

    def send(
        self, request: PolicyRequestWireFrame
    ) -> Ok[None] | Rejected: ...


class PolicyResponseReceiver(Protocol):
    """Receive raw policy response batch frames for this worker."""

    def receive(
        self, *, timeout_seconds: float
    ) -> Ok[PolicyResponseBatchWire] | Rejected: ...


class ModelReplicaProtocol(Protocol):
    """Model replica operations needed by same-process inference."""

    @property
    def device(self) -> torch.device: ...

    def decide_batch(
        self, requests: DevicePolicyRequestBatch
    ) -> tuple[ModelRankDecisionResult, ...]: ...


@dataclass(slots=True)
class ConnectionPolicyBatchTransport:
    """Connection-backed inference transport for remote model ranks."""

    request_sender: PolicyRequestSender
    response_receiver: PolicyResponseReceiver
    _encoder: PolicyRequestBatchBuilder | None = None
    _encoder_capacity: int = 0
    _encoder_max_observation_tokens: int = 0

    def submit_batch(
        self, *, batch: PolicyRequestBatch
    ) -> Ok[None] | Rejected:
        """Encode and send one request batch to the model rank."""
        encoder = self._encoder_for(batch)
        frame = encoder.encode_wire_frame(batch)
        return self.request_sender.send(frame)

    def receive(
        self, *, timeout_seconds: float
    ) -> Ok[tuple[PolicyResponse, ...]] | Rejected:
        """Receive and decode one response batch from the model rank."""
        response_result = self.response_receiver.receive(
            timeout_seconds=timeout_seconds
        )
        if isinstance(response_result, Rejected):
            return response_result
        return decode_policy_response_batch_wire(
            response_result.value.data
        )

    def _encoder_for(
        self, batch: PolicyRequestBatch
    ) -> PolicyRequestBatchBuilder:
        row_count = batch.row_count()
        if (
            self._encoder is None
            or self._encoder_capacity < row_count
            or self._encoder_max_observation_tokens
            != batch.max_observation_tokens
        ):
            self._encoder_capacity = row_count
            self._encoder_max_observation_tokens = (
                batch.max_observation_tokens
            )
            self._encoder = PolicyRequestBatchBuilder(
                batch_capacity=self._encoder_capacity,
                max_observation_tokens=(
                    self._encoder_max_observation_tokens
                ),
            )
        return self._encoder


@dataclass(slots=True)
class LocalPolicyBatchTransport:
    """Same-process inference transport for CPU worker model ranks."""

    replica: ModelReplicaProtocol
    _pending_response: tuple[PolicyResponse, ...] | None = None

    def submit_batch(
        self, *, batch: PolicyRequestBatch
    ) -> Ok[None] | Rejected:
        """Submit one prepared batch to the local model rank."""
        request_result = materialize_policy_request_batch(
            batch=batch, device=self.replica.device
        )
        if isinstance(request_result, Rejected):
            return request_result
        decisions = self.replica.decide_batch(request_result.value)
        if len(decisions) != batch.row_count():
            return Rejected(
                reason="local policy response batch mismatch"
            )
        responses = tuple(
            _policy_response_from_decision(
                route=route, decision=decision
            )
            for route, decision in zip(
                batch.routes, decisions, strict=True
            )
        )
        if self._pending_response is not None:
            return Rejected(
                reason="local policy response is still pending"
            )
        self._pending_response = responses
        return Ok(value=None)

    def receive(
        self, *, timeout_seconds: float
    ) -> Ok[tuple[PolicyResponse, ...]] | Rejected:
        """Return responses for the submitted local frame."""
        assert timeout_seconds > 0.0
        response = self._pending_response
        if response is None:
            return Rejected(
                reason="local model rank has no pending policy response"
            )
        self._pending_response = None
        return Ok(value=response)


@dataclass(slots=True)
class _PendingPolicyRequest:
    legal_actions: LegalActionIndex
    future: asyncio.Future[PolicyDecisionResult]


@dataclass(slots=True)
class _QueuedPolicyRequest:
    request_id: int
    request: PolicyRequestInput
    legal_actions: LegalActionIndex
    future: asyncio.Future[PolicyDecisionResult]


@dataclass(frozen=True, slots=True)
class _BuiltPolicyBatch:
    batch: PolicyRequestBatch
    requests: tuple[_QueuedPolicyRequest, ...]


@dataclass(frozen=True, slots=True)
class PolicyClientStats:
    """Policy wait counters accumulated by one worker client."""

    decision_count: int
    wait_seconds: float

    def __post_init__(self) -> None:
        assert self.decision_count >= 0
        assert self.wait_seconds >= 0.0


def _in_flight_request_dict() -> dict[int, _PendingPolicyRequest]:
    return {}


def _request_queue() -> list[_QueuedPolicyRequest]:
    return []


def _ignored_request_set() -> set[int]:
    return set()


def _policy_response_list() -> list[PolicyResponse]:
    return []


@dataclass(slots=True)
class BatchedPolicyClient:
    """Batch policy requests over one model-rank transport."""

    worker_index: int
    max_observation_tokens: int
    transport: PolicyBatchTransport
    timeout_seconds: float
    batch_size: int
    _next_request_id: int = 0
    _in_flight: dict[int, _PendingPolicyRequest] = field(
        default_factory=_in_flight_request_dict
    )
    _send_queue: list[_QueuedPolicyRequest] = field(
        default_factory=_request_queue
    )
    _ignored_response_request_ids: set[int] = field(
        default_factory=_ignored_request_set
    )
    _deferred_responses: list[PolicyResponse] = field(
        default_factory=_policy_response_list
    )
    _preparer: PolicyRequestBatchBuilder = field(init=False)
    _send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _send_task: asyncio.Task[None] | None = None
    _receive_task: asyncio.Task[None] | None = None
    _in_flight_event: asyncio.Event | None = None
    _decision_count: int = 0
    _wait_seconds: float = 0.0

    def __post_init__(self) -> None:
        assert self.worker_index >= 0
        assert self.max_observation_tokens > 0
        assert self.timeout_seconds > 0.0
        assert self.batch_size > 0
        self._preparer = PolicyRequestBatchBuilder(
            batch_capacity=self.batch_size,
            max_observation_tokens=self.max_observation_tokens,
        )

    async def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
        decision_key: PolicyDecisionKey,
    ) -> PolicyDecisionResult:
        """Return one policy decision, batching concurrent requests."""
        loop = asyncio.get_running_loop()
        request_id = self._next_request_id
        self._next_request_id += 1
        future: asyncio.Future[PolicyDecisionResult] = (
            loop.create_future()
        )
        route = PolicyRequestRoute(
            worker_index=self.worker_index,
            request_id=request_id,
        )
        request = PolicyRequestInput(
            route=route,
            observation=observation,
            legal_actions=legal_actions,
            decision_key=decision_key,
        )
        self._send_queue.append(
            _QueuedPolicyRequest(
                request_id=request_id,
                request=request,
                legal_actions=legal_actions,
                future=future,
            )
        )
        self._ensure_send_task()
        wait_start = time.perf_counter()
        try:
            result = await asyncio.shield(future)
            self._record_wait(wait_start=wait_start)
            return result
        except asyncio.CancelledError:
            await self._cancel_request(request_id=request_id)
            future.cancel()
            raise
        finally:
            if future.cancelled():
                await self._cancel_request(request_id=request_id)

    def drain_stats(self) -> PolicyClientStats:
        """Return and reset accumulated policy wait counters."""
        stats = PolicyClientStats(
            decision_count=self._decision_count,
            wait_seconds=self._wait_seconds,
        )
        self._decision_count = 0
        self._wait_seconds = 0.0
        return stats

    def _record_wait(self, *, wait_start: float) -> None:
        self._decision_count += 1
        self._wait_seconds += max(time.perf_counter() - wait_start, 0.0)

    def _ensure_send_task(self) -> None:
        task = self._send_task
        if task is None or task.done():
            self._send_task = asyncio.create_task(self._send_loop())

    def _ensure_receive_task(self) -> None:
        task = self._receive_task
        if task is None or task.done():
            self._receive_task = asyncio.create_task(
                self._receive_loop()
            )

    def _event(self) -> asyncio.Event:
        event = self._in_flight_event
        if event is None:
            event = asyncio.Event()
            self._in_flight_event = event
        return event

    async def _send_loop(self) -> None:
        try:
            async with self._send_lock:
                while self._send_queue:
                    await asyncio.sleep(0)
                    queued = self._pop_send_batch()
                    active = tuple(
                        request
                        for request in queued
                        if not request.future.cancelled()
                    )
                    if not active:
                        continue
                    batch_result = self._build_batch(active)
                    if isinstance(batch_result, Rejected):
                        self._reject_queued_requests(
                            active, batch_result.reason
                        )
                        return
                    sent_requests = batch_result.value.requests
                    submit_result = await asyncio.to_thread(
                        self.transport.submit_batch,
                        batch=batch_result.value.batch,
                    )
                    if isinstance(submit_result, Rejected):
                        self._reject_queued_requests(
                            sent_requests, submit_result.reason
                        )
                        self._reject_all(submit_result.reason)
                        return
                    await self._register_sent_requests(sent_requests)
                    self._ensure_receive_task()
        finally:
            if self._send_task is asyncio.current_task():
                self._send_task = None
            if self._send_queue:
                self._ensure_send_task()

    def _pop_send_batch(self) -> tuple[_QueuedPolicyRequest, ...]:
        count = min(self.batch_size, len(self._send_queue))
        assert count > 0
        queued = tuple(self._send_queue[:count])
        del self._send_queue[:count]
        return queued

    async def _register_sent_requests(
        self, requests: tuple[_QueuedPolicyRequest, ...]
    ) -> None:
        async with self._state_lock:
            for request in requests:
                if request.future.cancelled():
                    self._ignored_response_request_ids.add(
                        request.request_id
                    )
                    continue
                self._in_flight[request.request_id] = (
                    _PendingPolicyRequest(
                        legal_actions=request.legal_actions,
                        future=request.future,
                    )
                )
            if self._in_flight:
                self._event().set()
            if self._deferred_responses:
                deferred = tuple(self._deferred_responses)
                self._deferred_responses.clear()
                dispatch_result = self._dispatch_validated_responses(
                    deferred
                )
                if isinstance(dispatch_result, Rejected):
                    self._reject_all(dispatch_result.reason)

    def _build_batch(
        self, active: tuple[_QueuedPolicyRequest, ...]
    ) -> Ok[_BuiltPolicyBatch] | Rejected:
        assert active
        self._preparer.reset()
        accepted: list[_QueuedPolicyRequest] = []
        for request in active:
            append_result = self._preparer.push_request(request.request)
            if isinstance(append_result, Rejected):
                return append_result
            accepted.append(request)
        assert accepted
        sent_requests = tuple(accepted)
        return Ok(
            value=_BuiltPolicyBatch(
                batch=self._preparer.finish_batch(),
                requests=sent_requests,
            )
        )

    async def _receive_loop(self) -> None:
        try:
            while True:
                if not self._in_flight:
                    return
                response_result = await asyncio.to_thread(
                    self.transport.receive,
                    timeout_seconds=self.timeout_seconds,
                )
                if isinstance(response_result, Rejected):
                    self._reject_all(response_result.reason)
                    return
                dispatch_result = await self._dispatch_responses(
                    response_result.value,
                )
                if isinstance(dispatch_result, Rejected):
                    self._reject_all(dispatch_result.reason)
                    return
        finally:
            if self._receive_task is asyncio.current_task():
                self._receive_task = None

    async def _dispatch_responses(
        self,
        responses: tuple[PolicyResponse, ...],
    ) -> Ok[None] | Rejected:
        async with self._state_lock:
            validation_result = _validate_response_routes(
                responses=responses,
                worker_index=self.worker_index,
            )
            if isinstance(validation_result, Rejected):
                return validation_result
            return self._dispatch_validated_responses(responses)

    def _dispatch_validated_responses(
        self, responses: tuple[PolicyResponse, ...]
    ) -> Ok[None] | Rejected:
        for response in responses:
            route = response.route
            if route.request_id in self._ignored_response_request_ids:
                self._ignored_response_request_ids.remove(
                    route.request_id
                )
                continue
            pending = self._in_flight.pop(route.request_id, None)
            if pending is None:
                return Rejected(
                    reason=(
                        "model rank inference response route mismatch"
                    )
                )
            if pending.future.done():
                continue
            if isinstance(response, RejectedPolicyResponse):
                pending.future.set_result(
                    Rejected(reason=response.reason)
                )
                continue
            assert isinstance(response, CompletedPolicyResponse)
            decoded = decode_policy_response(
                legal_actions=pending.legal_actions,
                response=response,
            )
            pending.future.set_result(decoded)
        return Ok(value=None)

    async def _cancel_request(self, *, request_id: int) -> None:
        async with self._state_lock:
            pending = self._in_flight.pop(request_id, None)
            if pending is not None:
                self._ignored_response_request_ids.add(request_id)

    def _reject_all(self, reason: str) -> None:
        queued = tuple(self._send_queue)
        self._send_queue.clear()
        self._reject_queued_requests(queued, reason)
        rejection = Rejected(reason=reason)
        for request_id, pending in tuple(self._in_flight.items()):
            if not pending.future.done():
                pending.future.set_result(rejection)
            self._in_flight.pop(request_id, None)
        self._event().set()

    def _validate_response_routes(
        self, responses: tuple[PolicyResponse, ...]
    ) -> Ok[None] | Rejected:
        return _validate_response_routes(
            responses=responses,
            worker_index=self.worker_index,
        )

    def _reject_queued_requests(
        self, requests: tuple[_QueuedPolicyRequest, ...], reason: str
    ) -> None:
        rejection = Rejected(reason=reason)
        for request in requests:
            if not request.future.done():
                request.future.set_result(rejection)


def _validate_response_routes(
    *,
    responses: tuple[PolicyResponse, ...],
    worker_index: int,
) -> Ok[None] | Rejected:
    seen_ids: set[int] = set()
    for response in responses:
        route = response.route
        if route.worker_index != worker_index:
            return Rejected(
                reason="model rank inference response worker mismatch"
            )
        if route.request_id in seen_ids:
            return Rejected(
                reason="model rank inference response route mismatch"
            )
        seen_ids.add(route.request_id)
    return Ok(value=None)


def _policy_response_from_decision(
    *,
    route: PolicyRequestRoute,
    decision: ModelRankDecisionResult,
) -> PolicyResponse:
    if isinstance(decision, Rejected):
        return RejectedPolicyResponse(
            route=route, reason=decision.reason
        )
    handle = decision.value.decision_handle
    return CompletedPolicyResponse(
        route=route,
        trace_token_ids=decision.value.trace_token_ids,
        decision_handle_model_rank=handle.model_rank_index,
        decision_handle_policy_version=handle.policy_version,
        decision_handle_row_index=handle.row_index,
        choice_count=decision.value.choice_count,
    )
