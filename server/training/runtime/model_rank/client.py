"""Unified batched policy client for local and process model ranks."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Protocol

import torch

from server.foundation.result import Ok, Rejected
from server.training.legal_actions import LegalActionIndex
from server.training.observation import Observation
from server.training.policy import PolicyDecision
from server.training.policy_inference_batch import (
    BorrowedPolicyRequestBatch,
    CompletedPolicyResponse,
    DevicePolicyRequestBatch,
    PolicyRequestCompiler,
    PolicyRequestInput,
    PolicyRequestRoute,
    PolicyResponse,
    RejectedPolicyResponse,
    build_completed_policy_responses,
    build_rejected_policy_responses,
    decode_policy_response,
    decode_policy_response_batch_wire,
    materialize_borrowed_policy_request_batch,
)
from server.training.policy_sampling import CompactPolicyDecisionBatch
from server.training.runtime.model_rank.inference_transport import (
    AsyncPolicyPeer,
)
from server.training.sampling import PolicyDecisionKey
from server.training_events import EventContext, EventSink

type PolicyDecisionResult = Ok[PolicyDecision] | Rejected
type ModelRankDecisionResult = Ok[CompactPolicyDecisionBatch] | Rejected


class PolicyBatchTransport(Protocol):
    """Transport one request frame through a model-rank boundary."""

    async def submit_batch(
        self, *, batch: BorrowedPolicyRequestBatch
    ) -> Ok[None] | Rejected: ...

    async def receive(
        self, *, timeout_seconds: float
    ) -> Ok[tuple[PolicyResponse, ...]] | Rejected: ...


class ModelReplicaProtocol(Protocol):
    """Model replica operations needed by same-process inference."""

    @property
    def device(self) -> torch.device: ...

    def decide_batch(
        self, requests: DevicePolicyRequestBatch
    ) -> ModelRankDecisionResult: ...


@dataclass(slots=True)
class AsyncRemotePolicyBatchTransport:
    """Async socket-backed remote inference transport."""

    peer: AsyncPolicyPeer

    async def submit_batch(
        self, *, batch: BorrowedPolicyRequestBatch
    ) -> Ok[None] | Rejected:
        """Send one compiled request batch to the model rank."""
        return await self.peer.send_request(batch.frame)

    async def receive(
        self, *, timeout_seconds: float
    ) -> Ok[tuple[PolicyResponse, ...]] | Rejected:
        """Receive and decode one response batch from the model rank."""
        response_result = await self.peer.receive_response(
            timeout_seconds=timeout_seconds
        )
        if isinstance(response_result, Rejected):
            return response_result
        return decode_policy_response_batch_wire(
            response_result.value.data
        )


@dataclass(slots=True)
class LocalPolicyBatchTransport:
    """Same-process inference transport for CPU worker model ranks."""

    replica: ModelReplicaProtocol
    _pending_response: tuple[PolicyResponse, ...] | None = None

    async def submit_batch(
        self, *, batch: BorrowedPolicyRequestBatch
    ) -> Ok[None] | Rejected:
        """Submit one prepared batch to the local model rank."""
        request_result = materialize_borrowed_policy_request_batch(
            batch=batch, device=self.replica.device
        )
        if isinstance(request_result, Rejected):
            return request_result
        decision_result = self.replica.decide_batch(
            request_result.value
        )
        if isinstance(decision_result, Rejected):
            responses_result = build_rejected_policy_responses(
                routes=batch.routes,
                reason=decision_result.reason,
            )
            if isinstance(responses_result, Rejected):
                return responses_result
            responses = responses_result.value
        else:
            if decision_result.value.row_count() != batch.row_count():
                return Rejected(
                    reason="local policy response batch mismatch"
                )
            responses_result = build_completed_policy_responses(
                routes=batch.routes,
                decisions=decision_result.value,
            )
            if isinstance(responses_result, Rejected):
                return responses_result
            responses = responses_result.value
        if self._pending_response is not None:
            return Rejected(
                reason="local policy response is still pending"
            )
        self._pending_response = responses
        return Ok(value=None)

    async def receive(
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
    inference_batch: _PendingInferenceBatch


@dataclass(slots=True)
class _PendingInferenceBatch:
    context: EventContext
    started_at: float
    batch_size: int
    configured_batch_size: int
    remaining: int
    error: str | None = None


@dataclass(slots=True)
class _QueuedPolicyRequest:
    request_id: int
    request: PolicyRequestInput
    legal_actions: LegalActionIndex
    future: asyncio.Future[PolicyDecisionResult]


@dataclass(frozen=True, slots=True)
class _BuiltPolicyBatch:
    batch: BorrowedPolicyRequestBatch
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
    transport: PolicyBatchTransport
    timeout_seconds: float
    batch_size: int
    event_sink: EventSink
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
    _compiler: PolicyRequestCompiler = field(init=False)
    _send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _send_task: asyncio.Task[None] | None = None
    _receive_task: asyncio.Task[None] | None = None
    _in_flight_event: asyncio.Event | None = None
    _decision_count: int = 0
    _wait_seconds: float = 0.0

    def __post_init__(self) -> None:
        assert self.worker_index >= 0
        assert self.timeout_seconds > 0.0
        assert self.batch_size > 0
        self._compiler = PolicyRequestCompiler(
            batch_capacity=self.batch_size
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
        context = EventContext(
            policy_version=decision_key.policy_version,
            rollout_id=decision_key.rollout_id,
            worker_index=self.worker_index,
            episode_id=decision_key.episode_id,
            player_index=decision_key.player_index,
            decision_index=decision_key.decision_index,
            request_id=request_id,
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
            wait_seconds = self._record_wait(wait_start=wait_start)
            if isinstance(result, Rejected):
                self.event_sink.emit(
                    "decision",
                    context=context,
                    fields={
                        "wait_seconds": wait_seconds,
                    },
                    error=result.reason,
                )
            else:
                self.event_sink.emit(
                    "decision",
                    context=context,
                    fields={
                        "wait_seconds": wait_seconds,
                        "choice_count": result.value.choice_count,
                        "model_rank_index": (
                            result.value.decision_handle.model_rank_index
                        ),
                        "row_index": (
                            result.value.decision_handle.row_index
                        ),
                    },
                )
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
        stats = self.stats()
        self._decision_count = 0
        self._wait_seconds = 0.0
        return stats

    def stats(self) -> PolicyClientStats:
        """Return accumulated policy wait counters without resetting."""
        return PolicyClientStats(
            decision_count=self._decision_count,
            wait_seconds=self._wait_seconds,
        )

    def _record_wait(self, *, wait_start: float) -> float:
        wait_seconds = max(time.perf_counter() - wait_start, 0.0)
        self._decision_count += 1
        self._wait_seconds += wait_seconds
        return wait_seconds

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
                    if not self._send_queue:
                        break
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
                    rollout_ids = {
                        item.request.decision_key.rollout_id
                        for item in sent_requests
                    }
                    policy_versions = {
                        item.request.decision_key.policy_version
                        for item in sent_requests
                    }
                    assert len(rollout_ids) == 1
                    assert len(policy_versions) == 1
                    batch_context = EventContext(
                        policy_version=(
                            sent_requests[
                                0
                            ].request.decision_key.policy_version
                        ),
                        rollout_id=(
                            sent_requests[
                                0
                            ].request.decision_key.rollout_id
                        ),
                        worker_index=self.worker_index,
                        batch_id=sent_requests[0].request_id,
                    )
                    inference_started = time.perf_counter()
                    submit_result = await self.transport.submit_batch(
                        batch=batch_result.value.batch,
                    )
                    if isinstance(submit_result, Rejected):
                        self.event_sink.emit(
                            "inference.batch",
                            context=batch_context,
                            fields={
                                "batch_size": len(sent_requests),
                                "duration_seconds": max(
                                    time.perf_counter()
                                    - inference_started,
                                    0.0,
                                ),
                            },
                            error=submit_result.reason,
                        )
                        self._reject_queued_requests(
                            sent_requests, submit_result.reason
                        )
                        self._reject_all(submit_result.reason)
                        return
                    await self._register_sent_requests(
                        sent_requests,
                        inference_batch=_PendingInferenceBatch(
                            context=batch_context,
                            started_at=inference_started,
                            batch_size=len(sent_requests),
                            configured_batch_size=self.batch_size,
                            remaining=len(sent_requests),
                        ),
                    )
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
        self,
        requests: tuple[_QueuedPolicyRequest, ...],
        *,
        inference_batch: _PendingInferenceBatch,
    ) -> None:
        async with self._state_lock:
            for request in requests:
                pending = _PendingPolicyRequest(
                    legal_actions=request.legal_actions,
                    future=request.future,
                    inference_batch=inference_batch,
                )
                self._in_flight[request.request_id] = pending
                if request.future.cancelled():
                    self._ignored_response_request_ids.add(
                        request.request_id
                    )
                    continue
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
        sent_requests = active
        batch_result = self._compiler.compile_batch(
            tuple(request.request for request in sent_requests)
        )
        if isinstance(batch_result, Rejected):
            return batch_result
        return Ok(
            value=_BuiltPolicyBatch(
                batch=batch_result.value,
                requests=sent_requests,
            )
        )

    async def _receive_loop(self) -> None:
        try:
            while True:
                if (
                    not self._in_flight
                    and not self._ignored_response_request_ids
                ):
                    return
                response_result = await self.transport.receive(
                    timeout_seconds=self.timeout_seconds,
                )
                if isinstance(response_result, Rejected):
                    self._reject_all(response_result.reason)
                    return
                dispatch_result = await self._dispatch_responses(
                    response_result.value,
                )
                if isinstance(dispatch_result, Rejected):
                    self._reject_in_flight(dispatch_result.reason)
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
                pending = self._in_flight.pop(route.request_id, None)
                if pending is not None:
                    self._finish_inference_request(pending)
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
                pending.inference_batch.error = response.reason
                pending.future.set_result(
                    Rejected(reason=response.reason)
                )
                self._finish_inference_request(pending)
                continue
            assert isinstance(response, CompletedPolicyResponse)
            decoded = decode_policy_response(
                legal_actions=pending.legal_actions,
                response=response,
            )
            if isinstance(decoded, Rejected):
                pending.inference_batch.error = decoded.reason
            pending.future.set_result(decoded)
            self._finish_inference_request(pending)
        return Ok(value=None)

    def _finish_inference_request(
        self, pending: _PendingPolicyRequest
    ) -> None:
        batch = pending.inference_batch
        assert batch.remaining > 0
        batch.remaining -= 1
        if batch.remaining != 0:
            return
        elapsed = max(time.perf_counter() - batch.started_at, 0.0)
        self.event_sink.emit(
            "inference.batch",
            context=batch.context,
            fields={
                "batch_size": batch.batch_size,
                "fill_ratio": (
                    batch.batch_size
                    / float(batch.configured_batch_size)
                ),
                "inference_seconds": elapsed,
                "duration_seconds": elapsed,
            },
            error=batch.error,
        )

    async def _cancel_request(self, *, request_id: int) -> None:
        async with self._state_lock:
            if request_id in self._in_flight:
                self._ignored_response_request_ids.add(request_id)

    def _reject_all(self, reason: str) -> None:
        queued = tuple(self._send_queue)
        self._send_queue.clear()
        self._reject_queued_requests(queued, reason)
        self._reject_in_flight(reason)

    def _reject_in_flight(self, reason: str) -> None:
        rejection = Rejected(reason=reason)
        for request_id, pending in tuple(self._in_flight.items()):
            if not pending.future.done():
                pending.future.set_result(rejection)
            pending.inference_batch.error = reason
            self._finish_inference_request(pending)
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
