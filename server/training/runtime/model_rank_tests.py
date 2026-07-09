"""Tests for model-rank package public boundaries."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue
import threading
from multiprocessing.connection import Connection

import pytest
import torch

from server.player.test_helpers import card, make_snapshot
from server.result import Ok, Rejected
from server.training.legal_actions import (
    LegalActionIndex,
    build_legal_action_index,
)
from server.training.observation import (
    Observation,
    build_observation,
)
from server.training.policy_inference_batch import (
    CompletedPolicyResponse,
    DevicePolicyRequestBatch,
    PolicyRequestBatch,
    PolicyRequestRoute,
    PolicyResponse,
    PolicyResponseBatchWire,
    build_policy_response_batch_wire,
)
from server.training.policy_inference_batch.frame import (
    decode_policy_request_frame_metadata,
)
from server.training.policy_inference_batch.schema import (
    max_policy_request_batch_frame_bytes,
)
from server.training.policy_inference_batch.types import (
    PolicyRequestFrameMetadata,
    PolicyRequestWireFrame,
)
from server.training.policy_sampling import (
    CompactTraceTokenIds,
    DecisionHandle,
    ModelRankPolicyDecision,
    RankReturnTargets,
)
from server.training.ppo import PPOUpdateProfile, PPOUpdateStats
from server.training.runtime.model_rank import (
    BatchedPolicyClient,
    ConnectionPolicyBatchTransport,
    LocalModelRank,
    LocalPolicyBatchTransport,
)
from server.training.runtime.model_rank.inference_transport import (
    ConnectionPolicyRequestReceiver,
    ConnectionPolicyRequestSender,
    ConnectionPolicyResponseReceiver,
    ConnectionPolicyResponseSender,
    send_policy_response_batch,
)
from server.training.runtime.state import RuntimeTrainingState
from server.training.sampling import PolicyDecisionKey
from server.training.semantic_actions import (
    SemanticArgument,
    SemanticArgumentTrace,
)
from server.training.semantic_actions.codec import (
    SEMANTIC_CODEC,
    semantic_argument_id,
)


@pytest.mark.asyncio
async def test_local_policy_client_batches_same_loop() -> None:
    observation = _observation()
    legal_actions = _legal_actions(observation)
    rank_decision = _model_rank_policy_decision()
    replica = _FakeReplica(
        state=_runtime_state(), decision=rank_decision
    )

    client = BatchedPolicyClient(
        worker_index=2,
        max_observation_tokens=512,
        transport=LocalPolicyBatchTransport(replica=replica),
        timeout_seconds=1.0,
        batch_size=4,
    )
    first_task = asyncio.create_task(
        client.decide(observation, legal_actions, _decision_key())
    )
    second_task = asyncio.create_task(
        client.decide(observation, legal_actions, _decision_key())
    )

    first_result, second_result = await asyncio.gather(
        first_task, second_task
    )

    assert isinstance(first_result, Ok)
    assert isinstance(second_result, Ok)
    assert first_result.value.action.semantic_trace == _pass_trace()
    assert second_result.value.action.semantic_trace == _pass_trace()
    assert first_result.value.decision_handle == (
        rank_decision.decision_handle
    )
    assert second_result.value.decision_handle == (
        rank_decision.decision_handle
    )
    assert first_result.value.choice_count == rank_decision.choice_count
    assert (
        second_result.value.choice_count == rank_decision.choice_count
    )
    assert replica.calls == ("decide_batch",)
    assert replica.batch_sizes == (2,)


def test_local_model_rank_loads_updates_and_snapshots_replica() -> None:
    state = _runtime_state()
    replica = _FakeReplica(
        state=state,
        decision=_model_rank_policy_decision(),
    )
    rank = LocalModelRank(replica=replica)

    load_result = rank.load_state(state=state, policy_version=3)
    update_result = rank.update(
        returns=_return_batch(),
        policy_version=3,
    )
    snapshot_result = rank.snapshot()

    assert isinstance(load_result, Ok)
    assert isinstance(update_result, Ok)
    assert isinstance(snapshot_result, Ok)
    assert replica.calls == (
        "load_state",
        "update_returns",
        "snapshot",
    )
    assert update_result.value == _ppo_update_stats()
    assert snapshot_result.value is state


@pytest.mark.asyncio
async def test_remote_policy_client_roundtrips_connection_payload() -> (
    None
):
    context = mp.get_context("spawn")
    request_receiver, request_sender = context.Pipe(duplex=False)
    response_receiver, response_sender = context.Pipe(duplex=False)
    rank_decision = _model_rank_policy_decision()
    try:
        observation = _observation()
        legal_actions = _legal_actions(observation)

        task = asyncio.create_task(
            BatchedPolicyClient(
                worker_index=2,
                max_observation_tokens=512,
                transport=ConnectionPolicyBatchTransport(
                    request_sender=ConnectionPolicyRequestSender(
                        connection=request_sender
                    ),
                    response_receiver=ConnectionPolicyResponseReceiver(
                        response_receiver
                    ),
                ),
                timeout_seconds=1.0,
                batch_size=4,
            ).decide(
                observation,
                legal_actions,
                _decision_key(),
            )
        )
        metadata = await _receive_request_metadata(
            request_receiver=request_receiver
        )
        route = metadata.routes[0]
        assert route.worker_index == 2
        assert route.request_id == 0
        response_wire = build_policy_response_batch_wire(
            routes=(route,),
            decisions=(Ok(value=rank_decision),),
        )
        assert isinstance(response_wire, Ok)
        send_result = send_policy_response_batch(
            sender=ConnectionPolicyResponseSender(
                worker_index=2, connection=response_sender
            ),
            response=response_wire.value,
        )
        assert isinstance(send_result, Ok)

        result = await task

        assert isinstance(result, Ok)
        assert result.value.action.semantic_trace == _pass_trace()
        assert (
            result.value.decision_handle
            == rank_decision.decision_handle
        )
        assert result.value.choice_count == rank_decision.choice_count
    finally:
        request_receiver.close()
        request_sender.close()
        response_receiver.close()
        response_sender.close()


@pytest.mark.asyncio
async def test_remote_policy_client_rejects_response_mismatch() -> None:
    context = mp.get_context("spawn")
    request_receiver, request_sender = context.Pipe(duplex=False)
    response_receiver, response_sender = context.Pipe(duplex=False)
    try:
        observation = _observation()
        legal_actions = _legal_actions(observation)
        task = asyncio.create_task(
            BatchedPolicyClient(
                worker_index=2,
                max_observation_tokens=512,
                transport=ConnectionPolicyBatchTransport(
                    request_sender=ConnectionPolicyRequestSender(
                        connection=request_sender
                    ),
                    response_receiver=ConnectionPolicyResponseReceiver(
                        response_receiver
                    ),
                ),
                timeout_seconds=1.0,
                batch_size=4,
            ).decide(
                observation,
                legal_actions,
                _decision_key(),
            )
        )
        metadata = await _receive_request_metadata(
            request_receiver=request_receiver
        )
        route = metadata.routes[0]
        assert route.request_id == 0
        mismatch_route = PolicyRequestRoute(
            worker_index=3,
            request_id=route.request_id,
        )
        response_wire = build_policy_response_batch_wire(
            routes=(mismatch_route,),
            decisions=(Ok(value=_model_rank_policy_decision()),),
        )
        assert isinstance(response_wire, Ok)
        send_result = send_policy_response_batch(
            sender=ConnectionPolicyResponseSender(
                worker_index=2, connection=response_sender
            ),
            response=response_wire.value,
        )
        assert isinstance(send_result, Ok)
        result = await task
        assert isinstance(result, Rejected)
        assert (
            result.reason
            == "model rank inference response worker mismatch"
        )
    finally:
        request_receiver.close()
        request_sender.close()
        response_receiver.close()
        response_sender.close()


@pytest.mark.asyncio
async def test_remote_policy_client_serializes_request_writes() -> None:
    observation = _observation()
    legal_actions = _legal_actions(observation)
    response_receiver = _QueuedPolicyResponseReceiver()
    request_sender = _BlockingPolicyRequestSender(
        response_receiver=response_receiver
    )
    client = BatchedPolicyClient(
        worker_index=2,
        max_observation_tokens=512,
        transport=ConnectionPolicyBatchTransport(
            request_sender=request_sender,
            response_receiver=response_receiver,
        ),
        timeout_seconds=1.0,
        batch_size=4,
    )

    first = asyncio.create_task(
        client.decide(observation, legal_actions, _decision_key())
    )
    first_started = await asyncio.to_thread(
        request_sender.wait_first_started
    )
    assert first_started
    second = asyncio.create_task(
        client.decide(observation, legal_actions, _decision_key())
    )
    await asyncio.sleep(0.05)
    started_before_release = request_sender.started_request_ids

    request_sender.release_first()
    first_result, second_result = await asyncio.gather(first, second)

    assert started_before_release == (0,)
    assert request_sender.started_request_ids == (0, 1)
    assert not request_sender.overlap_detected
    assert isinstance(first_result, Ok)
    assert isinstance(second_result, Ok)
    assert first_result.value.action.semantic_trace == _pass_trace()
    assert second_result.value.action.semantic_trace == _pass_trace()


@pytest.mark.asyncio
async def test_remote_policy_client_rejects_unsent_route_response() -> (
    None
):
    observation = _observation()
    legal_actions = _legal_actions(observation)
    transport = _MismatchedFirstResponseTransport()
    client = BatchedPolicyClient(
        worker_index=2,
        max_observation_tokens=512,
        transport=transport,
        timeout_seconds=1.0,
        batch_size=1,
    )

    first = asyncio.create_task(
        client.decide(observation, legal_actions, _decision_key())
    )
    first_sent = await asyncio.to_thread(transport.wait_first_sent)
    assert first_sent
    second = asyncio.create_task(
        client.decide(observation, legal_actions, _decision_key())
    )
    await asyncio.sleep(0)
    sent_before_release = transport.sent_request_ids

    transport.release_first_response()
    first_result, second_result = await asyncio.gather(first, second)

    assert sent_before_release == (0,)
    assert transport.sent_request_ids == (0, 1)
    assert isinstance(first_result, Rejected)
    assert (
        first_result.reason
        == "model rank inference response route mismatch"
    )
    assert isinstance(second_result, Ok)
    assert second_result.value.action.semantic_trace == _pass_trace()


async def _receive_request_metadata(
    *,
    request_receiver: Connection,
) -> PolicyRequestFrameMetadata:
    request_buffer = bytearray(
        max_policy_request_batch_frame_bytes(
            batch_capacity=4,
            max_observation_tokens=512,
            padded_generation_steps=SEMANTIC_CODEC.max_argument_tokens,
        )
    )
    request_result = await asyncio.to_thread(
        ConnectionPolicyRequestReceiver(
            connection=request_receiver
        ).receive_batch_bytes_into,
        memoryview(request_buffer),
    )
    assert isinstance(request_result, Ok)
    metadata_result = decode_policy_request_frame_metadata(
        memoryview(request_buffer)[: request_result.value]
    )
    assert isinstance(metadata_result, Ok)
    assert len(metadata_result.value.routes) == 1
    return metadata_result.value


class _QueuedPolicyResponseReceiver:
    def __init__(self) -> None:
        self._responses: queue.Queue[PolicyResponseBatchWire] = (
            queue.Queue()
        )

    def push(
        self,
        *,
        route: PolicyRequestRoute,
        decision: ModelRankPolicyDecision,
    ) -> None:
        response = build_policy_response_batch_wire(
            routes=(route,),
            decisions=(Ok(value=decision),),
        )
        assert isinstance(response, Ok)
        self._responses.put(response.value)

    def receive(
        self, *, timeout_seconds: float
    ) -> Ok[PolicyResponseBatchWire] | Rejected:
        try:
            return Ok(
                value=self._responses.get(timeout=timeout_seconds)
            )
        except queue.Empty:
            return Rejected(
                reason="model rank policy inference timed out"
            )


class _BlockingPolicyRequestSender:
    def __init__(
        self, *, response_receiver: _QueuedPolicyResponseReceiver
    ) -> None:
        self._response_receiver = response_receiver
        self._lock = threading.Lock()
        self._first_started = threading.Event()
        self._release_first = threading.Event()
        self._active_count = 0
        self._started_request_ids: list[int] = []
        self._overlap_detected = False

    @property
    def started_request_ids(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._started_request_ids)

    @property
    def overlap_detected(self) -> bool:
        with self._lock:
            return self._overlap_detected

    def wait_first_started(self) -> bool:
        return self._first_started.wait(timeout=5.0)

    def release_first(self) -> None:
        self._release_first.set()

    def send(
        self, request: PolicyRequestWireFrame
    ) -> Ok[None] | Rejected:
        metadata_result = decode_policy_request_frame_metadata(
            request.view()
        )
        assert isinstance(metadata_result, Ok)
        assert len(metadata_result.value.routes) == 1
        route = metadata_result.value.routes[0]
        with self._lock:
            if self._active_count > 0:
                self._overlap_detected = True
            self._active_count += 1
            self._started_request_ids.append(route.request_id)
            is_first_request = len(self._started_request_ids) == 1
            if is_first_request:
                self._first_started.set()
        try:
            if is_first_request and not self._release_first.wait(
                timeout=5.0
            ):
                return Rejected(reason="test request sender timed out")
            self._response_receiver.push(
                route=route,
                decision=_model_rank_policy_decision(),
            )
            return Ok(value=None)
        finally:
            with self._lock:
                self._active_count -= 1


class _MismatchedFirstResponseTransport:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._first_sent = threading.Event()
        self._release_first_response = threading.Event()
        self._sent_request_ids: list[int] = []
        self._receive_count = 0

    @property
    def sent_request_ids(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._sent_request_ids)

    def wait_first_sent(self) -> bool:
        return self._first_sent.wait(timeout=5.0)

    def release_first_response(self) -> None:
        self._release_first_response.set()

    def submit_batch(self, *, batch: PolicyRequestBatch) -> Ok[None]:
        assert batch.row_count() == 1
        route = batch.routes[0]
        with self._lock:
            self._sent_request_ids.append(route.request_id)
            if len(self._sent_request_ids) == 1:
                self._first_sent.set()
        return Ok(value=None)

    def receive(
        self, *, timeout_seconds: float
    ) -> Ok[tuple[PolicyResponse, ...]] | Rejected:
        assert timeout_seconds > 0.0
        with self._lock:
            self._receive_count += 1
            receive_index = self._receive_count
            sent_request_id = self._sent_request_ids[-1]
        if receive_index == 1:
            if not self._release_first_response.wait(timeout=5.0):
                return Rejected(reason="test response timed out")
            return Ok(
                value=(
                    _completed_policy_response(
                        PolicyRequestRoute(
                            worker_index=2,
                            request_id=1,
                        )
                    ),
                )
            )
        return Ok(
            value=(
                _completed_policy_response(
                    PolicyRequestRoute(
                        worker_index=2,
                        request_id=sent_request_id,
                    )
                ),
            )
        )


class _FakeReplica:
    def __init__(
        self,
        *,
        state: RuntimeTrainingState,
        decision: ModelRankPolicyDecision,
    ) -> None:
        self._state = state
        self._decision = decision
        self._calls: list[str] = []
        self._batch_sizes: list[int] = []

    @property
    def calls(self) -> tuple[str, ...]:
        return tuple(self._calls)

    @property
    def batch_sizes(self) -> tuple[int, ...]:
        return tuple(self._batch_sizes)

    def load_state(self, *, snapshot: RuntimeTrainingState) -> None:
        assert snapshot is self._state
        self._calls.append("load_state")

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def decide_batch(
        self, requests: DevicePolicyRequestBatch
    ) -> tuple[Ok[ModelRankPolicyDecision], ...]:
        self._calls.append("decide_batch")
        self._batch_sizes.append(len(requests.policy_versions))
        return tuple(
            Ok(value=self._decision) for _ in requests.policy_versions
        )

    def update_returns(
        self, *, returns: RankReturnTargets
    ) -> Ok[PPOUpdateStats]:
        assert not returns.is_empty()
        self._calls.append("update_returns")
        return Ok(value=_ppo_update_stats())

    def snapshot(self) -> RuntimeTrainingState:
        self._calls.append("snapshot")
        return self._state


def _model_rank_policy_decision() -> ModelRankPolicyDecision:
    return ModelRankPolicyDecision(
        trace_token_ids=CompactTraceTokenIds.from_tuple(
            (semantic_argument_id(SemanticArgument("pass")),)
        ),
        decision_handle=DecisionHandle(
            model_rank_index=0,
            policy_version=0,
            row_index=0,
        ),
        choice_count=1,
    )


def _completed_policy_response(
    route: PolicyRequestRoute,
) -> CompletedPolicyResponse:
    decision = _model_rank_policy_decision()
    handle = decision.decision_handle
    return CompletedPolicyResponse(
        route=route,
        trace_token_ids=decision.trace_token_ids,
        decision_handle_model_rank=handle.model_rank_index,
        decision_handle_policy_version=handle.policy_version,
        decision_handle_row_index=handle.row_index,
        choice_count=decision.choice_count,
    )


def _pass_trace() -> SemanticArgumentTrace:
    return SemanticArgumentTrace(arguments=(SemanticArgument("pass"),))


def _observation() -> Observation:
    snapshot = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand=[card("hearts", "2", 1)],
        trump_rank="2",
    )
    return build_observation(
        player_index=0,
        snapshot=snapshot,
        history=(),
    )


def _legal_actions(
    observation: Observation,
) -> LegalActionIndex:
    snapshot = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand=[card("hearts", "2", 1)],
        trump_rank="2",
    )
    return build_legal_action_index(
        player_index=0,
        snapshot=snapshot,
        query=observation.action_query,
    )


def _decision_key() -> PolicyDecisionKey:
    return PolicyDecisionKey(
        base_seed=0,
        policy_version=0,
        episode_id=0,
        player_index=0,
        decision_index=0,
    )


def _return_batch() -> RankReturnTargets:
    return RankReturnTargets(
        policy_version=3,
        model_rank_index=0,
        row_indices=torch.tensor((0,), dtype=torch.long),
        step_counts=torch.tensor((1,), dtype=torch.long),
        return_values=torch.tensor((1.0,), dtype=torch.float32),
        round_count=1,
        total_step_count=1,
        max_step_count=1,
    )


def _runtime_state() -> RuntimeTrainingState:
    return RuntimeTrainingState(
        model_state={"weight": torch.tensor([1.0])},
        optimizer_state={
            "kind": "adamw",
            "step_count": 0,
            "exp_avgs": [],
            "exp_avg_sqs": [],
        },
    )


def _ppo_update_stats() -> PPOUpdateStats:
    return PPOUpdateStats(
        policy_loss=0.0,
        value_loss=0.0,
        entropy=0.0,
        total_loss=0.0,
        approx_kl=0.0,
        clip_fraction=0.0,
        profile=PPOUpdateProfile(
            update_seconds=0.0,
            minibatch_loss_seconds=0.0,
            observation_batch_seconds=0.0,
            observation_encode_seconds=0.0,
            value_head_seconds=0.0,
            argument_select_seconds=0.0,
            argument_decode_seconds=0.0,
            argument_distribution_seconds=0.0,
            backward_seconds=0.0,
            optimizer_step_seconds=0.0,
            argument_decode_fraction=0.0,
            argument_trace_batch_count=0,
            argument_trace_row_count=0,
            argument_trace_token_count=0,
            argument_trace_valid_token_count=0,
            argument_trace_padding_token_count=0,
        ),
    )
