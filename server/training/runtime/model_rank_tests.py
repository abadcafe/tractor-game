"""Tests for model-rank package public boundaries."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

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
    CompiledPolicyRequestBatch,
    CompletedPolicyResponse,
    DevicePolicyRequestBatch,
    PolicyRequestRoute,
    PolicyResponse,
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
)
from server.training.policy_sampling import (
    CompactPolicyDecisionBatch,
    CompactTraceTokenBatch,
    CompactTraceTokenIds,
    RankReturnTargets,
)
from server.training.ppo import PPOUpdateProfile, PPOUpdateStats
from server.training.runtime.async_ipc import create_async_socket_pair
from server.training.runtime.model_rank import (
    AsyncRemotePolicyBatchTransport,
    BatchedPolicyClient,
    LocalModelRank,
    LocalPolicyBatchTransport,
)
from server.training.runtime.model_rank.inference_transport import (
    AsyncPolicyPeer,
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


@dataclass(frozen=True, slots=True)
class _PolicyPeerPair:
    worker_peer: AsyncPolicyPeer
    model_rank_peer: AsyncPolicyPeer

    def close(self) -> None:
        self.worker_peer.close()
        self.model_rank_peer.close()


@pytest.mark.asyncio
async def test_local_policy_client_batches_same_loop() -> None:
    observation = _observation()
    legal_actions = _legal_actions(observation)
    replica = _FakeReplica(state=_runtime_state())

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
    assert first_result.value.decision_handle.row_index == 0
    assert second_result.value.decision_handle.row_index == 1
    assert first_result.value.choice_count == 1
    assert second_result.value.choice_count == 1
    assert replica.calls == ("decide_batch",)
    assert replica.batch_sizes == (2,)


def test_local_model_rank_loads_updates_and_snapshots_replica() -> None:
    state = _runtime_state()
    replica = _FakeReplica(state=state)
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
async def test_remote_policy_client_roundtrips_async_payload() -> None:
    peers = _policy_peer_pair(worker_index=2)
    try:
        observation = _observation()
        legal_actions = _legal_actions(observation)

        task = asyncio.create_task(
            BatchedPolicyClient(
                worker_index=2,
                max_observation_tokens=512,
                transport=AsyncRemotePolicyBatchTransport(
                    peer=peers.worker_peer,
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
            peer=peers.model_rank_peer
        )
        route = metadata.routes[0]
        assert route.worker_index == 2
        assert route.request_id == 0
        response_wire = build_policy_response_batch_wire(
            routes=(route,),
            decisions=_compact_policy_decision_batch(row_count=1),
        )
        assert isinstance(response_wire, Ok)
        send_result = await peers.model_rank_peer.send_response(
            response_wire.value
        )
        assert isinstance(send_result, Ok)

        result = await task

        assert isinstance(result, Ok)
        assert result.value.action.semantic_trace == _pass_trace()
        assert result.value.decision_handle.row_index == 0
        assert result.value.choice_count == 1
    finally:
        peers.close()


@pytest.mark.asyncio
async def test_remote_policy_client_rejects_response_mismatch() -> None:
    peers = _policy_peer_pair(worker_index=2)
    try:
        observation = _observation()
        legal_actions = _legal_actions(observation)
        task = asyncio.create_task(
            BatchedPolicyClient(
                worker_index=2,
                max_observation_tokens=512,
                transport=AsyncRemotePolicyBatchTransport(
                    peer=peers.worker_peer,
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
            peer=peers.model_rank_peer
        )
        route = metadata.routes[0]
        assert route.request_id == 0
        mismatch_route = PolicyRequestRoute(
            worker_index=3,
            request_id=route.request_id,
        )
        response_wire = build_policy_response_batch_wire(
            routes=(mismatch_route,),
            decisions=_compact_policy_decision_batch(row_count=1),
        )
        assert isinstance(response_wire, Ok)
        send_result = await peers.model_rank_peer.send_response(
            response_wire.value
        )
        assert isinstance(send_result, Ok)
        result = await task
        assert isinstance(result, Rejected)
        assert (
            result.reason
            == "model rank inference response worker mismatch"
        )
    finally:
        peers.close()


@pytest.mark.asyncio
async def test_remote_policy_client_sends_concurrent_frames() -> None:
    observation = _observation()
    legal_actions = _legal_actions(observation)
    peers = _policy_peer_pair(worker_index=2)
    try:
        client = BatchedPolicyClient(
            worker_index=2,
            max_observation_tokens=512,
            transport=AsyncRemotePolicyBatchTransport(
                peer=peers.worker_peer
            ),
            timeout_seconds=1.0,
            batch_size=1,
        )
        first = asyncio.create_task(
            client.decide(observation, legal_actions, _decision_key())
        )
        second = asyncio.create_task(
            client.decide(observation, legal_actions, _decision_key())
        )

        first_metadata = await _receive_request_metadata(
            peer=peers.model_rank_peer
        )
        second_metadata = await _receive_request_metadata(
            peer=peers.model_rank_peer
        )
        first_route = first_metadata.routes[0]
        second_route = second_metadata.routes[0]
        await _send_decision_response(
            peer=peers.model_rank_peer,
            route=first_route,
        )
        await _send_decision_response(
            peer=peers.model_rank_peer,
            route=second_route,
        )
        first_result, second_result = await asyncio.gather(
            first, second
        )

        assert (first_route.request_id, second_route.request_id) == (
            0,
            1,
        )
        assert isinstance(first_result, Ok)
        assert isinstance(second_result, Ok)
        assert first_result.value.action.semantic_trace == _pass_trace()
        assert (
            second_result.value.action.semantic_trace == _pass_trace()
        )
    finally:
        peers.close()


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
    first_sent = await transport.wait_first_sent()
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
    peer: AsyncPolicyPeer,
) -> PolicyRequestFrameMetadata:
    request_buffer = bytearray(
        max_policy_request_batch_frame_bytes(
            batch_capacity=4,
            max_observation_tokens=512,
            padded_generation_steps=SEMANTIC_CODEC.max_argument_tokens,
        )
    )
    request_result = await peer.receive_request_into(
        memoryview(request_buffer)
    )
    assert isinstance(request_result, Ok)
    metadata_result = decode_policy_request_frame_metadata(
        memoryview(request_buffer)[: request_result.value]
    )
    assert isinstance(metadata_result, Ok)
    assert len(metadata_result.value.routes) == 1
    return metadata_result.value


async def _send_decision_response(
    *,
    peer: AsyncPolicyPeer,
    route: PolicyRequestRoute,
) -> None:
    response_wire = build_policy_response_batch_wire(
        routes=(route,),
        decisions=_compact_policy_decision_batch(row_count=1),
    )
    assert isinstance(response_wire, Ok)
    send_result = await peer.send_response(response_wire.value)
    assert isinstance(send_result, Ok)


def _policy_peer_pair(*, worker_index: int) -> _PolicyPeerPair:
    pair = create_async_socket_pair()
    return _PolicyPeerPair(
        worker_peer=AsyncPolicyPeer(
            worker_index=worker_index,
            endpoint=pair.first,
        ),
        model_rank_peer=AsyncPolicyPeer(
            worker_index=worker_index,
            endpoint=pair.second,
        ),
    )


class _MismatchedFirstResponseTransport:
    def __init__(self) -> None:
        self._first_sent = asyncio.Event()
        self._release_first_response = asyncio.Event()
        self._sent_request_ids: list[int] = []
        self._receive_count = 0

    @property
    def sent_request_ids(self) -> tuple[int, ...]:
        return tuple(self._sent_request_ids)

    async def wait_first_sent(self) -> bool:
        try:
            await asyncio.wait_for(self._first_sent.wait(), timeout=5.0)
        except TimeoutError:
            return False
        return True

    def release_first_response(self) -> None:
        self._release_first_response.set()

    async def submit_batch(
        self, *, batch: CompiledPolicyRequestBatch
    ) -> Ok[None]:
        assert batch.row_count() == 1
        route = batch.routes[0]
        self._sent_request_ids.append(route.request_id)
        if len(self._sent_request_ids) == 1:
            self._first_sent.set()
        return Ok(value=None)

    async def receive(
        self, *, timeout_seconds: float
    ) -> Ok[tuple[PolicyResponse, ...]] | Rejected:
        assert timeout_seconds > 0.0
        self._receive_count += 1
        receive_index = self._receive_count
        sent_request_id = self._sent_request_ids[-1]
        if receive_index == 1:
            try:
                await asyncio.wait_for(
                    self._release_first_response.wait(),
                    timeout=timeout_seconds,
                )
            except TimeoutError:
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
    ) -> None:
        self._state = state
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
    ) -> Ok[CompactPolicyDecisionBatch]:
        self._calls.append("decide_batch")
        batch_size = len(requests.policy_versions)
        self._batch_sizes.append(batch_size)
        return Ok(
            value=_compact_policy_decision_batch(row_count=batch_size)
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


def _compact_policy_decision_batch(
    *, row_count: int
) -> CompactPolicyDecisionBatch:
    trace_token_ids = CompactTraceTokenIds.from_tuple(
        (semantic_argument_id(SemanticArgument("pass")),)
    )
    return CompactPolicyDecisionBatch(
        model_rank_index=0,
        policy_versions=tuple(0 for _ in range(row_count)),
        row_indices=tuple(range(row_count)),
        choice_counts=tuple(1 for _ in range(row_count)),
        trace_token_batch=CompactTraceTokenBatch(
            encoded_i64_rows=trace_token_ids.encoded_i64 * row_count,
            row_count=row_count,
            max_trace_count=1,
            trace_counts=tuple(1 for _ in range(row_count)),
        ),
    )


def _completed_policy_response(
    route: PolicyRequestRoute,
) -> CompletedPolicyResponse:
    decisions = _compact_policy_decision_batch(row_count=1)
    return CompletedPolicyResponse(
        route=route,
        trace_token_ids=decisions.trace_token_batch.compact_row(0),
        decision_handle_model_rank=decisions.model_rank_index,
        decision_handle_policy_version=decisions.policy_versions[0],
        decision_handle_row_index=decisions.row_indices[0],
        choice_count=decisions.choice_counts[0],
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
