"""Tests for model-rank package public boundaries."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
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
from server.training.policy_inference_wire import (
    PolicyRequestMetadata,
    PolicyRequestRoute,
    PolicyRequestWireBatch,
    build_completed_policy_response_wire,
    decode_policy_request_metadata,
    max_policy_request_wire_bytes,
)
from server.training.policy_sampling import (
    DecisionHandle,
    ModelRankPolicyDecision,
)
from server.training.ppo import PPOUpdateProfile, PPOUpdateStats
from server.training.returns import ReturnCommit
from server.training.runtime.model_rank import (
    DirectPolicyClient,
    FramedPolicyClient,
    LocalModelRank,
)
from server.training.runtime.model_rank.inference_transport import (
    ConnectionPolicyRequestReceiver,
    ConnectionPolicyRequestSender,
    ConnectionPolicyResponseReceiver,
    send_policy_response,
)
from server.training.runtime.state import RuntimeTrainingState
from server.training.sampling import PolicyDecisionKey
from server.training.semantic_actions import (
    SemanticArgument,
    SemanticArgumentTrace,
)
from server.training.semantic_actions.codec import semantic_argument_id


@pytest.mark.asyncio
async def test_direct_policy_client_calls_replica_without_queue() -> (
    None
):
    observation = _observation()
    legal_actions = _legal_actions(observation)
    rank_decision = _model_rank_policy_decision()
    replica = _FakeReplica(
        state=_runtime_state(), decision=rank_decision
    )

    result = await DirectPolicyClient(replica=replica).decide(
        observation,
        legal_actions,
        _decision_key(),
    )

    assert isinstance(result, Ok)
    assert result.value.action.semantic_trace == _pass_trace()
    assert result.value.decision_handle == rank_decision.decision_handle
    assert result.value.choice_count == rank_decision.choice_count
    assert replica.calls == ("decide_wires",)


def test_local_model_rank_loads_updates_and_snapshots_replica() -> None:
    state = _runtime_state()
    replica = _FakeReplica(
        state=state,
        decision=_model_rank_policy_decision(),
    )
    rank = LocalModelRank(replica=replica)

    load_result = rank.load_state(state=state, policy_version=3)
    update_result = rank.update(
        commit=_return_commit(),
        policy_version=3,
    )
    snapshot_result = rank.snapshot()

    assert isinstance(load_result, Ok)
    assert isinstance(update_result, Ok)
    assert isinstance(snapshot_result, Ok)
    assert replica.calls == (
        "load_state",
        "update_commit",
        "snapshot",
        "snapshot",
    )
    assert update_result.value.state is state
    assert snapshot_result.value is state


@pytest.mark.asyncio
async def test_framed_policy_client_roundtrips_connection_payload() -> (
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
            FramedPolicyClient(
                worker_index=2,
                request_sender=ConnectionPolicyRequestSender(
                    connection=request_sender
                ),
                response_receiver=(
                    ConnectionPolicyResponseReceiver(response_receiver)
                ),
                timeout_seconds=1.0,
            ).decide(
                observation,
                legal_actions,
                _decision_key(),
            )
        )
        metadata = await _receive_request_metadata(
            request_receiver=request_receiver
        )
        assert metadata.route.worker_index == 2
        assert metadata.route.request_id == 0
        send_result = send_policy_response(
            sender=response_sender,
            response=build_completed_policy_response_wire(
                route=metadata.route,
                decision=rank_decision,
            ),
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
async def test_framed_policy_client_rejects_response_mismatch() -> None:
    context = mp.get_context("spawn")
    request_receiver, request_sender = context.Pipe(duplex=False)
    response_receiver, response_sender = context.Pipe(duplex=False)
    try:
        observation = _observation()
        legal_actions = _legal_actions(observation)
        task = asyncio.create_task(
            FramedPolicyClient(
                worker_index=2,
                request_sender=ConnectionPolicyRequestSender(
                    connection=request_sender
                ),
                response_receiver=(
                    ConnectionPolicyResponseReceiver(response_receiver)
                ),
                timeout_seconds=1.0,
            ).decide(
                observation,
                legal_actions,
                _decision_key(),
            )
        )
        metadata = await _receive_request_metadata(
            request_receiver=request_receiver
        )
        assert metadata.route.request_id == 0
        send_result = send_policy_response(
            sender=response_sender,
            response=build_completed_policy_response_wire(
                route=PolicyRequestRoute(
                    worker_index=3,
                    request_id=metadata.route.request_id,
                ),
                decision=_model_rank_policy_decision(),
            ),
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


async def _receive_request_metadata(
    *,
    request_receiver: Connection,
) -> PolicyRequestMetadata:
    request_buffer = bytearray(
        max_policy_request_wire_bytes(max_observation_tokens=512)
    )
    request_result = await asyncio.to_thread(
        ConnectionPolicyRequestReceiver(
            connection=request_receiver
        ).receive_bytes_into,
        memoryview(request_buffer),
    )
    assert isinstance(request_result, Ok)
    metadata = decode_policy_request_metadata(
        memoryview(request_buffer)[: request_result.value]
    )
    assert isinstance(metadata, Ok)
    return metadata.value


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

    @property
    def calls(self) -> tuple[str, ...]:
        return tuple(self._calls)

    def load_state(self, *, snapshot: RuntimeTrainingState) -> None:
        assert snapshot is self._state
        self._calls.append("load_state")

    def decide_wires(
        self, requests: PolicyRequestWireBatch
    ) -> tuple[Ok[ModelRankPolicyDecision], ...]:
        assert requests.batch_size() == 1
        self._calls.append("decide_wires")
        return (Ok(value=self._decision),)

    def update_commit(
        self, *, commit: ReturnCommit
    ) -> Ok[PPOUpdateStats]:
        assert not commit.is_empty()
        self._calls.append("update_commit")
        return Ok(value=_ppo_update_stats())

    def snapshot(self) -> RuntimeTrainingState:
        self._calls.append("snapshot")
        return self._state


def _model_rank_policy_decision() -> ModelRankPolicyDecision:
    return ModelRankPolicyDecision(
        trace_token_ids=(
            semantic_argument_id(SemanticArgument("pass")),
        ),
        decision_handle=DecisionHandle(
            model_rank_index=0,
            policy_version=0,
            slot_index=0,
            slot_generation=1,
        ),
        choice_count=1,
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


def _return_commit() -> ReturnCommit:
    return ReturnCommit(
        policy_version=3,
        first_episode_id=0,
        episode_count=1,
        decision_handles=(
            DecisionHandle(
                model_rank_index=0,
                policy_version=3,
                slot_index=0,
                slot_generation=1,
            ),
        ),
        return_values=(1.0,),
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
            argument_prefix_tensorize_seconds=0.0,
            argument_decode_seconds=0.0,
            argument_distribution_seconds=0.0,
            backward_seconds=0.0,
            optimizer_step_seconds=0.0,
            argument_decode_fraction=0.0,
            argument_prefix_batch_count=0,
            argument_prefix_row_count=0,
            argument_prefix_token_count=0,
            argument_prefix_valid_token_count=0,
            argument_prefix_padding_token_count=0,
        ),
    )
