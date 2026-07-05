"""Tests for model-rank package public boundaries."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

import torch

from server.player.test_helpers import card, make_snapshot
from server.result import Ok
from server.training.legal_actions import (
    LegalActionIndex,
    build_legal_action_index,
)
from server.training.observation import (
    Observation,
    build_observation,
)
from server.training.policy_request_frame import (
    CompletedPolicyResponseFrame,
    PolicyRequestBatchFrame,
    build_policy_request_frame,
)
from server.training.policy_sampling import (
    DecisionHandle,
    ModelRankPolicyDecision,
)
from server.training.ppo import PPOUpdateProfile, PPOUpdateStats
from server.training.rollout_commit import RolloutCommit
from server.training.runtime.model_rank import (
    DirectPolicyClient,
    FramedPolicyClient,
    InlineModelRank,
)
from server.training.runtime.model_rank.inference_transport import (
    ConnectionPolicyResponseReceiver,
    PolicyInferenceRequestBatch,
    PolicyInferenceResponseEnvelope,
    SharedMemoryPolicyRequestReceiver,
    SharedMemoryPolicyRequestSender,
    receive_policy_request_batch,
    send_policy_response,
)
from server.training.runtime.state import RuntimeTrainingState
from server.training.runtime.update_wave import SynchronizedUpdateShard
from server.training.sampling import PolicyDecisionKey
from server.training.semantic_actions import (
    SemanticArgument,
    SemanticArgumentTrace,
)
from server.training.semantic_actions.codec import semantic_argument_id


def test_direct_policy_client_calls_replica_without_queue() -> None:
    observation = _observation()
    legal_actions = _legal_actions(observation)
    rank_decision = _model_rank_policy_decision()
    replica = _FakeReplica(
        state=_runtime_state(), decision=rank_decision
    )

    result = DirectPolicyClient(replica=replica).decide(
        observation,
        legal_actions,
        _decision_key(),
    )

    assert isinstance(result, Ok)
    assert result.value.action.semantic_trace == _pass_trace()
    assert result.value.decision_handle == rank_decision.decision_handle
    assert result.value.choice_count == rank_decision.choice_count
    assert replica.calls == ("decide_batch",)


def test_inline_model_rank_loads_updates_and_snapshots_replica() -> (
    None
):
    state = _runtime_state()
    replica = _FakeReplica(
        state=state,
        decision=_model_rank_policy_decision(),
    )
    rank = InlineModelRank(replica=replica)

    load_result = rank.load_state(state=state, policy_version=3)
    update_result = rank.update(
        shard=_update_shard(),
        policy_version=3,
    )
    snapshot_result = rank.snapshot()

    assert isinstance(load_result, Ok)
    assert isinstance(update_result, Ok)
    assert isinstance(snapshot_result, Ok)
    assert replica.calls == (
        "load_state",
        "update_shard",
        "snapshot",
        "snapshot",
    )
    assert update_result.value.state is state
    assert snapshot_result.value is state


def test_framed_policy_client_sends_request_and_reads_response() -> (
    None
):
    context = mp.get_context("spawn")
    request_receiver, request_sender = context.Pipe(duplex=False)
    response_receiver, response_sender = context.Pipe(duplex=False)
    request_slot = SharedMemory(
        create=True, size=1024 * 1024, track=False
    )
    rank_decision = _model_rank_policy_decision()
    try:
        send_result = send_policy_response(
            sender=response_sender,
            envelope=PolicyInferenceResponseEnvelope(
                worker_index=2,
                request_id=0,
                frame=CompletedPolicyResponseFrame(
                    trace_token_ids=rank_decision.trace_token_ids,
                    decision_handle=rank_decision.decision_handle,
                    choice_count=rank_decision.choice_count,
                ),
            ),
        )
        assert isinstance(send_result, Ok)
        observation = _observation()
        legal_actions = _legal_actions(observation)

        result = FramedPolicyClient(
            worker_index=2,
            request_sender=SharedMemoryPolicyRequestSender(
                connection=request_sender,
                slot_name=request_slot.name,
                slot_size=1024 * 1024,
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

        assert isinstance(result, Ok)
        assert result.value.action.semantic_trace == _pass_trace()
        assert (
            result.value.decision_handle
            == rank_decision.decision_handle
        )
        assert result.value.choice_count == rank_decision.choice_count
        request_result = receive_policy_request_batch(
            receivers=(
                SharedMemoryPolicyRequestReceiver(
                    connection=request_receiver
                ),
            ),
            batch_size=1,
            wait_seconds=0.0,
        )
        assert isinstance(request_result, Ok)
        request_batch = request_result.value
        assert isinstance(request_batch, PolicyInferenceRequestBatch)
        request = request_batch.requests[0]
        assert request.worker_index == 2
        assert request.request_id == 0
        expected_frame = build_policy_request_frame(
            observation=observation,
            legal_actions=legal_actions,
            decision_key=_decision_key(),
        )
        assert isinstance(expected_frame, Ok)
        assert request.frame.component_rows == (
            expected_frame.value.component_rows
        )
        assert request.frame.numeric_mask_rows == (
            expected_frame.value.numeric_mask_rows
        )
        assert _float_rows_close(
            request.frame.numeric_value_rows,
            expected_frame.value.numeric_value_rows,
        )
        assert (
            request.frame.action_plan
            == expected_frame.value.action_plan
        )
        assert request.frame.decision_key == (
            expected_frame.value.decision_key
        )
    finally:
        request_slot.close()
        request_slot.unlink()


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

    def decide_batch(
        self, requests: PolicyRequestBatchFrame
    ) -> tuple[Ok[ModelRankPolicyDecision], ...]:
        assert len(requests.frames) == 1
        self._calls.append("decide_batch")
        return (Ok(value=self._decision),)

    def update_shard(
        self, *, shard: SynchronizedUpdateShard
    ) -> Ok[PPOUpdateStats]:
        assert not shard.is_empty()
        self._calls.append("update_shard")
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


def _float_rows_close(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
) -> bool:
    if len(left) != len(right):
        return False
    for left_row, right_row in zip(left, right, strict=True):
        if len(left_row) != len(right_row):
            return False
        for left_value, right_value in zip(
            left_row, right_row, strict=True
        ):
            if abs(left_value - right_value) > 0.000001:
                return False
    return True


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


def _update_shard() -> SynchronizedUpdateShard:
    return SynchronizedUpdateShard(
        rank_index=0,
        policy_version=3,
        rollout_commit=_rollout_commit(),
    )


def _rollout_commit() -> RolloutCommit:
    return RolloutCommit(
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
        reward_after_step=(0.0,),
        terminal_rewards=(1.0,),
        trajectory_team_indices=(0,),
        trajectory_offsets=(0, 1),
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
