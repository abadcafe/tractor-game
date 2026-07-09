"""Tests for columnar policy inference request batches."""

from __future__ import annotations

import torch

from server.player.test_helpers import card, make_snapshot
from server.result import Ok, Rejected
from server.training.feature_schema import NUMERIC_FEATURE_COUNT
from server.training.legal_actions import (
    LegalActionIndex,
    build_legal_action_index,
)
from server.training.observation import Observation, build_observation
from server.training.policy_inference_batch import (
    PolicyRequestBatch,
    PolicyRequestBatchBuilder,
    PolicyRequestInput,
    PolicyRequestRoute,
    materialize_policy_request_batch,
)
from server.training.sampling import PolicyDecisionKey
from server.training.tensorize import OBSERVATION_COMPONENT_COUNT


def test_compile_request_batch_materializes_device_batch() -> None:
    batch_result = _request_batch((_request_input(),))
    assert isinstance(batch_result, Ok)

    device_result = materialize_policy_request_batch(
        batch=batch_result.value,
        device=torch.device("cpu"),
    )

    assert isinstance(device_result, Ok)
    batch = device_result.value
    assert batch_result.value.routes == (
        PolicyRequestRoute(worker_index=2, request_id=5),
    )
    assert batch.observation_batch.component_ids.shape == (
        1,
        batch_result.value.max_observation_tokens,
        OBSERVATION_COMPONENT_COUNT,
    )
    assert batch.observation_batch.numeric_values.shape == (
        1,
        batch_result.value.max_observation_tokens,
        NUMERIC_FEATURE_COUNT,
    )
    assert batch.policy_versions == (_decision_key().policy_version,)
    assert (
        batch.padded_generation_steps
        == batch_result.value.padded_generation_steps
    )
    assert tuple(
        int(value) for value in batch.generation_step_counts
    ) == (batch.padded_generation_steps,)


def test_compile_request_batch_accepts_mixed_generation_widths() -> (
    None
):
    batch_result = _request_batch(
        (_request_input(), _play_request_input())
    )

    assert isinstance(batch_result, Ok)
    batch = batch_result.value
    assert batch.row_count() == 2
    assert (
        batch.generation_step_counts[0]
        < (batch.generation_step_counts[1])
    )
    assert batch.padded_generation_steps == max(
        batch.generation_step_counts
    )


def test_compile_request_batch_rejects_token_over_budget() -> None:
    preparer = PolicyRequestBatchBuilder(
        batch_capacity=4,
        max_observation_tokens=1,
    )
    result = preparer.push_request(_request_input())

    assert isinstance(result, Rejected)
    assert (
        result.reason
        == "policy request observation exceeds token budget"
    )


def _request_batch(
    requests: tuple[PolicyRequestInput, ...],
) -> Ok[PolicyRequestBatch] | Rejected:
    preparer = PolicyRequestBatchBuilder(
        batch_capacity=4,
        max_observation_tokens=512,
    )
    return preparer.compile_batch(requests)


def _request_input() -> PolicyRequestInput:
    observation = _observation()
    return PolicyRequestInput(
        route=PolicyRequestRoute(worker_index=2, request_id=5),
        observation=observation,
        legal_actions=_legal_actions(observation),
        decision_key=_decision_key(),
    )


def _play_request_input() -> PolicyRequestInput:
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[
            card("hearts", "2", 1),
            card("hearts", "3", 1),
            card("hearts", "4", 1),
        ],
        trump_rank="2",
    )
    observation = build_observation(
        player_index=0,
        snapshot=snapshot,
        history=(),
    )
    return PolicyRequestInput(
        route=PolicyRequestRoute(worker_index=2, request_id=6),
        observation=observation,
        legal_actions=build_legal_action_index(
            player_index=0,
            snapshot=snapshot,
            query=observation.action_query,
        ),
        decision_key=PolicyDecisionKey(
            base_seed=0,
            policy_version=3,
            episode_id=0,
            player_index=0,
            decision_index=1,
        ),
    )


def _observation() -> Observation:
    return build_observation(
        player_index=0,
        snapshot=make_snapshot(
            phase="DEAL_BID",
            awaiting_action="bid",
            player_hand=[card("hearts", "2", 1)],
            trump_rank="2",
        ),
        history=(),
    )


def _legal_actions(observation: Observation) -> LegalActionIndex:
    return build_legal_action_index(
        player_index=0,
        snapshot=make_snapshot(
            phase="DEAL_BID",
            awaiting_action="bid",
            player_hand=[card("hearts", "2", 1)],
            trump_rank="2",
        ),
        query=observation.action_query,
    )


def _decision_key() -> PolicyDecisionKey:
    return PolicyDecisionKey(
        base_seed=0,
        policy_version=3,
        episode_id=0,
        player_index=0,
        decision_index=0,
    )
