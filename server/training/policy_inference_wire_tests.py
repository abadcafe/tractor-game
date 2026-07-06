"""Tests for policy inference wire public contract."""

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
from server.training.policy_inference_wire import (
    PolicyRequestWire,
    build_completed_policy_response_wire,
    build_policy_request_wire,
    decode_policy_request_metadata,
    decode_policy_response,
    decode_policy_response_wire,
    device_policy_request_batch_from_wire,
)
from server.training.policy_sampling import (
    DecisionHandle,
    ModelRankPolicyDecision,
)
from server.training.sampling import PolicyDecisionKey
from server.training.semantic_actions import SemanticArgument
from server.training.semantic_actions.codec import semantic_argument_id
from server.training.tensorize import OBSERVATION_COMPONENT_COUNT


def test_policy_request_wire_stages_device_batch() -> None:
    observation = _observation()
    wire_result = build_policy_request_wire(
        worker_index=2,
        request_id=5,
        observation=observation,
        legal_actions=_legal_actions(),
        decision_key=_decision_key(),
    )
    assert isinstance(wire_result, Ok)

    metadata_result = decode_policy_request_metadata(
        wire_result.value.data
    )
    assert isinstance(metadata_result, Ok)
    metadata = metadata_result.value
    assert metadata.route.worker_index == 2
    assert metadata.route.request_id == 5

    device_result = device_policy_request_batch_from_wire(
        device_bytes=torch.tensor(
            tuple(wire_result.value.data),
            dtype=torch.uint8,
        ).reshape(1, -1),
        metadata=(metadata,),
        max_observation_tokens=512,
    )

    assert isinstance(device_result, Ok)
    batch = device_result.value
    assert batch.observation_batch.component_ids.shape == (
        1,
        metadata.token_count,
        OBSERVATION_COMPONENT_COUNT,
    )
    assert batch.observation_batch.numeric_values.shape == (
        1,
        metadata.token_count,
        NUMERIC_FEATURE_COUNT,
    )
    assert batch.policy_versions == (_decision_key().policy_version,)


def test_policy_response_wire_decodes_rule_action() -> None:
    legal_actions = _legal_actions()
    metadata_result = decode_policy_request_metadata(
        _request_wire().data
    )
    assert isinstance(metadata_result, Ok)
    response_wire = build_completed_policy_response_wire(
        route=metadata_result.value.route,
        decision=ModelRankPolicyDecision(
            trace_token_ids=(
                semantic_argument_id(SemanticArgument("pass")),
            ),
            decision_handle=DecisionHandle(
                model_rank_index=1,
                policy_version=3,
                slot_index=7,
                slot_generation=11,
            ),
            choice_count=1,
        ),
    )

    decoded_wire = decode_policy_response_wire(response_wire.data)
    assert isinstance(decoded_wire, Ok)
    decoded_decision = decode_policy_response(
        legal_actions=legal_actions,
        response=decoded_wire.value,
    )

    assert isinstance(decoded_decision, Ok)
    assert decoded_decision.value.decision_handle.model_rank_index == 1
    assert decoded_decision.value.decision_handle.policy_version == 3


def test_policy_request_wire_rejects_bad_length() -> None:
    wire = bytearray(_request_wire().data)
    wire.pop()

    result = decode_policy_request_metadata(bytes(wire))

    assert isinstance(result, Rejected)
    assert result.reason == "policy request wire length mismatch"


def _request_wire() -> PolicyRequestWire:
    wire_result = build_policy_request_wire(
        worker_index=0,
        request_id=0,
        observation=_observation(),
        legal_actions=_legal_actions(),
        decision_key=_decision_key(),
    )
    assert isinstance(wire_result, Ok)
    return wire_result.value


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


def _legal_actions() -> LegalActionIndex:
    observation = _observation()
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
