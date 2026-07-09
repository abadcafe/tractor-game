"""Tests for policy inference batch response ABI."""

from __future__ import annotations

import struct

from server.player.test_helpers import card, make_snapshot
from server.result import Ok, Rejected
from server.training.legal_actions import (
    LegalActionIndex,
    build_legal_action_index,
)
from server.training.observation import Observation, build_observation
from server.training.policy_inference_batch import (
    CompletedPolicyResponse,
    PolicyRequestRoute,
    PolicyResponseBatchWire,
    build_policy_response_batch_wire,
    decode_policy_response,
    decode_policy_response_batch_wire,
)
from server.training.policy_sampling import (
    CompactTraceTokenIds,
    DecisionHandle,
    ModelRankPolicyDecision,
)
from server.training.semantic_actions import SemanticArgument
from server.training.semantic_actions.codec import semantic_argument_id


def test_policy_response_batch_wire_decodes_rule_action() -> None:
    response_wire = build_policy_response_batch_wire(
        routes=(_route(),),
        decisions=(Ok(value=_model_rank_policy_decision()),),
    )
    assert isinstance(response_wire, Ok)

    decoded_wire = decode_policy_response_batch_wire(
        response_wire.value.data
    )
    assert isinstance(decoded_wire, Ok)
    assert len(decoded_wire.value) == 1
    completed = decoded_wire.value[0]
    assert isinstance(completed, CompletedPolicyResponse)
    assert not isinstance(completed.trace_token_ids, tuple)
    assert completed.trace_token_ids.to_tuple() == (
        semantic_argument_id(SemanticArgument("pass")),
    )
    decoded_decision = decode_policy_response(
        legal_actions=_legal_actions(),
        response=completed,
    )

    assert isinstance(decoded_decision, Ok)
    assert decoded_decision.value.decision_handle.model_rank_index == 1
    assert decoded_decision.value.decision_handle.policy_version == 3


def test_policy_response_batch_wire_rejects_negative_worker_route() -> (
    None
):
    wire = _completed_response_wire()
    corrupted = bytearray(wire.data)
    _I64.pack_into(corrupted, _WORKER_INDEX_OFFSET, -1)

    result = decode_policy_response_batch_wire(bytes(corrupted))

    assert isinstance(result, Rejected)
    assert result.reason == "policy response route is invalid"


def test_decode_response_batch_rejects_negative_request_route() -> None:
    wire = _completed_response_wire()
    corrupted = bytearray(wire.data)
    _I64.pack_into(corrupted, _REQUEST_ID_OFFSET, -1)

    result = decode_policy_response_batch_wire(bytes(corrupted))

    assert isinstance(result, Rejected)
    assert result.reason == "policy response route is invalid"


def _completed_response_wire() -> PolicyResponseBatchWire:
    result = build_policy_response_batch_wire(
        routes=(_route(),),
        decisions=(Ok(value=_model_rank_policy_decision()),),
    )
    assert isinstance(result, Ok)
    return result.value


def _model_rank_policy_decision() -> ModelRankPolicyDecision:
    return ModelRankPolicyDecision(
        trace_token_ids=CompactTraceTokenIds.from_tuple(
            (semantic_argument_id(SemanticArgument("pass")),)
        ),
        decision_handle=DecisionHandle(
            model_rank_index=1,
            policy_version=3,
            row_index=7,
        ),
        choice_count=1,
    )


def _route() -> PolicyRequestRoute:
    return PolicyRequestRoute(worker_index=2, request_id=5)


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


_I64 = struct.Struct("<q")
_HEADER_BYTES = 5 * _I64.size
_WORKER_INDEX_OFFSET = _HEADER_BYTES
_REQUEST_ID_OFFSET = _HEADER_BYTES + _I64.size
