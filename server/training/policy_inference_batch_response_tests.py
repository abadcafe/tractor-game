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
    RejectedPolicyResponse,
    build_completed_policy_responses,
    build_rejected_policy_responses,
    decode_policy_response,
    decode_policy_response_batch_wire,
    encode_policy_response_batch_wire,
)
from server.training.policy_sampling import (
    CompactPolicyDecisionBatch,
    CompactTraceTokenBatch,
    CompactTraceTokenIds,
)
from server.training.semantic_actions import SemanticArgument
from server.training.semantic_actions.codec import semantic_argument_id


def test_policy_response_batch_wire_decodes_rule_action() -> None:
    responses = build_completed_policy_responses(
        routes=(_route(),),
        decisions=_compact_policy_decision_batch(),
    )
    assert isinstance(responses, Ok)
    response_wire = encode_policy_response_batch_wire(responses.value)
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


def test_policy_response_batch_wire_decodes_mixed_rows() -> None:
    completed = build_completed_policy_responses(
        routes=(_route(),),
        decisions=_compact_policy_decision_batch(),
    )
    rejected = build_rejected_policy_responses(
        routes=(PolicyRequestRoute(worker_index=2, request_id=6),),
        reason="bad logits",
    )
    assert isinstance(completed, Ok)
    assert isinstance(rejected, Ok)

    wire = encode_policy_response_batch_wire(
        completed.value + rejected.value
    )
    assert isinstance(wire, Ok)
    decoded = decode_policy_response_batch_wire(wire.value.data)

    assert isinstance(decoded, Ok)
    assert len(decoded.value) == 2
    assert isinstance(decoded.value[0], CompletedPolicyResponse)
    assert isinstance(decoded.value[1], RejectedPolicyResponse)
    assert decoded.value[1].route.request_id == 6
    assert decoded.value[1].reason == "bad logits"


def test_build_completed_policy_responses_uses_compact_decisions() -> (
    None
):
    result = build_completed_policy_responses(
        routes=(_route(),),
        decisions=_compact_policy_decision_batch(),
    )

    assert isinstance(result, Ok)
    assert len(result.value) == 1
    response = result.value[0]
    assert response.route == _route()
    assert response.decision_handle_model_rank == 1
    assert response.decision_handle_policy_version == 3
    assert response.decision_handle_row_index == 7
    assert response.choice_count == 1
    assert response.trace_token_ids.to_tuple() == (
        semantic_argument_id(SemanticArgument("pass")),
    )


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
    responses = build_completed_policy_responses(
        routes=(_route(),),
        decisions=_compact_policy_decision_batch(),
    )
    assert isinstance(responses, Ok)
    result = encode_policy_response_batch_wire(responses.value)
    assert isinstance(result, Ok)
    return result.value


def _compact_policy_decision_batch() -> CompactPolicyDecisionBatch:
    trace_token_ids = CompactTraceTokenIds.from_tuple(
        (semantic_argument_id(SemanticArgument("pass")),)
    )
    return CompactPolicyDecisionBatch(
        model_rank_index=1,
        policy_versions=(3,),
        row_indices=(7,),
        choice_counts=(1,),
        trace_token_batch=CompactTraceTokenBatch(
            encoded_i64_rows=trace_token_ids.encoded_i64,
            row_count=1,
            max_trace_count=1,
            trace_counts=(1,),
        ),
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
