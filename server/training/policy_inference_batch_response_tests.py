"""Black-box tests for strict action-choice response transport."""

from __future__ import annotations

import torch

from server.foundation.result import Ok, Rejected
from server.game.players.test_helpers import card, make_snapshot
from server.game.rules.card_faces import CardFace, FaceCount
from server.training.legal_actions import build_legal_action_index
from server.training.policy_inference_batch import (
    CompletedPolicyResponse,
    PolicyRequestRoute,
    RejectedPolicyResponse,
    build_completed_policy_responses,
    decode_policy_response,
    decode_policy_response_batch_wire,
    encode_policy_response_batch_wire,
)
from server.training.policy_sampling import (
    CompactActionChoiceBatch,
    CompactActionChoiceIds,
    CompactPolicyDecisionBatch,
)
from server.training.semantic_actions.choices import (
    FINISH_CHOICE_ID,
    PASS_CHOICE_ID,
    ActionChoice,
    action_choice_id,
)


def test_response_wire_round_trips_completed_and_rejected_rows() -> (
    None
):
    completed = CompletedPolicyResponse(
        route=PolicyRequestRoute(worker_index=2, request_id=7),
        action_choice_ids=CompactActionChoiceIds.from_tuple(
            (PASS_CHOICE_ID,)
        ),
        decision_handle_model_rank=1,
        decision_handle_policy_version=5,
        decision_handle_row_index=13,
        choice_count=1,
    )
    rejected = RejectedPolicyResponse(
        route=PolicyRequestRoute(worker_index=3, request_id=8),
        reason="policy unavailable",
    )

    encoded = encode_policy_response_batch_wire((completed, rejected))
    assert isinstance(encoded, Ok)
    decoded = decode_policy_response_batch_wire(encoded.value.data)

    assert isinstance(decoded, Ok)
    assert decoded.value == (completed, rejected)


def test_completed_response_batch_keeps_fixed_choice_ids() -> None:
    choice_ids = torch.tensor(
        ((PASS_CHOICE_ID, 0), (FINISH_CHOICE_ID, 0)),
        dtype=torch.long,
    )
    decisions = CompactPolicyDecisionBatch(
        model_rank_index=1,
        policy_versions=(4, 4),
        row_indices=(10, 11),
        choice_counts=(1, 1),
        action_choice_batch=CompactActionChoiceBatch.from_cpu_tensor(
            choice_ids=choice_ids,
            choice_counts=(1, 1),
        ),
    )
    routes = (
        PolicyRequestRoute(worker_index=0, request_id=3),
        PolicyRequestRoute(worker_index=0, request_id=4),
    )

    result = build_completed_policy_responses(
        routes=routes, decisions=decisions
    )

    assert isinstance(result, Ok)
    assert result.value[0].action_choice_ids.to_tuple() == (
        PASS_CHOICE_ID,
    )
    assert result.value[1].action_choice_ids.to_tuple() == (
        FINISH_CHOICE_ID,
    )


def test_worker_decodes_card_then_finish_through_legal_rules() -> None:
    ace = card("spades", "A", 1)
    snapshot = make_snapshot(player_hand=[ace])
    legal = build_legal_action_index(player_index=0, snapshot=snapshot)
    card_choice_id = action_choice_id(
        ActionChoice(
            "card",
            FaceCount(CardFace(ace.suit, ace.rank), 1),
        )
    )
    response = CompletedPolicyResponse(
        route=PolicyRequestRoute(worker_index=0, request_id=1),
        action_choice_ids=CompactActionChoiceIds.from_tuple(
            (card_choice_id, FINISH_CHOICE_ID)
        ),
        decision_handle_model_rank=2,
        decision_handle_policy_version=6,
        decision_handle_row_index=9,
        choice_count=2,
    )

    decoded = decode_policy_response(
        legal_actions=legal, response=response
    )

    assert isinstance(decoded, Ok)
    assert decoded.value.action.face_counts == (
        FaceCount(CardFace(ace.suit, ace.rank), 1),
    )
    assert decoded.value.decision_handle.row_index == 9


def test_response_wire_rejects_invalid_schema_magic() -> None:
    response = RejectedPolicyResponse(
        route=PolicyRequestRoute(worker_index=0, request_id=1),
        reason="no model",
    )
    encoded = encode_policy_response_batch_wire((response,))
    assert isinstance(encoded, Ok)
    invalid = bytearray(encoded.value.data)
    invalid[0] ^= 1

    decoded = decode_policy_response_batch_wire(bytes(invalid))

    assert isinstance(decoded, Rejected)
    assert (
        decoded.reason == "policy response batch wire schema is invalid"
    )
