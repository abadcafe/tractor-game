"""Tests for policy request frame public contracts."""

from __future__ import annotations

from server.player.test_helpers import card, make_snapshot
from server.result import Ok
from server.training.legal_actions import build_legal_action_index
from server.training.observation import build_observation
from server.training.policy_request_frame import (
    CompletedPolicyResponseFrame,
    PolicyRequestBatchFrame,
    PolicyRequestFrame,
    RejectedPolicyResponseFrame,
    build_policy_request_frame,
    decode_policy_request_batch_frame,
    decode_policy_request_frame,
    decode_policy_response_frame,
    encode_policy_request_batch_frame,
    encode_policy_request_frame,
    encode_policy_response_frame,
)
from server.training.policy_sampling import DecisionHandle
from server.training.sampling import PolicyDecisionKey
from server.training.semantic_actions import (
    SemanticArgument,
)
from server.training.semantic_actions.codec import semantic_argument_id


def test_policy_request_frame_binary_roundtrip() -> None:
    snapshot = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand=[card("hearts", "2", 1)],
        trump_rank="2",
    )
    observation = build_observation(
        player_index=0,
        snapshot=snapshot,
        history=(),
    )
    legal_actions = build_legal_action_index(
        player_index=0,
        snapshot=snapshot,
        query=observation.action_query,
    )
    frame_result = build_policy_request_frame(
        observation=observation,
        legal_actions=legal_actions,
        decision_key=_decision_key(),
    )
    assert isinstance(frame_result, Ok)

    decoded = decode_policy_request_frame(
        encode_policy_request_frame(frame_result.value)
    )

    assert isinstance(decoded, Ok)
    assert (
        decoded.value.component_rows
        == frame_result.value.component_rows
    )
    assert _float_rows_close(
        decoded.value.numeric_value_rows,
        frame_result.value.numeric_value_rows,
    )
    assert decoded.value.numeric_mask_rows == (
        frame_result.value.numeric_mask_rows
    )
    assert decoded.value.action_plan == frame_result.value.action_plan
    assert decoded.value.decision_key == frame_result.value.decision_key


def test_policy_request_batch_frame_binary_roundtrip() -> None:
    frame = _request_frame()
    batch = PolicyRequestBatchFrame(frames=(frame, frame))

    decoded = decode_policy_request_batch_frame(
        encode_policy_request_batch_frame(batch)
    )

    assert isinstance(decoded, Ok)
    assert len(decoded.value.frames) == 2
    _assert_frames_equal(decoded.value.frames[0], frame)
    _assert_frames_equal(decoded.value.frames[1], frame)


def test_policy_response_frame_binary_roundtrip() -> None:
    frame = CompletedPolicyResponseFrame(
        trace_token_ids=(
            semantic_argument_id(SemanticArgument("pass")),
        ),
        decision_handle=DecisionHandle(
            model_rank_index=1,
            policy_version=2,
            slot_index=3,
            slot_generation=4,
        ),
        choice_count=5,
    )

    decoded = decode_policy_response_frame(
        encode_policy_response_frame(frame)
    )

    assert isinstance(decoded, Ok)
    assert decoded.value == frame


def test_rejected_policy_response_frame_binary_roundtrip() -> None:
    frame = RejectedPolicyResponseFrame(reason="bad policy")

    decoded = decode_policy_response_frame(
        encode_policy_response_frame(frame)
    )

    assert isinstance(decoded, Ok)
    assert decoded.value == frame


def _decision_key() -> PolicyDecisionKey:
    return PolicyDecisionKey(
        base_seed=0,
        policy_version=0,
        episode_id=0,
        player_index=0,
        decision_index=0,
    )


def _request_frame() -> PolicyRequestFrame:
    snapshot = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand=[card("hearts", "2", 1)],
        trump_rank="2",
    )
    observation = build_observation(
        player_index=0,
        snapshot=snapshot,
        history=(),
    )
    legal_actions = build_legal_action_index(
        player_index=0,
        snapshot=snapshot,
        query=observation.action_query,
    )
    frame_result = build_policy_request_frame(
        observation=observation,
        legal_actions=legal_actions,
        decision_key=_decision_key(),
    )
    assert isinstance(frame_result, Ok)
    return frame_result.value


def _assert_frames_equal(
    left: PolicyRequestFrame, right: PolicyRequestFrame
) -> None:
    assert left.component_rows == right.component_rows
    assert _float_rows_close(
        left.numeric_value_rows, right.numeric_value_rows
    )
    assert left.numeric_mask_rows == right.numeric_mask_rows
    assert left.action_plan == right.action_plan
    assert left.decision_key == right.decision_key


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
