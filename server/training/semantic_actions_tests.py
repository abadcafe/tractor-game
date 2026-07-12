"""Tests for semantic action records and binding."""

from __future__ import annotations

from server.foundation.result import Ok, Rejected
from server.game.players.test_helpers import card, make_snapshot
from server.game.protocol import TrickSlotSnapshot, TrickSnapshot
from server.game.rules.card_faces import CardFace, FaceCount
from server.training.semantic_actions import (
    GeneratedAction,
    SemanticArgument,
    SemanticArgumentPrefix,
    SemanticArgumentTrace,
    bind_generated_action,
    build_action_query,
    semantic_prefix_state,
)
from server.training.semantic_actions.codec import (
    semantic_argument_from_id,
    semantic_argument_id,
    semantic_argument_name,
)


def test_bind_generated_action_binds_face_to_current_hand_ids() -> None:
    spade_ace = card("spades", "A", 1)
    heart_five = card("hearts", "5", 1)
    action = GeneratedAction(
        action_kind="play",
        message_type="play",
        face_counts=(
            FaceCount(CardFace(heart_five.suit, heart_five.rank), 1),
        ),
        semantic_trace=SemanticArgumentTrace(
            arguments=(
                SemanticArgument(
                    "select_face_count",
                    FaceCount(
                        CardFace(heart_five.suit, heart_five.rank),
                        1,
                    ),
                ),
                SemanticArgument("stop"),
            )
        ),
        is_pass=False,
    )

    bound = bind_generated_action(action, [spade_ace, heart_five])

    assert isinstance(bound, Ok)
    assert bound.value.raw == {
        "type": "play",
        "cards": [heart_five.id],
    }


def test_bind_generated_action_binds_pair_face_count_to_two_cards() -> (
    None
):
    first_heart_five = card("hearts", "5", 1)
    second_heart_five = card("hearts", "5", 2)
    face = CardFace(first_heart_five.suit, first_heart_five.rank)
    action = GeneratedAction(
        action_kind="play",
        message_type="play",
        face_counts=(FaceCount(face, 2),),
        semantic_trace=SemanticArgumentTrace(arguments=()),
        is_pass=False,
    )

    bound = bind_generated_action(
        action, [first_heart_five, second_heart_five]
    )

    assert isinstance(bound, Ok)
    assert bound.value.raw == {
        "type": "play",
        "cards": [first_heart_five.id, second_heart_five.id],
    }


def test_bind_generated_action_rejects_duplicate_face_counts() -> None:
    first_heart_five = card("hearts", "5", 1)
    second_heart_five = card("hearts", "5", 2)
    face = CardFace(first_heart_five.suit, first_heart_five.rank)
    action = GeneratedAction(
        action_kind="play",
        message_type="play",
        face_counts=(
            FaceCount(face, 1),
            FaceCount(face, 1),
        ),
        semantic_trace=SemanticArgumentTrace(arguments=()),
        is_pass=False,
    )

    bound = bind_generated_action(
        action, [first_heart_five, second_heart_five]
    )

    assert isinstance(bound, Rejected)
    assert "同一牌面重复选择" in bound.reason


def test_bind_generated_action_rejects_missing_face_count() -> None:
    spade_ace = card("spades", "A", 1)
    heart_five = card("hearts", "5", 1)
    action = GeneratedAction(
        action_kind="play",
        message_type="play",
        face_counts=(
            FaceCount(CardFace(heart_five.suit, heart_five.rank), 1),
        ),
        semantic_trace=SemanticArgumentTrace(arguments=()),
        is_pass=False,
    )

    bound = bind_generated_action(action, [spade_ace])

    assert isinstance(bound, Rejected)


def test_semantic_prefix_state_rejects_duplicate_face() -> None:
    first = card("spades", "A", 1)
    spade_single = FaceCount(CardFace(first.suit, first.rank), 1)

    result = semantic_prefix_state(
        SemanticArgumentPrefix(
            arguments=(
                SemanticArgument("select_face_count", spade_single),
                SemanticArgument("select_face_count", spade_single),
            )
        )
    )

    assert isinstance(result, Rejected)


def test_semantic_prefix_state_rejects_argument_after_stop() -> None:
    first = card("spades", "A", 1)
    spade_single = FaceCount(CardFace(first.suit, first.rank), 1)

    result = semantic_prefix_state(
        SemanticArgumentPrefix(
            arguments=(
                SemanticArgument("select_face_count", spade_single),
                SemanticArgument("stop"),
                SemanticArgument("select_face_count", spade_single),
            )
        )
    )

    assert isinstance(result, Rejected)


def test_action_query_requires_current_trick_width_when_following() -> (
    None
):
    lead = card("hearts", "A", 1)
    hand_first = card("spades", "K", 1)
    hand_second = card("clubs", "K", 1)
    trick = TrickSnapshot(
        lead_player=1,
        current_player=2,
        slots=[
            TrickSlotSnapshot(player=0, cards=[]),
            TrickSlotSnapshot(player=1, cards=[lead]),
            TrickSlotSnapshot(player=2, cards=[]),
            TrickSlotSnapshot(player=3, cards=[]),
        ],
    )
    query = build_action_query(
        player_index=2,
        snapshot=make_snapshot(
            phase="PLAYING",
            awaiting_action="play",
            trick=trick,
            player_hand=[hand_first, hand_second],
        ),
    )

    assert query.kind == "follow_play"
    assert query.action_play_order == 1
    assert query.current_trick_width == 1
    assert query.exact_select == 1


def test_semantic_argument_id_round_trips_face_counts() -> None:
    argument = SemanticArgument(
        "select_face_count",
        FaceCount(
            CardFace(
                card("clubs", "3", 1).suit,
                card("clubs", "3", 1).rank,
            ),
            2,
        ),
    )

    decoded = semantic_argument_from_id(semantic_argument_id(argument))

    assert isinstance(decoded, Ok)
    assert decoded.value == argument
    assert semantic_argument_name(argument) == "SELECT_clubs_3_X2"
