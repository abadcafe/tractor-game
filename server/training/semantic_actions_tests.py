"""Tests for semantic action records and binding."""

from __future__ import annotations

from server.foundation.result import Ok, Rejected
from server.game.players.test_helpers import card, make_snapshot
from server.game.protocol import TrickSlotSnapshot, TrickSnapshot
from server.game.rules.card_faces import CardFace, FaceCount
from server.training.semantic_actions import (
    ActionChoice,
    ActionPrefix,
    ActionTrace,
    GeneratedAction,
    action_prefix_cards,
    bind_generated_action,
    build_action_query,
)
from server.training.semantic_actions.choices import (
    action_choice_from_id,
    action_choice_id,
    action_choice_name,
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
        trace=ActionTrace(
            choices=(
                ActionChoice(
                    "card",
                    FaceCount(
                        CardFace(heart_five.suit, heart_five.rank),
                        1,
                    ),
                ),
                ActionChoice("finish"),
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
        trace=ActionTrace(choices=()),
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
        trace=ActionTrace(choices=()),
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
        trace=ActionTrace(choices=()),
        is_pass=False,
    )

    bound = bind_generated_action(action, [spade_ace])

    assert isinstance(bound, Rejected)


def test_action_prefix_rejects_duplicate_face() -> None:
    first = card("spades", "A", 1)
    spade_single = FaceCount(CardFace(first.suit, first.rank), 1)

    result = action_prefix_cards(
        ActionPrefix(
            choices=(
                ActionChoice("card", spade_single),
                ActionChoice("card", spade_single),
            )
        )
    )

    assert isinstance(result, Rejected)


def test_action_prefix_rejects_choice_after_finish() -> None:
    first = card("spades", "A", 1)
    spade_single = FaceCount(CardFace(first.suit, first.rank), 1)

    result = action_prefix_cards(
        ActionPrefix(
            choices=(
                ActionChoice("card", spade_single),
                ActionChoice("finish"),
                ActionChoice("card", spade_single),
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


def test_action_choice_id_round_trips_face_counts() -> None:
    choice = ActionChoice(
        "card",
        FaceCount(
            CardFace(
                card("clubs", "3", 1).suit,
                card("clubs", "3", 1).rank,
            ),
            2,
        ),
    )

    decoded = action_choice_from_id(action_choice_id(choice))

    assert isinstance(decoded, Ok)
    assert decoded.value == choice
    assert action_choice_name(choice) == "CARD_clubs_3_X2"
