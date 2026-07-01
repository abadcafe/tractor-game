"""Tests for model selection action grammar."""

from __future__ import annotations

from server.player.test_helpers import card, make_snapshot
from server.protocol import TrickSlotSnapshot, TrickSnapshot
from server.result import Ok, Rejected
from server.training.selection_actions import (
    SelectionChoice,
    SelectionState,
    SelectionTrace,
    build_action_query,
    decode_selection_action,
    selection_choice_name,
    valid_selection_choices,
)


def test_valid_selection_choices_ignores_action_hints() -> None:
    spade_ace = card("spades", "A", 1)
    heart_five = card("hearts", "5", 1)
    snapshot_without_hints = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[spade_ace, heart_five],
        action_hints=[],
    )
    snapshot_with_hints = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[spade_ace, heart_five],
        action_hints=[[heart_five]],
    )

    query_without_hints = build_action_query(
        player_index=0,
        snapshot=snapshot_without_hints,
    )
    query_with_hints = build_action_query(
        player_index=0,
        snapshot=snapshot_with_hints,
    )

    empty_state = SelectionState(selected_slots=())
    assert valid_selection_choices(
        query_without_hints, empty_state
    ) == valid_selection_choices(query_with_hints, empty_state)


def test_decode_selection_action_play_cards_from_hand_slots() -> None:
    spade_ace = card("spades", "A", 1)
    heart_five = card("hearts", "5", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[spade_ace, heart_five],
    )
    query = build_action_query(player_index=0, snapshot=snapshot)

    result = decode_selection_action(
        query,
        SelectionTrace(
            choices=(
                SelectionChoice("select_card", 1),
                SelectionChoice("stop"),
            )
        ),
    )

    assert isinstance(result, Ok)
    assert result.value.raw == {
        "type": "play",
        "cards": [heart_five.id],
    }
    assert result.value.card_ids == (heart_five.id,)


def test_valid_selection_choices_prevents_duplicate_card_slot() -> None:
    spade_ace = card("spades", "A", 1)
    heart_five = card("hearts", "5", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[spade_ace, heart_five],
    )
    query = build_action_query(player_index=0, snapshot=snapshot)

    allowed = valid_selection_choices(
        query,
        SelectionState(selected_slots=(0,)),
    )

    assert SelectionChoice("select_card", 0) not in allowed
    assert SelectionChoice("select_card", 1) in allowed
    assert SelectionChoice("stop") in allowed


def test_decode_selection_action_rejects_card_slot_outside_hand() -> (
    None
):
    spade_ace = card("spades", "A", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[spade_ace],
    )
    query = build_action_query(player_index=0, snapshot=snapshot)

    result = decode_selection_action(
        query,
        SelectionTrace(
            choices=(
                SelectionChoice("select_card", 1),
                SelectionChoice("stop"),
            )
        ),
    )

    assert isinstance(result, Rejected)


def test_decode_selection_action_rejects_choice_after_stop() -> None:
    spade_ace = card("spades", "A", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[spade_ace],
    )
    query = build_action_query(player_index=0, snapshot=snapshot)

    result = decode_selection_action(
        query,
        SelectionTrace(
            choices=(
                SelectionChoice("select_card", 0),
                SelectionChoice("stop"),
                SelectionChoice("select_card", 0),
            )
        ),
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
    first_card_allowed = valid_selection_choices(
        query,
        SelectionState(selected_slots=()),
    )
    assert SelectionChoice("select_card", 0) in first_card_allowed
    assert SelectionChoice("select_card", 1) in first_card_allowed
    allowed_after_one_card = valid_selection_choices(
        query,
        SelectionState(selected_slots=(0,)),
    )
    assert allowed_after_one_card == ()
    decoded = decode_selection_action(
        query,
        SelectionTrace(choices=(SelectionChoice("select_card", 0),)),
    )
    assert isinstance(decoded, Ok)


def test_selection_choice_name_covers_card_slots() -> None:
    assert selection_choice_name(SelectionChoice("pass")) == "PASS"
    assert (
        selection_choice_name(SelectionChoice("select_card", 7))
        == "SELECT_CARD_7"
    )
