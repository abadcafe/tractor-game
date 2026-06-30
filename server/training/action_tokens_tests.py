"""Tests for model action token grammar."""

from __future__ import annotations

from server.player.test_helpers import card, make_snapshot
from server.protocol import TrickSlotSnapshot, TrickSnapshot
from server.result import Ok, Rejected
from server.training.action_tokens import (
    ACTION_PLAY_TOKEN_ID,
    ACTION_TOKEN_VOCAB_SIZE,
    BEGIN_TOKEN_ID,
    FIRST_CARD_TOKEN_ID,
    STOP_TOKEN_ID,
    build_action_query,
    decode_action_tokens,
    token_name,
    valid_next_token_ids,
)


def test_valid_next_token_ids_ignores_action_hints() -> None:
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

    assert valid_next_token_ids(
        query_without_hints, (BEGIN_TOKEN_ID,)
    ) == valid_next_token_ids(query_with_hints, (BEGIN_TOKEN_ID,))


def test_decode_action_tokens_play_cards_from_hand_slots() -> None:
    spade_ace = card("spades", "A", 1)
    heart_five = card("hearts", "5", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[spade_ace, heart_five],
    )
    query = build_action_query(player_index=0, snapshot=snapshot)

    result = decode_action_tokens(
        query,
        (
            BEGIN_TOKEN_ID,
            ACTION_PLAY_TOKEN_ID,
            FIRST_CARD_TOKEN_ID + 1,
            STOP_TOKEN_ID,
        ),
    )

    assert isinstance(result, Ok)
    assert result.value.raw == {
        "type": "play",
        "cards": [heart_five.id],
    }
    assert result.value.card_ids == (heart_five.id,)


def test_valid_next_token_ids_prevents_duplicate_card_slot() -> None:
    spade_ace = card("spades", "A", 1)
    heart_five = card("hearts", "5", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[spade_ace, heart_five],
    )
    query = build_action_query(player_index=0, snapshot=snapshot)

    allowed = valid_next_token_ids(
        query,
        (
            BEGIN_TOKEN_ID,
            ACTION_PLAY_TOKEN_ID,
            FIRST_CARD_TOKEN_ID,
        ),
    )

    assert FIRST_CARD_TOKEN_ID not in allowed
    assert FIRST_CARD_TOKEN_ID + 1 in allowed
    assert STOP_TOKEN_ID in allowed


def test_decode_action_tokens_rejects_card_slot_outside_hand() -> None:
    spade_ace = card("spades", "A", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[spade_ace],
    )
    query = build_action_query(player_index=0, snapshot=snapshot)

    result = decode_action_tokens(
        query,
        (
            BEGIN_TOKEN_ID,
            ACTION_PLAY_TOKEN_ID,
            FIRST_CARD_TOKEN_ID + 1,
            STOP_TOKEN_ID,
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

    assert query.action_play_order == 1
    assert query.current_trick_width == 1
    assert query.exact_select == 1
    first_card_allowed = valid_next_token_ids(
        query,
        (BEGIN_TOKEN_ID, ACTION_PLAY_TOKEN_ID),
    )
    assert FIRST_CARD_TOKEN_ID in first_card_allowed
    assert FIRST_CARD_TOKEN_ID + 1 in first_card_allowed
    allowed_after_one_card = valid_next_token_ids(
        query,
        (
            BEGIN_TOKEN_ID,
            ACTION_PLAY_TOKEN_ID,
            FIRST_CARD_TOKEN_ID,
        ),
    )
    assert allowed_after_one_card == (STOP_TOKEN_ID,)


def test_token_name_covers_action_vocab_boundary() -> None:
    assert token_name(BEGIN_TOKEN_ID) == "BEGIN"
    assert token_name(ACTION_TOKEN_VOCAB_SIZE) == (
        f"UNKNOWN_{ACTION_TOKEN_VOCAB_SIZE}"
    )
