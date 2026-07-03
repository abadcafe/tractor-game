"""Black-box tests for semantic training observation tokens."""

from __future__ import annotations

from server.player.test_helpers import card, make_snapshot
from server.protocol import (
    BidEventSnapshot,
    BottomExchangeSnapshot,
    CompletedTrickSnapshot,
    FailedThrowSnapshot,
    StirDeclarationEventSnapshot,
    TrickSlotSnapshot,
    TrickSnapshot,
)
from server.training.observation import (
    PublicHistoryRecorder,
    build_observation,
    face_count_tokens,
)
from server.training.tokens import (
    ActionQueryFieldToken,
    FaceCountToken,
    RoundEventFieldToken,
    RoundFieldToken,
    TrickResultFieldToken,
)


def test_build_observation_groups_duplicate_hand_faces() -> None:
    first = card("hearts", "A", 1)
    second = card("hearts", "A", 2)
    snapshot = make_snapshot(player_hand=[first, second])

    observation = build_observation(
        player_index=0,
        snapshot=snapshot,
        history=(),
    )

    hand_faces = [
        token
        for token in face_count_tokens(observation)
        if token.segment == "self_hand"
    ]
    assert len(hand_faces) == 1
    assert hand_faces[0].suit == first.suit
    assert hand_faces[0].rank == first.rank
    assert hand_faces[0].count == 2
    assert not hasattr(hand_faces[0], "card_id")
    assert not hasattr(hand_faces[0], "card_order")


def test_build_observation_keeps_face_separate_from_context() -> None:
    test_card = card("spades", "A", 1)
    snapshot = make_snapshot(player_hand=[test_card])

    observation = build_observation(
        player_index=0,
        snapshot=snapshot,
        history=(),
    )

    face_token_values = [
        token
        for token in observation.tokens
        if isinstance(token, FaceCountToken)
    ]
    assert face_token_values
    token = face_token_values[0]
    assert token.suit == test_card.suit
    assert token.rank == test_card.rank
    assert token.segment == "self_hand"
    assert token.role == "self"
    assert not hasattr(token, "is_trump")


def test_build_observation_records_current_score_in_round_context() -> (
    None
):
    snapshot = make_snapshot(
        defender_points=65,
        team0_level="10",
        team1_level="K",
    )

    observation = build_observation(
        player_index=0,
        snapshot=snapshot,
        history=(),
    )

    assert RoundFieldToken("current_score", 65) in observation.tokens
    assert (
        RoundFieldToken("self_team_required_level", "A")
        in observation.tokens
    )
    assert (
        RoundFieldToken("enemy_team_required_level", "A")
        in observation.tokens
    )


def test_build_observation_uses_snapshot_visible_bottom_faces() -> None:
    bottom = card("diamonds", "5", 1)
    hidden_observation = build_observation(
        player_index=0,
        snapshot=make_snapshot(bottom_cards=[]),
        history=(),
    )
    visible_observation = build_observation(
        player_index=0,
        snapshot=make_snapshot(bottom_cards=[bottom]),
        history=(),
    )

    assert not any(
        isinstance(token, FaceCountToken)
        and token.suit == bottom.suit
        and token.rank == bottom.rank
        for token in hidden_observation.tokens
    )
    bottom_tokens = [
        token
        for token in visible_observation.tokens
        if isinstance(token, FaceCountToken)
        and token.segment == "visible_bottom"
    ]
    assert [
        (token.suit, token.rank, token.count) for token in bottom_tokens
    ] == [(bottom.suit, bottom.rank, 1)]


def test_completed_history_records_plays_and_result() -> None:
    first = card("hearts", "A", 1)
    second = card("hearts", "A", 2)
    completed = CompletedTrickSnapshot(
        lead_player=2,
        slots=[
            TrickSlotSnapshot(player=0, cards=[]),
            TrickSlotSnapshot(player=1, cards=[]),
            TrickSlotSnapshot(player=2, cards=[first, second]),
            TrickSlotSnapshot(player=3, cards=[]),
        ],
        winner=2,
        points=35,
    )
    recorder = PublicHistoryRecorder()
    recorder.update(make_snapshot(last_completed_trick=completed))

    observation = build_observation(
        player_index=0,
        snapshot=make_snapshot(),
        history=recorder.tricks(),
    )

    play_faces = [
        token
        for token in face_count_tokens(observation)
        if token.segment == "play_record"
    ]
    assert [(token.role, token.count) for token in play_faces] == [
        ("partner", 2),
    ]
    assert [token.trick_age for token in play_faces] == [1]
    assert [token.trick_state for token in play_faces] == ["completed"]
    assert TrickResultFieldToken("winner", "partner", 1) in (
        observation.tokens
    )
    assert TrickResultFieldToken("points", 35, 1) in observation.tokens


def test_completed_history_records_failed_throw_event() -> None:
    attempted_high = card("spades", "K", 1)
    attempted_low = card("spades", "Q", 1)
    failed_throw = FailedThrowSnapshot(
        player=1,
        attempted_cards=[attempted_high, attempted_low],
        forced_cards=[attempted_low],
    )
    completed = CompletedTrickSnapshot(
        lead_player=1,
        slots=[
            TrickSlotSnapshot(player=0, cards=[]),
            TrickSlotSnapshot(player=1, cards=[attempted_low]),
            TrickSlotSnapshot(player=2, cards=[]),
            TrickSlotSnapshot(player=3, cards=[]),
        ],
        winner=2,
        points=0,
        failed_throw=failed_throw,
    )
    recorder = PublicHistoryRecorder()
    recorder.update(make_snapshot(last_completed_trick=completed))

    observation = build_observation(
        player_index=0,
        snapshot=make_snapshot(),
        history=recorder.tricks(),
    )

    failed_faces = [
        token
        for token in face_count_tokens(observation)
        if token.segment
        in ("failed_throw_attempted", "failed_throw_forced")
    ]
    assert [
        (token.segment, token.role, token.rank, token.count)
        for token in failed_faces
    ] == [
        ("failed_throw_attempted", "left_enemy", attempted_low.rank, 1),
        (
            "failed_throw_attempted",
            "left_enemy",
            attempted_high.rank,
            1,
        ),
        ("failed_throw_forced", "left_enemy", attempted_low.rank, 1),
    ]
    assert [token.trick_age for token in failed_faces] == [1, 1, 1]
    assert [token.trick_state for token in failed_faces] == [
        "completed",
        "completed",
        "completed",
    ]


def test_current_trick_uses_trick_age_zero_and_query_shape() -> None:
    first = card("clubs", "K", 1)
    trick = TrickSnapshot(
        lead_player=1,
        current_player=0,
        slots=[
            TrickSlotSnapshot(player=0, cards=[]),
            TrickSlotSnapshot(player=1, cards=[first]),
            TrickSlotSnapshot(player=2, cards=[]),
            TrickSlotSnapshot(player=3, cards=[]),
        ],
    )
    observation = build_observation(
        player_index=0,
        snapshot=make_snapshot(
            phase="PLAYING",
            awaiting_action="play",
            trick=trick,
            player_hand=[card("spades", "A", 1)],
        ),
        history=(),
    )

    play_faces = [
        token
        for token in face_count_tokens(observation)
        if token.segment == "play_record"
    ]
    assert len(play_faces) == 1
    assert play_faces[0].role == "left_enemy"
    assert play_faces[0].trick_age == 0
    assert play_faces[0].trick_state == "open"
    assert play_faces[0].play_order == 0
    assert (
        ActionQueryFieldToken("current_trick_width", 1)
        in observation.tokens
    )
    assert ActionQueryFieldToken("action_play_order", 3) in (
        observation.tokens
    )


def test_current_trick_records_failed_throw_event() -> None:
    attempted_high = card("clubs", "K", 1)
    attempted_low = card("clubs", "Q", 1)
    trick = TrickSnapshot(
        lead_player=3,
        current_player=0,
        slots=[
            TrickSlotSnapshot(player=0, cards=[]),
            TrickSlotSnapshot(player=1, cards=[]),
            TrickSlotSnapshot(player=2, cards=[]),
            TrickSlotSnapshot(player=3, cards=[attempted_low]),
        ],
        failed_throw=FailedThrowSnapshot(
            player=3,
            attempted_cards=[attempted_high, attempted_low],
            forced_cards=[attempted_low],
        ),
    )

    observation = build_observation(
        player_index=0,
        snapshot=make_snapshot(
            phase="PLAYING",
            awaiting_action="play",
            trick=trick,
            player_hand=[card("spades", "A", 1)],
        ),
        history=(),
    )

    failed_faces = [
        token
        for token in face_count_tokens(observation)
        if token.segment
        in ("failed_throw_attempted", "failed_throw_forced")
    ]
    assert [
        (token.segment, token.role, token.rank, token.count)
        for token in failed_faces
    ] == [
        (
            "failed_throw_attempted",
            "right_enemy",
            attempted_low.rank,
            1,
        ),
        (
            "failed_throw_attempted",
            "right_enemy",
            attempted_high.rank,
            1,
        ),
        ("failed_throw_forced", "right_enemy", attempted_low.rank, 1),
    ]
    assert [token.trick_age for token in failed_faces] == [0, 0, 0]
    assert [token.trick_state for token in failed_faces] == [
        "open",
        "open",
        "open",
    ]


def test_round_events_include_event_age_and_revealed_faces() -> None:
    revealed = card("spades", "2", 1)
    event = BidEventSnapshot(
        player=3,
        cards=[revealed],
        kind="trump_rank",
        suit=revealed.suit,
        joker_type=None,
        count=1,
    )
    observation = build_observation(
        player_index=0,
        snapshot=make_snapshot(bid_events=[event], bid_winner=event),
        history=(),
    )

    assert (
        RoundFieldToken("level_card_revealer_role", "right_enemy")
        in observation.tokens
    )
    assert RoundEventFieldToken("actor", "right_enemy", 1) in (
        observation.tokens
    )
    event_faces = [
        token
        for token in face_count_tokens(observation)
        if token.segment == "round_event"
    ]
    assert [
        (token.suit, token.rank, token.count) for token in event_faces
    ] == [(revealed.suit, revealed.rank, 1)]
    assert event_faces[0].event_age == 1
    assert event_faces[0].role == "right_enemy"


def test_stir_and_own_exchange_history_are_observed() -> None:
    stir_card = card("spades", "2", 1)
    picked = card("diamonds", "3", 1)
    discarded = card("clubs", "4", 1)
    exchange_event = BottomExchangeSnapshot(
        picked_up_bottom_cards=[picked],
        discarded_bottom_cards=[discarded],
    )
    stir_event = StirDeclarationEventSnapshot(
        player=2,
        kind="stir",
        cards=[stir_card],
        new_suit=stir_card.suit,
        priority=203,
        own_bottom_exchange=exchange_event,
    )

    observation = build_observation(
        player_index=0,
        snapshot=make_snapshot(
            stir_events=[stir_event],
        ),
        history=(),
    )

    assert RoundEventFieldToken("stir_kind", "stir", 1) in (
        observation.tokens
    )
    assert RoundEventFieldToken("trigger", "stir", 1) in (
        observation.tokens
    )
    stir_faces = [
        token
        for token in face_count_tokens(observation)
        if token.segment == "stir_event"
    ]
    own_exchange_faces = [
        token
        for token in face_count_tokens(observation)
        if token.segment
        in (
            "own_exchange_pickup",
            "own_exchange_discard",
        )
    ]
    assert [
        (token.suit, token.rank, token.count) for token in stir_faces
    ] == [(stir_card.suit, stir_card.rank, 1)]
    assert [
        (token.suit, token.rank, token.count)
        for token in own_exchange_faces
    ] == [
        (picked.suit, picked.rank, 1),
        (discarded.suit, discarded.rank, 1),
    ]
