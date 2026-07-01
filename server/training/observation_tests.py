"""Black-box tests for training observation tokens."""

from __future__ import annotations

from server.player.test_helpers import card, make_snapshot
from server.protocol import (
    BidEventSnapshot,
    BottomExchangeEventSnapshot,
    CompletedTrickSnapshot,
    StirDeclarationEventSnapshot,
    TrickSlotSnapshot,
    TrickSnapshot,
)
from server.training.observation import (
    PublicHistoryRecorder,
    build_observation,
    card_tokens,
)
from server.training.tokens import (
    ActionQueryFieldToken,
    CardToken,
    RoundEventFieldToken,
    RoundFieldToken,
    TrickResultFieldToken,
)


def test_build_observation_preserves_duplicate_hand_card_ids() -> None:
    first = card("hearts", "A", 1)
    second = card("hearts", "A", 2)
    snapshot = make_snapshot(player_hand=[first, second])

    observation = build_observation(
        player_index=0,
        snapshot=snapshot,
        history=(),
    )

    assert observation.hand_card_ids == (first.id, second.id)
    hand_cards = [
        token
        for token in card_tokens(observation)
        if token.segment == "self_hand"
    ]
    assert [token.card_id for token in hand_cards] == [
        first.id,
        second.id,
    ]
    assert [token.card_order for token in hand_cards] == [0, 1]


def test_build_observation_keeps_card_face_separate_from_context() -> (
    None
):
    test_card = card("spades", "A", 1)
    snapshot = make_snapshot(player_hand=[test_card])

    observation = build_observation(
        player_index=0,
        snapshot=snapshot,
        history=(),
    )

    card_token_values = [
        token
        for token in observation.tokens
        if isinstance(token, CardToken)
    ]
    assert card_token_values
    token = card_token_values[0]
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


def test_build_observation_uses_snapshot_visible_bottom_cards() -> None:
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
        isinstance(token, CardToken) and token.card_id == bottom.id
        for token in hidden_observation.tokens
    )
    bottom_tokens = [
        token
        for token in visible_observation.tokens
        if isinstance(token, CardToken)
        and token.segment == "visible_bottom"
    ]
    assert [token.card_id for token in bottom_tokens] == [bottom.id]
    assert bottom_tokens[0].card_order == 0


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

    play_cards = [
        token
        for token in card_tokens(observation)
        if token.segment == "play_record"
    ]
    assert [token.role for token in play_cards] == [
        "partner",
        "partner",
    ]
    assert [token.trick_age for token in play_cards] == [1, 1]
    assert [token.trick_state for token in play_cards] == [
        "completed",
        "completed",
    ]
    assert TrickResultFieldToken("winner", "partner", 1) in (
        observation.tokens
    )
    assert TrickResultFieldToken("points", 35, 1) in observation.tokens


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

    play_cards = [
        token
        for token in card_tokens(observation)
        if token.segment == "play_record"
    ]
    assert len(play_cards) == 1
    assert play_cards[0].role == "left_enemy"
    assert play_cards[0].trick_age == 0
    assert play_cards[0].trick_state == "open"
    assert play_cards[0].play_order == 0
    assert (
        ActionQueryFieldToken("current_trick_width", 1)
        in observation.tokens
    )
    assert ActionQueryFieldToken("action_play_order", 3) in (
        observation.tokens
    )


def test_round_events_include_event_age_and_revealed_cards() -> None:
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
    event_cards = [
        token
        for token in card_tokens(observation)
        if token.segment == "round_event"
    ]
    assert [token.card_id for token in event_cards] == [revealed.id]
    assert event_cards[0].event_age == 1
    assert event_cards[0].role == "right_enemy"


def test_stir_and_own_exchange_history_are_observed() -> None:
    stir_card = card("spades", "2", 1)
    picked = card("diamonds", "3", 1)
    discarded = card("clubs", "4", 1)
    resulting = card("hearts", "5", 1)
    stir_event = StirDeclarationEventSnapshot(
        player=2,
        kind="stir",
        cards=[stir_card],
        new_suit=stir_card.suit,
        priority=203,
    )
    exchange_event = BottomExchangeEventSnapshot(
        player=0,
        trigger="stir",
        stir_event_index=0,
        picked_up_bottom_cards=[picked],
        discarded_bottom_cards=[discarded],
        resulting_bottom_cards=[resulting],
    )

    observation = build_observation(
        player_index=0,
        snapshot=make_snapshot(
            stir_events=[stir_event],
            own_bottom_exchange_events=[exchange_event],
        ),
        history=(),
    )

    assert RoundEventFieldToken("stir_kind", "stir", 1) in (
        observation.tokens
    )
    assert RoundEventFieldToken("trigger", "stir", 1) in (
        observation.tokens
    )
    assert RoundEventFieldToken("stir_event_age", 1, 1) in (
        observation.tokens
    )
    stir_cards = [
        token
        for token in card_tokens(observation)
        if token.segment == "stir_event"
    ]
    own_exchange_cards = [
        token
        for token in card_tokens(observation)
        if token.segment
        in (
            "own_exchange_pickup",
            "own_exchange_discard",
            "own_exchange_resulting_bottom",
        )
    ]
    assert [token.card_id for token in stir_cards] == [stir_card.id]
    assert [token.card_id for token in own_exchange_cards] == [
        picked.id,
        discarded.id,
        resulting.id,
    ]
