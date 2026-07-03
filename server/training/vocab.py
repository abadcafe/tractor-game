"""Explicit component vocabularies for training observation tokens."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from server.rules.card_faces import CardColor
from server.rules.cards import Rank, Suit
from server.training.feature_schema import (
    MAX_EVENT_AGE,
    MAX_FACE_COUNT,
    MAX_PLAY_ORDER,
    MAX_PLAY_WIDTH,
    MAX_TRICK_AGE,
    is_categorical_int_field_key,
    is_numeric_field_key,
)
from server.training.tokens import (
    FaceCountToken,
    GlobalFieldToken,
    ObservationSegment,
    ObservationToken,
    RelativeRole,
    RoundEventFieldToken,
    RoundFieldToken,
    TokenScalar,
    TrickRecordState,
    TrickResultFieldToken,
)

OBS_PAD_ID: int = 0
NONE_ID: int = 1
INT_BASE_ID: int = 2
MAX_SCALAR_INT: int = 256
INT_OVERFLOW_ID: int = INT_BASE_ID + MAX_SCALAR_INT + 1
VALUE_STRING_BASE_ID: int = INT_OVERFLOW_ID + 1

TOKEN_TYPE_VOCAB_SIZE: int = 8
SEGMENT_VOCAB_SIZE: int = 14
FIELD_VOCAB_SIZE: int = 72
SUIT_VOCAB_SIZE: int = 7
RANK_VOCAB_SIZE: int = 17
POINTS_VOCAB_SIZE: int = 5
COLOR_VOCAB_SIZE: int = 5
ROLE_VOCAB_SIZE: int = 6
TRICK_AGE_VOCAB_SIZE: int = MAX_TRICK_AGE + 3
TRICK_STATE_VOCAB_SIZE: int = 4
PLAY_ORDER_VOCAB_SIZE: int = MAX_PLAY_ORDER + 3
COUNT_VOCAB_SIZE: int = MAX_FACE_COUNT + 3
PLAY_WIDTH_VOCAB_SIZE: int = MAX_PLAY_WIDTH + 3
EVENT_AGE_VOCAB_SIZE: int = MAX_EVENT_AGE + 3

TOKEN_TYPE_FACE_COUNT_ID: int = 1
TOKEN_TYPE_GLOBAL_FIELD_ID: int = 2
TOKEN_TYPE_ROUND_FIELD_ID: int = 3
TOKEN_TYPE_ROUND_EVENT_FIELD_ID: int = 4
TOKEN_TYPE_TRICK_RESULT_FIELD_ID: int = 5
TOKEN_TYPE_ACTION_QUERY_FIELD_ID: int = 6

SEGMENT_ACTION_QUERY_ID: int = 13


@dataclass(frozen=True, slots=True)
class TokenComponentIds:
    """Embedding component ids for one observation token."""

    token_type: int
    segment: int
    field: int
    value: int
    suit: int
    rank: int
    points: int
    color: int
    role: int
    trick_age: int
    trick_state: int
    play_order: int
    count: int
    play_width: int
    event_age: int


PAD_COMPONENT_IDS = TokenComponentIds(
    token_type=OBS_PAD_ID,
    segment=OBS_PAD_ID,
    field=OBS_PAD_ID,
    value=OBS_PAD_ID,
    suit=OBS_PAD_ID,
    rank=OBS_PAD_ID,
    points=OBS_PAD_ID,
    color=OBS_PAD_ID,
    role=OBS_PAD_ID,
    trick_age=OBS_PAD_ID,
    trick_state=OBS_PAD_ID,
    play_order=OBS_PAD_ID,
    count=OBS_PAD_ID,
    play_width=OBS_PAD_ID,
    event_age=OBS_PAD_ID,
)

_SEGMENT_IDS: MappingProxyType[ObservationSegment, int] = (
    MappingProxyType(
        {
            "global_context": 1,
            "round_context": 2,
            "round_event": 3,
            "stir_event": 4,
            "self_hand": 5,
            "visible_bottom": 6,
            "own_exchange_pickup": 7,
            "own_exchange_discard": 8,
            "play_record": 9,
            "failed_throw_attempted": 10,
            "failed_throw_forced": 11,
            "trick_result": 12,
            "action_query": SEGMENT_ACTION_QUERY_ID,
        }
    )
)
_ROLE_IDS: MappingProxyType[RelativeRole, int] = MappingProxyType(
    {
        "self": 2,
        "partner": 3,
        "left_enemy": 4,
        "right_enemy": 5,
    }
)
_TRICK_STATE_IDS: MappingProxyType[TrickRecordState, int] = (
    MappingProxyType({"open": 2, "completed": 3})
)
_COLOR_IDS: MappingProxyType[CardColor, int] = MappingProxyType(
    {"red": 2, "black": 3, "none": 4}
)
_SUIT_IDS: MappingProxyType[Suit, int] = MappingProxyType(
    {
        Suit.HEARTS: 2,
        Suit.SPADES: 3,
        Suit.DIAMONDS: 4,
        Suit.CLUBS: 5,
        Suit.JOKER: 6,
    }
)
_RANK_IDS: MappingProxyType[Rank, int] = MappingProxyType(
    {
        Rank.TWO: 2,
        Rank.THREE: 3,
        Rank.FOUR: 4,
        Rank.FIVE: 5,
        Rank.SIX: 6,
        Rank.SEVEN: 7,
        Rank.EIGHT: 8,
        Rank.NINE: 9,
        Rank.TEN: 10,
        Rank.JACK: 11,
        Rank.QUEEN: 12,
        Rank.KING: 13,
        Rank.ACE: 14,
        Rank.SMALL_JOKER: 15,
        Rank.BIG_JOKER: 16,
    }
)
_POINT_IDS: MappingProxyType[int, int] = MappingProxyType(
    {0: 2, 5: 3, 10: 4}
)
_FIELD_IDS: MappingProxyType[str, int] = MappingProxyType(
    {
        "global:team_layout": 1,
        "global:left_player_role": 2,
        "global:right_player_role": 3,
        "global:partner_role": 4,
        "global:deck_count": 5,
        "global:player_count": 6,
        "global:bottom_card_count": 7,
        "global:required_level": 8,
        "global:final_target": 9,
        "global:rules_version": 10,
        "round:phase": 11,
        "round:awaiting_action": 12,
        "round:dealer_role": 13,
        "round:dealer_team": 14,
        "round:self_team_is_declarer": 15,
        "round:enemy_team_is_declarer": 16,
        "round:self_team_level": 17,
        "round:enemy_team_level": 18,
        "round:self_team_required_level": 19,
        "round:self_team_distance_to_required_level": 20,
        "round:enemy_team_distance_to_required_level": 21,
        "round:trump_suit": 22,
        "round:level_rank": 23,
        "round:level_card_revealer_role": 24,
        "round:current_score": 25,
        "round:remaining_cards_self": 26,
        "round:remaining_cards_partner": 27,
        "round:remaining_cards_left_enemy": 28,
        "round:remaining_cards_right_enemy": 29,
        "round:winning_team": 30,
        "round_event:event_kind": 31,
        "round_event:actor": 32,
        "round_event:bid_kind": 33,
        "round_event:stir_kind": 34,
        "round_event:suit": 35,
        "round_event:joker_type": 36,
        "round_event:count": 37,
        "round_event:priority": 38,
        "round_event:trigger": 39,
        "trick_result:winner": 40,
        "trick_result:points": 41,
        "action_query:kind": 42,
        "action_query:pass_allowed": 43,
        "action_query:min_select": 44,
        "action_query:max_select": 45,
        "action_query:exact_select": 46,
        "action_query:action_play_order": 47,
        "action_query:current_trick_width": 48,
        "action_query:lead_actor": 49,
        "action_query:discard_count": 50,
        "action_query:trump_suit": 51,
        "action_query:level_rank": 52,
        "action_query:current_best_bid_role": 53,
        "round:enemy_team_required_level": 54,
    }
)
_STRING_VALUE_IDS: MappingProxyType[str, int] = MappingProxyType(
    {
        "fixed_partner_opposite": VALUE_STRING_BASE_ID,
        "left_enemy": VALUE_STRING_BASE_ID + 1,
        "right_enemy": VALUE_STRING_BASE_ID + 2,
        "partner": VALUE_STRING_BASE_ID + 3,
        "self": VALUE_STRING_BASE_ID + 4,
        "required-levels": VALUE_STRING_BASE_ID + 5,
        "WIN": VALUE_STRING_BASE_ID + 6,
        "DEAL_BID": VALUE_STRING_BASE_ID + 7,
        "STIRRING": VALUE_STRING_BASE_ID + 8,
        "PLAYING": VALUE_STRING_BASE_ID + 9,
        "SCORING": VALUE_STRING_BASE_ID + 10,
        "WAITING": VALUE_STRING_BASE_ID + 11,
        "bid": VALUE_STRING_BASE_ID + 12,
        "stir": VALUE_STRING_BASE_ID + 13,
        "discard": VALUE_STRING_BASE_ID + 14,
        "play": VALUE_STRING_BASE_ID + 15,
        "next_round": VALUE_STRING_BASE_ID + 16,
        "hearts": VALUE_STRING_BASE_ID + 17,
        "spades": VALUE_STRING_BASE_ID + 18,
        "diamonds": VALUE_STRING_BASE_ID + 19,
        "clubs": VALUE_STRING_BASE_ID + 20,
        "joker": VALUE_STRING_BASE_ID + 21,
        "2": VALUE_STRING_BASE_ID + 22,
        "3": VALUE_STRING_BASE_ID + 23,
        "4": VALUE_STRING_BASE_ID + 24,
        "5": VALUE_STRING_BASE_ID + 25,
        "6": VALUE_STRING_BASE_ID + 26,
        "7": VALUE_STRING_BASE_ID + 27,
        "8": VALUE_STRING_BASE_ID + 28,
        "9": VALUE_STRING_BASE_ID + 29,
        "10": VALUE_STRING_BASE_ID + 30,
        "J": VALUE_STRING_BASE_ID + 31,
        "Q": VALUE_STRING_BASE_ID + 32,
        "K": VALUE_STRING_BASE_ID + 33,
        "A": VALUE_STRING_BASE_ID + 34,
        "SJ": VALUE_STRING_BASE_ID + 35,
        "BJ": VALUE_STRING_BASE_ID + 36,
        "trump_rank": VALUE_STRING_BASE_ID + 37,
        "big": VALUE_STRING_BASE_ID + 38,
        "small": VALUE_STRING_BASE_ID + 39,
        "lead_play": VALUE_STRING_BASE_ID + 40,
        "follow_play": VALUE_STRING_BASE_ID + 41,
        "pass": VALUE_STRING_BASE_ID + 42,
        "own_exchange": VALUE_STRING_BASE_ID + 43,
        "initial": VALUE_STRING_BASE_ID + 44,
    }
)
FALSE_VALUE_ID: int = VALUE_STRING_BASE_ID + len(_STRING_VALUE_IDS)
TRUE_VALUE_ID: int = FALSE_VALUE_ID + 1
VALUE_VOCAB_SIZE: int = TRUE_VALUE_ID + 1


def component_ids(token: ObservationToken) -> TokenComponentIds:
    """Return explicit embedding component ids for one token."""
    if isinstance(token, FaceCountToken):
        return _face_count_component_ids(token)
    if isinstance(token, GlobalFieldToken):
        return _field_component_ids(
            token_type=TOKEN_TYPE_GLOBAL_FIELD_ID,
            segment="global_context",
            field_key=f"global:{token.field}",
            value=token.value,
        )
    if isinstance(token, RoundFieldToken):
        return _field_component_ids(
            token_type=TOKEN_TYPE_ROUND_FIELD_ID,
            segment="round_context",
            field_key=f"round:{token.field}",
            value=token.value,
        )
    if isinstance(token, RoundEventFieldToken):
        return _field_component_ids(
            token_type=TOKEN_TYPE_ROUND_EVENT_FIELD_ID,
            segment="round_event",
            field_key=f"round_event:{token.field}",
            value=token.value,
            event_age=token.event_age,
        )
    if isinstance(token, TrickResultFieldToken):
        return _field_component_ids(
            token_type=TOKEN_TYPE_TRICK_RESULT_FIELD_ID,
            segment="trick_result",
            field_key=f"trick_result:{token.field}",
            value=token.value,
            trick_age=token.trick_age,
        )
    return _field_component_ids(
        token_type=TOKEN_TYPE_ACTION_QUERY_FIELD_ID,
        segment="action_query",
        field_key=f"action_query:{token.field}",
        value=token.value,
    )


def _face_count_component_ids(
    token: FaceCountToken,
) -> TokenComponentIds:
    return TokenComponentIds(
        token_type=TOKEN_TYPE_FACE_COUNT_ID,
        segment=_SEGMENT_IDS[token.segment],
        field=NONE_ID,
        value=NONE_ID,
        suit=_SUIT_IDS[token.suit],
        rank=_RANK_IDS[token.rank],
        points=_POINT_IDS[token.points],
        color=_COLOR_IDS[token.color],
        role=_role_id(token.role),
        trick_age=_bounded_optional_id(token.trick_age, MAX_TRICK_AGE),
        trick_state=_trick_state_id(token.trick_state),
        play_order=_bounded_optional_id(
            token.play_order, MAX_PLAY_ORDER
        ),
        count=_bounded_optional_id(token.count, MAX_FACE_COUNT),
        play_width=_bounded_optional_id(
            token.play_width, MAX_PLAY_WIDTH
        ),
        event_age=_bounded_optional_id(token.event_age, MAX_EVENT_AGE),
    )


def _field_component_ids(
    *,
    token_type: int,
    segment: ObservationSegment,
    field_key: str,
    value: TokenScalar,
    trick_age: int | None = None,
    event_age: int | None = None,
) -> TokenComponentIds:
    return TokenComponentIds(
        token_type=token_type,
        segment=_SEGMENT_IDS[segment],
        field=_FIELD_IDS[field_key],
        value=_field_value_id(field_key, value),
        suit=NONE_ID,
        rank=NONE_ID,
        points=NONE_ID,
        color=NONE_ID,
        role=NONE_ID,
        trick_age=_bounded_optional_id(trick_age, MAX_TRICK_AGE),
        trick_state=NONE_ID,
        play_order=NONE_ID,
        count=NONE_ID,
        play_width=NONE_ID,
        event_age=_bounded_optional_id(event_age, MAX_EVENT_AGE),
    )


def _field_value_id(field_key: str, value: TokenScalar) -> int:
    if is_numeric_field_key(field_key):
        return NONE_ID
    return _categorical_value_id(field_key, value)


def _categorical_value_id(field_key: str, value: TokenScalar) -> int:
    if value is None:
        return NONE_ID
    if isinstance(value, bool):
        return TRUE_VALUE_ID if value else FALSE_VALUE_ID
    if type(value) is int:
        assert is_categorical_int_field_key(field_key)
        if value < 0:
            return INT_OVERFLOW_ID
        if value > MAX_SCALAR_INT:
            return INT_OVERFLOW_ID
        return INT_BASE_ID + value
    assert isinstance(value, str)
    return _STRING_VALUE_IDS[value]


def _role_id(role: RelativeRole | None) -> int:
    if role is None:
        return NONE_ID
    return _ROLE_IDS[role]


def _trick_state_id(state: TrickRecordState | None) -> int:
    if state is None:
        return NONE_ID
    return _TRICK_STATE_IDS[state]


def _bounded_optional_id(value: int | None, max_value: int) -> int:
    if value is None:
        return NONE_ID
    if value < 0:
        return max_value + 2
    if value > max_value:
        return max_value + 2
    return value + 2
