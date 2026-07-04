"""Explicit component vocabularies for training observation tokens."""

from __future__ import annotations

from types import MappingProxyType

from server.rules.card_faces import CardColor
from server.rules.cards import Rank, Suit
from server.training import vocab_schema as _schema
from server.training.feature_schema import (
    MAX_EVENT_AGE,
    MAX_FACE_COUNT,
    MAX_PLAY_ORDER,
    MAX_PLAY_WIDTH,
    MAX_TRICK_AGE,
    is_categorical_int_field_key,
    is_numeric_field_key,
)
from server.training.token_context import (
    ObservationSegment,
    RelativeRole,
    TokenScalar,
    TrickRecordState,
)
from server.training.tokens import (
    FaceCountToken,
    GlobalFieldToken,
    ObservationToken,
    RoundEventFieldToken,
    RoundFieldToken,
    TrickResultFieldToken,
)

__all__ = ("component_ids",)

_VOCAB_SCHEMA = _schema.VOCAB_SCHEMA
_STRING_VALUE_BASE = _VOCAB_SCHEMA.value_string_base_id

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
            "action_query": _VOCAB_SCHEMA.segment_action_query_id,
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
        "fixed_partner_opposite": _STRING_VALUE_BASE,
        "left_enemy": _STRING_VALUE_BASE + 1,
        "right_enemy": _STRING_VALUE_BASE + 2,
        "partner": _STRING_VALUE_BASE + 3,
        "self": _STRING_VALUE_BASE + 4,
        "rules-required-progress": _STRING_VALUE_BASE + 5,
        "WIN": _STRING_VALUE_BASE + 6,
        "DEAL_BID": _STRING_VALUE_BASE + 7,
        "STIRRING": _STRING_VALUE_BASE + 8,
        "PLAYING": _STRING_VALUE_BASE + 9,
        "SCORING": _STRING_VALUE_BASE + 10,
        "WAITING": _STRING_VALUE_BASE + 11,
        "bid": _STRING_VALUE_BASE + 12,
        "stir": _STRING_VALUE_BASE + 13,
        "discard": _STRING_VALUE_BASE + 14,
        "play": _STRING_VALUE_BASE + 15,
        "next_round": _STRING_VALUE_BASE + 16,
        "hearts": _STRING_VALUE_BASE + 17,
        "spades": _STRING_VALUE_BASE + 18,
        "diamonds": _STRING_VALUE_BASE + 19,
        "clubs": _STRING_VALUE_BASE + 20,
        "joker": _STRING_VALUE_BASE + 21,
        "2": _STRING_VALUE_BASE + 22,
        "3": _STRING_VALUE_BASE + 23,
        "4": _STRING_VALUE_BASE + 24,
        "5": _STRING_VALUE_BASE + 25,
        "6": _STRING_VALUE_BASE + 26,
        "7": _STRING_VALUE_BASE + 27,
        "8": _STRING_VALUE_BASE + 28,
        "9": _STRING_VALUE_BASE + 29,
        "10": _STRING_VALUE_BASE + 30,
        "J": _STRING_VALUE_BASE + 31,
        "Q": _STRING_VALUE_BASE + 32,
        "K": _STRING_VALUE_BASE + 33,
        "A": _STRING_VALUE_BASE + 34,
        "SJ": _STRING_VALUE_BASE + 35,
        "BJ": _STRING_VALUE_BASE + 36,
        "trump_rank": _STRING_VALUE_BASE + 37,
        "big": _STRING_VALUE_BASE + 38,
        "small": _STRING_VALUE_BASE + 39,
        "lead_play": _STRING_VALUE_BASE + 40,
        "follow_play": _STRING_VALUE_BASE + 41,
        "pass": _STRING_VALUE_BASE + 42,
        "own_exchange": _STRING_VALUE_BASE + 43,
        "initial": _STRING_VALUE_BASE + 44,
    }
)
assert len(_STRING_VALUE_IDS) == (
    _VOCAB_SCHEMA.false_value_id - _VOCAB_SCHEMA.value_string_base_id
)


def component_ids(token: ObservationToken) -> _schema.TokenComponentIds:
    """Return explicit embedding component ids for one token."""
    if isinstance(token, FaceCountToken):
        return _face_count_component_ids(token)
    if isinstance(token, GlobalFieldToken):
        return _field_component_ids(
            token_type=_VOCAB_SCHEMA.token_type_global_field_id,
            segment="global_context",
            field_key=f"global:{token.field}",
            value=token.value,
        )
    if isinstance(token, RoundFieldToken):
        return _field_component_ids(
            token_type=_VOCAB_SCHEMA.token_type_round_field_id,
            segment="round_context",
            field_key=f"round:{token.field}",
            value=token.value,
        )
    if isinstance(token, RoundEventFieldToken):
        return _field_component_ids(
            token_type=_VOCAB_SCHEMA.token_type_round_event_field_id,
            segment="round_event",
            field_key=f"round_event:{token.field}",
            value=token.value,
            event_age=token.event_age,
        )
    if isinstance(token, TrickResultFieldToken):
        return _field_component_ids(
            token_type=_VOCAB_SCHEMA.token_type_trick_result_field_id,
            segment="trick_result",
            field_key=f"trick_result:{token.field}",
            value=token.value,
            trick_age=token.trick_age,
        )
    return _field_component_ids(
        token_type=_VOCAB_SCHEMA.token_type_action_query_field_id,
        segment="action_query",
        field_key=f"action_query:{token.field}",
        value=token.value,
    )


def _face_count_component_ids(
    token: FaceCountToken,
) -> _schema.TokenComponentIds:
    return _schema.TokenComponentIds(
        token_type=_VOCAB_SCHEMA.token_type_face_count_id,
        segment=_SEGMENT_IDS[token.segment],
        field=_VOCAB_SCHEMA.none_id,
        value=_VOCAB_SCHEMA.none_id,
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
) -> _schema.TokenComponentIds:
    return _schema.TokenComponentIds(
        token_type=token_type,
        segment=_SEGMENT_IDS[segment],
        field=_FIELD_IDS[field_key],
        value=_field_value_id(field_key, value),
        suit=_VOCAB_SCHEMA.none_id,
        rank=_VOCAB_SCHEMA.none_id,
        points=_VOCAB_SCHEMA.none_id,
        color=_VOCAB_SCHEMA.none_id,
        role=_VOCAB_SCHEMA.none_id,
        trick_age=_bounded_optional_id(trick_age, MAX_TRICK_AGE),
        trick_state=_VOCAB_SCHEMA.none_id,
        play_order=_VOCAB_SCHEMA.none_id,
        count=_VOCAB_SCHEMA.none_id,
        play_width=_VOCAB_SCHEMA.none_id,
        event_age=_bounded_optional_id(event_age, MAX_EVENT_AGE),
    )


def _field_value_id(field_key: str, value: TokenScalar) -> int:
    if is_numeric_field_key(field_key):
        return _VOCAB_SCHEMA.none_id
    return _categorical_value_id(field_key, value)


def _categorical_value_id(field_key: str, value: TokenScalar) -> int:
    if value is None:
        return _VOCAB_SCHEMA.none_id
    if isinstance(value, bool):
        return (
            _VOCAB_SCHEMA.true_value_id
            if value
            else _VOCAB_SCHEMA.false_value_id
        )
    if type(value) is int:
        assert is_categorical_int_field_key(field_key)
        if value < 0:
            return _VOCAB_SCHEMA.int_overflow_id
        if value > _VOCAB_SCHEMA.max_scalar_int:
            return _VOCAB_SCHEMA.int_overflow_id
        return _VOCAB_SCHEMA.int_base_id + value
    assert isinstance(value, str)
    return _STRING_VALUE_IDS[value]


def _role_id(role: RelativeRole | None) -> int:
    if role is None:
        return _VOCAB_SCHEMA.none_id
    return _ROLE_IDS[role]


def _trick_state_id(state: TrickRecordState | None) -> int:
    if state is None:
        return _VOCAB_SCHEMA.none_id
    return _TRICK_STATE_IDS[state]


def _bounded_optional_id(value: int | None, max_value: int) -> int:
    if value is None:
        return _VOCAB_SCHEMA.none_id
    if value < 0:
        return max_value + 2
    if value > max_value:
        return max_value + 2
    return value + 2
