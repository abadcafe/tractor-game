"""Embedding schema for training observation token components."""

from __future__ import annotations

from dataclasses import dataclass

from server.training.feature_schema import (
    MAX_EVENT_AGE,
    MAX_FACE_COUNT,
    MAX_PLAY_ORDER,
    MAX_PLAY_WIDTH,
    MAX_TRICK_AGE,
)


@dataclass(frozen=True, slots=True)
class ObservationVocabSchema:
    """Stable embedding vocabulary sizes and reserved ids."""

    obs_pad_id: int
    none_id: int
    int_base_id: int
    max_scalar_int: int
    int_overflow_id: int
    value_string_base_id: int
    token_type_vocab_size: int
    segment_vocab_size: int
    field_vocab_size: int
    suit_vocab_size: int
    rank_vocab_size: int
    points_vocab_size: int
    color_vocab_size: int
    role_vocab_size: int
    trick_age_vocab_size: int
    trick_state_vocab_size: int
    play_order_vocab_size: int
    count_vocab_size: int
    play_width_vocab_size: int
    event_age_vocab_size: int
    token_type_face_count_id: int
    token_type_global_field_id: int
    token_type_round_field_id: int
    token_type_round_event_field_id: int
    token_type_trick_result_field_id: int
    token_type_action_query_field_id: int
    segment_action_query_id: int
    false_value_id: int
    true_value_id: int
    value_vocab_size: int


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


_OBS_PAD_ID: int = 0
_NONE_ID: int = 1
_INT_BASE_ID: int = 2
_MAX_SCALAR_INT: int = 256
_INT_OVERFLOW_ID: int = _INT_BASE_ID + _MAX_SCALAR_INT + 1
_VALUE_STRING_BASE_ID: int = _INT_OVERFLOW_ID + 1
_STRING_VALUE_COUNT: int = 45
_FALSE_VALUE_ID: int = _VALUE_STRING_BASE_ID + _STRING_VALUE_COUNT
_TRUE_VALUE_ID: int = _FALSE_VALUE_ID + 1

VOCAB_SCHEMA = ObservationVocabSchema(
    obs_pad_id=_OBS_PAD_ID,
    none_id=_NONE_ID,
    int_base_id=_INT_BASE_ID,
    max_scalar_int=_MAX_SCALAR_INT,
    int_overflow_id=_INT_OVERFLOW_ID,
    value_string_base_id=_VALUE_STRING_BASE_ID,
    token_type_vocab_size=8,
    segment_vocab_size=14,
    field_vocab_size=72,
    suit_vocab_size=7,
    rank_vocab_size=17,
    points_vocab_size=5,
    color_vocab_size=5,
    role_vocab_size=6,
    trick_age_vocab_size=MAX_TRICK_AGE + 3,
    trick_state_vocab_size=4,
    play_order_vocab_size=MAX_PLAY_ORDER + 3,
    count_vocab_size=MAX_FACE_COUNT + 3,
    play_width_vocab_size=MAX_PLAY_WIDTH + 3,
    event_age_vocab_size=MAX_EVENT_AGE + 3,
    token_type_face_count_id=1,
    token_type_global_field_id=2,
    token_type_round_field_id=3,
    token_type_round_event_field_id=4,
    token_type_trick_result_field_id=5,
    token_type_action_query_field_id=6,
    segment_action_query_id=13,
    false_value_id=_FALSE_VALUE_ID,
    true_value_id=_TRUE_VALUE_ID,
    value_vocab_size=_TRUE_VALUE_ID + 1,
)

PAD_COMPONENT_IDS = TokenComponentIds(
    token_type=VOCAB_SCHEMA.obs_pad_id,
    segment=VOCAB_SCHEMA.obs_pad_id,
    field=VOCAB_SCHEMA.obs_pad_id,
    value=VOCAB_SCHEMA.obs_pad_id,
    suit=VOCAB_SCHEMA.obs_pad_id,
    rank=VOCAB_SCHEMA.obs_pad_id,
    points=VOCAB_SCHEMA.obs_pad_id,
    color=VOCAB_SCHEMA.obs_pad_id,
    role=VOCAB_SCHEMA.obs_pad_id,
    trick_age=VOCAB_SCHEMA.obs_pad_id,
    trick_state=VOCAB_SCHEMA.obs_pad_id,
    play_order=VOCAB_SCHEMA.obs_pad_id,
    count=VOCAB_SCHEMA.obs_pad_id,
    play_width=VOCAB_SCHEMA.obs_pad_id,
    event_age=VOCAB_SCHEMA.obs_pad_id,
)

__all__ = (
    "PAD_COMPONENT_IDS",
    "VOCAB_SCHEMA",
    "ObservationVocabSchema",
    "TokenComponentIds",
)
