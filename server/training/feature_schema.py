"""Observation feature schema for categorical and numeric inputs."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

MAX_TRICK_AGE: int = 33
MAX_PLAY_ORDER: int = 3
MAX_CARD_ORDER: int = 32
MAX_PLAY_WIDTH: int = 33
MAX_EVENT_AGE: int = 32


@dataclass(frozen=True, slots=True)
class NumericFeatureSpec:
    """One normalized numeric feature lane for an observation token."""

    key: str
    index: int
    scale: float


def _numeric_specs() -> tuple[NumericFeatureSpec, ...]:
    items: tuple[tuple[str, float], ...] = (
        ("card:points", 10.0),
        ("card:trick_age", float(MAX_TRICK_AGE)),
        ("card:play_order", float(MAX_PLAY_ORDER)),
        ("card:card_order", float(MAX_CARD_ORDER)),
        ("card:play_width", float(MAX_PLAY_WIDTH)),
        ("card:event_age", float(MAX_EVENT_AGE)),
        ("global:deck_count", 4.0),
        ("global:player_count", 4.0),
        ("global:bottom_card_count", float(MAX_CARD_ORDER)),
        ("round:self_team_distance_to_required_level", 13.0),
        ("round:enemy_team_distance_to_required_level", 13.0),
        ("round:current_score", 200.0),
        ("round:remaining_cards_self", float(MAX_PLAY_WIDTH)),
        ("round:remaining_cards_partner", float(MAX_PLAY_WIDTH)),
        ("round:remaining_cards_left_enemy", float(MAX_PLAY_WIDTH)),
        ("round:remaining_cards_right_enemy", float(MAX_PLAY_WIDTH)),
        ("round_event:count", 4.0),
        ("round_event:priority", 205.0),
        ("round_event:stir_event_age", float(MAX_EVENT_AGE)),
        ("trick_result:points", 200.0),
        ("action_query:min_select", float(MAX_PLAY_WIDTH)),
        ("action_query:max_select", float(MAX_PLAY_WIDTH)),
        ("action_query:exact_select", float(MAX_PLAY_WIDTH)),
        ("action_query:action_play_order", float(MAX_PLAY_ORDER)),
        ("action_query:current_trick_width", float(MAX_PLAY_WIDTH)),
        ("action_query:discard_count", float(MAX_PLAY_WIDTH)),
    )
    return tuple(
        NumericFeatureSpec(key=key, index=index, scale=scale)
        for index, (key, scale) in enumerate(items)
    )


NUMERIC_FEATURE_SPECS: tuple[NumericFeatureSpec, ...] = _numeric_specs()
NUMERIC_FEATURE_COUNT: int = len(NUMERIC_FEATURE_SPECS)

_NUMERIC_FEATURES_BY_KEY: MappingProxyType[str, NumericFeatureSpec] = (
    MappingProxyType({spec.key: spec for spec in NUMERIC_FEATURE_SPECS})
)

_CATEGORICAL_INT_FIELD_KEYS: frozenset[str] = frozenset(
    (
        "round:dealer_team",
        "round:winning_team",
    )
)


def numeric_feature_spec(key: str) -> NumericFeatureSpec | None:
    """Return the numeric feature lane for a field or card component."""
    return _NUMERIC_FEATURES_BY_KEY.get(key)


def is_numeric_field_key(key: str) -> bool:
    """Return whether a field value must use the numeric input path."""
    return key in _NUMERIC_FEATURES_BY_KEY


def is_categorical_int_field_key(key: str) -> bool:
    """Return whether an integer value is an explicit category."""
    return key in _CATEGORICAL_INT_FIELD_KEYS
