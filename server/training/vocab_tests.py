"""Tests for explicit training vocab component ids."""

from __future__ import annotations

from server.training.tokens import (
    ActionQueryFieldToken,
    RoundFieldToken,
    TrickResultFieldToken,
)
from server.training.vocab import component_ids
from server.training.vocab_schema import VOCAB_SCHEMA


def test_numeric_field_values_do_not_use_value_embedding() -> None:
    assert (
        component_ids(RoundFieldToken("current_score", 65)).value
        == VOCAB_SCHEMA.none_id
    )
    assert (
        component_ids(TrickResultFieldToken("points", 35, 1)).value
        == VOCAB_SCHEMA.none_id
    )
    assert (
        component_ids(ActionQueryFieldToken("max_select", 4)).value
        == VOCAB_SCHEMA.none_id
    )


def test_categorical_field_values_keep_value_embedding() -> None:
    assert (
        component_ids(RoundFieldToken("phase", "PLAYING")).value
        != VOCAB_SCHEMA.none_id
    )
    assert (
        component_ids(ActionQueryFieldToken("kind", "lead_play")).value
        != VOCAB_SCHEMA.none_id
    )


def test_categorical_integer_values_keep_value_embedding() -> None:
    assert component_ids(RoundFieldToken("dealer_team", 1)).value != (
        VOCAB_SCHEMA.none_id
    )
