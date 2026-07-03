"""Tests for explicit training vocab component ids."""

from __future__ import annotations

from server.training.tokens import (
    ActionQueryFieldToken,
    RoundFieldToken,
    TrickResultFieldToken,
)
from server.training.vocab import NONE_ID, component_ids


def test_numeric_field_values_do_not_use_value_embedding() -> None:
    assert (
        component_ids(RoundFieldToken("current_score", 65)).value
        == NONE_ID
    )
    assert (
        component_ids(TrickResultFieldToken("points", 35, 1)).value
        == NONE_ID
    )
    assert (
        component_ids(ActionQueryFieldToken("max_select", 4)).value
        == NONE_ID
    )


def test_categorical_field_values_keep_value_embedding() -> None:
    assert (
        component_ids(RoundFieldToken("phase", "PLAYING")).value
        != NONE_ID
    )
    assert (
        component_ids(ActionQueryFieldToken("kind", "lead_play")).value
        != NONE_ID
    )


def test_categorical_integer_values_keep_value_embedding() -> None:
    assert component_ids(RoundFieldToken("dealer_team", 1)).value != (
        NONE_ID
    )
