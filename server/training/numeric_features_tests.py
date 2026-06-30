"""Tests for numeric observation feature extraction."""

from __future__ import annotations

from server.player.test_helpers import card
from server.training.feature_schema import (
    NUMERIC_FEATURE_COUNT,
    numeric_feature_spec,
)
from server.training.numeric_features import numeric_feature_values
from server.training.tokens import RoundFieldToken, card_token


def test_numeric_feature_distinguishes_zero_from_missing() -> None:
    spec = numeric_feature_spec("round:current_score")
    assert spec is not None

    zero_score = numeric_feature_values(
        RoundFieldToken("current_score", 0)
    )
    missing_score = numeric_feature_values(
        RoundFieldToken("phase", "PLAYING")
    )

    assert len(zero_score.values) == NUMERIC_FEATURE_COUNT
    assert zero_score.values[spec.index] == 0.0
    assert zero_score.masks[spec.index] == 1.0
    assert missing_score.values[spec.index] == 0.0
    assert missing_score.masks[spec.index] == 0.0


def test_numeric_feature_normalizes_score_fields() -> None:
    spec = numeric_feature_spec("round:current_score")
    assert spec is not None

    features = numeric_feature_values(
        RoundFieldToken("current_score", 100)
    )

    assert features.values[spec.index] == 0.5
    assert features.masks[spec.index] == 1.0


def test_card_numeric_features_include_points_and_structure() -> None:
    point_spec = numeric_feature_spec("card:points")
    order_spec = numeric_feature_spec("card:card_order")
    assert point_spec is not None
    assert order_spec is not None

    token = card_token(
        card("hearts", "5", 1),
        segment="self_hand",
        role="self",
        card_order=16,
    )
    features = numeric_feature_values(token)

    assert features.values[point_spec.index] == 0.5
    assert features.masks[point_spec.index] == 1.0
    assert features.values[order_spec.index] == 0.5
    assert features.masks[order_spec.index] == 1.0
