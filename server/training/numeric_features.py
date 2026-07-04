"""Numeric feature extraction for observation tokens."""

from __future__ import annotations

from dataclasses import dataclass

from server.training.feature_schema import (
    NUMERIC_FEATURE_COUNT,
    NumericFeatureSpec,
    numeric_feature_spec,
)
from server.training.token_context import TokenScalar
from server.training.tokens import (
    FaceCountToken,
    GlobalFieldToken,
    ObservationToken,
    RoundEventFieldToken,
    RoundFieldToken,
    TrickResultFieldToken,
)


@dataclass(frozen=True, slots=True)
class NumericFeatureValues:
    """Numeric values and presence masks for one observation token."""

    values: tuple[float, ...]
    masks: tuple[float, ...]


PAD_NUMERIC_FEATURES = NumericFeatureValues(
    values=tuple(0.0 for _ in range(NUMERIC_FEATURE_COUNT)),
    masks=tuple(0.0 for _ in range(NUMERIC_FEATURE_COUNT)),
)


def numeric_feature_values(
    token: ObservationToken,
) -> NumericFeatureValues:
    """Return normalized numeric features for one observation token."""
    values = [0.0 for _ in range(NUMERIC_FEATURE_COUNT)]
    masks = [0.0 for _ in range(NUMERIC_FEATURE_COUNT)]
    if isinstance(token, FaceCountToken):
        _set_numeric(values, masks, "face_count:points", token.points)
        _set_numeric(values, masks, "face_count:count", token.count)
        _set_optional_numeric(
            values, masks, "face_count:trick_age", token.trick_age
        )
        _set_optional_numeric(
            values, masks, "face_count:play_order", token.play_order
        )
        _set_optional_numeric(
            values, masks, "face_count:play_width", token.play_width
        )
        _set_optional_numeric(
            values, masks, "face_count:event_age", token.event_age
        )
    elif isinstance(token, GlobalFieldToken):
        _set_scalar(values, masks, f"global:{token.field}", token.value)
    elif isinstance(token, RoundFieldToken):
        _set_scalar(values, masks, f"round:{token.field}", token.value)
    elif isinstance(token, RoundEventFieldToken):
        _set_scalar(
            values, masks, f"round_event:{token.field}", token.value
        )
    elif isinstance(token, TrickResultFieldToken):
        _set_scalar(
            values, masks, f"trick_result:{token.field}", token.value
        )
    else:
        _set_scalar(
            values, masks, f"action_query:{token.field}", token.value
        )
    return NumericFeatureValues(
        values=tuple(values),
        masks=tuple(masks),
    )


def _set_scalar(
    values: list[float],
    masks: list[float],
    key: str,
    value: TokenScalar,
) -> None:
    spec = numeric_feature_spec(key)
    if spec is None:
        return
    if value is None:
        return
    assert type(value) is int
    _set_spec_value(values, masks, spec, value)


def _set_optional_numeric(
    values: list[float],
    masks: list[float],
    key: str,
    value: int | None,
) -> None:
    if value is None:
        return
    _set_numeric(values, masks, key, value)


def _set_numeric(
    values: list[float],
    masks: list[float],
    key: str,
    value: int,
) -> None:
    spec = numeric_feature_spec(key)
    assert spec is not None
    _set_spec_value(values, masks, spec, value)


def _set_spec_value(
    values: list[float],
    masks: list[float],
    spec: NumericFeatureSpec,
    value: int,
) -> None:
    assert value >= 0
    assert spec.scale > 0.0
    values[spec.index] = float(value) / spec.scale
    masks[spec.index] = 1.0
