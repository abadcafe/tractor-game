"""Gumbel policy-improvement transform for sampled root values."""

from __future__ import annotations

import math

_VISIT_OFFSET = 50.0
_VALUE_SCALE = 0.1


def transformed_q_values(
    *,
    means: tuple[float, ...],
    visits: tuple[int, ...],
) -> tuple[float, ...]:
    """Normalize finite Q estimates and apply visit-aware scaling."""
    assert means
    assert len(means) == len(visits)
    assert all(math.isfinite(value) for value in means)
    assert all(value > 0 for value in visits)
    minimum = min(means)
    maximum = max(means)
    value_range = maximum - minimum
    if value_range == 0.0:
        normalized = tuple(0.0 for _value in means)
    else:
        normalized = tuple(
            (value - minimum) / value_range for value in means
        )
    scale = (_VISIT_OFFSET + float(max(visits))) * _VALUE_SCALE
    return tuple(value * scale for value in normalized)


__all__ = ("transformed_q_values",)
