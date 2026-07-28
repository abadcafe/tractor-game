"""Stable typed coordinates for observation structure."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class StructureAxis(IntEnum):
    """Fixed tensor order of the three independent structure axes."""

    ROUND_EVENT = 0
    TRICK = 1
    PLAY_POSITION = 2


STRUCTURE_AXIS_COUNT = len(StructureAxis)


@dataclass(frozen=True, slots=True)
class RoundEventOrdinal:
    """One-based ordinal in the visible round-event timeline."""

    value: int

    def __post_init__(self) -> None:
        assert self.value > 0


@dataclass(frozen=True, slots=True)
class TrickRecency:
    """Zero-based trick recency, where zero is the open trick."""

    value: int

    def __post_init__(self) -> None:
        assert self.value >= 0


__all__ = (
    "RoundEventOrdinal",
    "STRUCTURE_AXIS_COUNT",
    "StructureAxis",
    "TrickRecency",
)
