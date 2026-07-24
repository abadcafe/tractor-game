"""Mandatory-level game progression rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .cards import Rank

__all__ = [
    "ProgressTarget",
    "TeamAdvance",
    "TerminalProgress",
    "advance_team",
    "distance_to_target",
    "progress_target_index",
    "progress_target_value",
    "stage_target",
]

_LEVELS: tuple[Rank, ...] = (
    Rank.TWO,
    Rank.THREE,
    Rank.FOUR,
    Rank.FIVE,
    Rank.SIX,
    Rank.SEVEN,
    Rank.EIGHT,
    Rank.NINE,
    Rank.TEN,
    Rank.JACK,
    Rank.QUEEN,
    Rank.KING,
    Rank.ACE,
)


class TerminalProgress(str, Enum):
    """Virtual target beyond physical card ranks."""

    WIN = "WIN"


type ProgressTarget = Rank | TerminalProgress


@dataclass(frozen=True, slots=True)
class TeamAdvance:
    """One team's progress after a round."""

    level: Rank
    won_game: bool


def stage_target(
    level: Rank,
    mandatory_levels: tuple[Rank, ...],
) -> ProgressTarget:
    """Return the next mandatory target."""
    current = _LEVELS.index(level)
    for required in mandatory_levels:
        if _LEVELS.index(required) > current:
            return required
    return TerminalProgress.WIN


def distance_to_target(
    level: Rank,
    target: ProgressTarget,
) -> int:
    """Return non-negative progress steps to a target."""
    return max(0, _target_index(target) - _LEVELS.index(level))


def advance_team(
    *,
    level: Rank,
    raw_gain: int,
    was_declarer: bool,
    mandatory_levels: tuple[Rank, ...],
) -> TeamAdvance:
    """Apply a raw round gain without crossing mandatory targets."""
    assert raw_gain >= 0
    if raw_gain == 0:
        return TeamAdvance(level=level, won_game=False)
    target = stage_target(level, mandatory_levels)
    if target == TerminalProgress.WIN:
        return TeamAdvance(level=level, won_game=was_declarer)
    next_index = min(
        _LEVELS.index(level) + raw_gain,
        _target_index(target),
    )
    return TeamAdvance(
        level=_LEVELS[next_index],
        won_game=False,
    )


def progress_target_value(target: ProgressTarget) -> str:
    """Return the stable serialized target value."""
    return target.value


def progress_target_index(target: ProgressTarget) -> int:
    """Return the ordered target index."""
    return _target_index(target)


def _target_index(target: ProgressTarget) -> int:
    if target == TerminalProgress.WIN:
        return len(_LEVELS)
    return _LEVELS.index(target)
