"""Required-level progress rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from server.rules.cards import Rank
from server.rules.level_progression import LEVELS


class TerminalProgress(str, Enum):
    """Virtual progress targets beyond real card ranks."""

    WIN = "WIN"


type ProgressTarget = Rank | TerminalProgress


MANDATORY_LEVELS: tuple[Rank, ...] = (Rank.ACE,)


@dataclass(frozen=True, slots=True)
class TeamAdvance:
    """One team's level after applying required-level progress rules."""

    level: Rank
    won_game: bool


def stage_target(level: Rank) -> ProgressTarget:
    """Return the next mandatory target from a team's current level."""
    level_index = _rank_index(level)
    for required_level in MANDATORY_LEVELS:
        if _rank_index(required_level) > level_index:
            return required_level
    return TerminalProgress.WIN


def distance_to_target(level: Rank, target: ProgressTarget) -> int:
    """Return non-negative level steps from level to target."""
    return max(0, _target_index(target) - _rank_index(level))


def advance_team_progress(
    *,
    level: Rank,
    raw_gain: int,
    was_declarer: bool,
) -> TeamAdvance:
    """Advance one team through mandatory levels for one round."""
    assert raw_gain >= 0
    if raw_gain == 0:
        return TeamAdvance(level=level, won_game=False)

    target = stage_target(level)
    if target == TerminalProgress.WIN:
        return TeamAdvance(level=level, won_game=was_declarer)

    next_index = min(
        _rank_index(level) + raw_gain,
        _target_index(target),
    )
    return TeamAdvance(level=LEVELS[next_index], won_game=False)


def progress_target_value(target: ProgressTarget) -> str:
    """Return protocol/training scalar value for a progress target."""
    return target.value


def progress_target_index(target: ProgressTarget) -> int:
    """Return the ordered progress index for rank or WIN targets."""
    return _target_index(target)


def _rank_index(level: Rank) -> int:
    return LEVELS.index(level)


def _target_index(target: ProgressTarget) -> int:
    if target == TerminalProgress.WIN:
        return len(LEVELS)
    return _rank_index(target)


def _assert_mandatory_levels() -> None:
    assert MANDATORY_LEVELS
    assert MANDATORY_LEVELS[-1] == Rank.ACE
    previous_index = -1
    for level in MANDATORY_LEVELS:
        level_index = _rank_index(level)
        assert level_index > previous_index
        previous_index = level_index


_assert_mandatory_levels()
