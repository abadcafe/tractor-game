"""Required-level progress and zero-sum training reward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from server.rules.cards import Rank
from server.sm.constants import LEVELS

type LevelTarget = Rank | Literal["WIN"]


@dataclass(frozen=True, slots=True)
class ProgressConfig:
    """Ordered mandatory levels followed by virtual WIN target."""

    required_levels: tuple[Rank, ...]


@dataclass(frozen=True, slots=True)
class TeamProgress:
    """One team's public game progress at a round boundary."""

    level: Rank
    is_declarer: bool


@dataclass(frozen=True, slots=True)
class TeamReward:
    """Zero-sum reward pair for one completed round."""

    team0: float
    team1: float


DEFAULT_PROGRESS_CONFIG = ProgressConfig(required_levels=(Rank.ACE,))


def stage_target(level: Rank, config: ProgressConfig) -> LevelTarget:
    """Return the next mandatory target from the current level."""
    level_index = _level_index(level)
    for required_level in config.required_levels:
        if _level_index(required_level) > level_index:
            return required_level
    return "WIN"


def distance_to_target(level: Rank, target: LevelTarget) -> int:
    """Return non-negative level steps from level to target."""
    return max(0, _target_index(target) - _level_index(level))


def progress_delta(
    before: TeamProgress,
    after: TeamProgress,
    config: ProgressConfig = DEFAULT_PROGRESS_CONFIG,
) -> int:
    """
    Return this team's stage progress delta for one round.

    A team that was off-stage and becomes declarer gets one control
    step.  Level movement is clipped to the stage target determined at
    the beginning of the round.
    """
    target = stage_target(before.level, config)
    before_index = _level_index(before.level)
    after_index = min(_level_index(after.level), _target_index(target))
    control_delta = (
        1 if not before.is_declarer and after.is_declarer else 0
    )
    return control_delta + max(0, after_index - before_index)


def zero_sum_rewards(
    *,
    team0_before: TeamProgress,
    team1_before: TeamProgress,
    team0_after: TeamProgress,
    team1_after: TeamProgress,
    config: ProgressConfig = DEFAULT_PROGRESS_CONFIG,
) -> TeamReward:
    """Return zero-sum rewards from both teams' progress deltas."""
    team0_delta = progress_delta(team0_before, team0_after, config)
    team1_delta = progress_delta(team1_before, team1_after, config)
    reward = float(team0_delta - team1_delta)
    return TeamReward(team0=reward, team1=-reward)


def _level_index(level: Rank) -> int:
    return LEVELS.index(level)


def _target_index(target: LevelTarget) -> int:
    if target == "WIN":
        return len(LEVELS)
    return _level_index(target)
