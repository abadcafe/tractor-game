"""Zero-sum training reward from required-level progress."""

from __future__ import annotations

from dataclasses import dataclass

from server.rules.cards import Rank
from server.sm.required_progress import (
    DEFAULT_REQUIRED_LEVEL_PLAN,
    RequiredLevelPlan,
    TerminalProgress,
    progress_target_index,
    stage_target,
)

type ProgressLevel = Rank | TerminalProgress


@dataclass(frozen=True, slots=True)
class TeamProgress:
    """One team's public game progress at a round boundary."""

    level: ProgressLevel
    is_declarer: bool


@dataclass(frozen=True, slots=True)
class TeamReward:
    """Zero-sum reward pair for one completed round."""

    team0: float
    team1: float


def progress_delta(
    before: TeamProgress,
    after: TeamProgress,
    required_level_plan: RequiredLevelPlan = (
        DEFAULT_REQUIRED_LEVEL_PLAN
    ),
) -> int:
    """
    Return this team's stage progress delta for one round.

    A team that was off-stage and becomes declarer gets one control
    step. Level movement is clipped to the stage target determined at
    the beginning of the round. WIN is a valid terminal progress level,
    but only from a round the team started as declarer.
    """
    assert isinstance(before.level, Rank)
    target = stage_target(before.level, required_level_plan)
    if after.level == TerminalProgress.WIN:
        assert before.is_declarer
        assert after.is_declarer
        assert target == TerminalProgress.WIN
    before_index = progress_target_index(before.level)
    after_index = min(
        progress_target_index(after.level),
        progress_target_index(target),
    )
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
    required_level_plan: RequiredLevelPlan = (
        DEFAULT_REQUIRED_LEVEL_PLAN
    ),
) -> TeamReward:
    """Return zero-sum rewards from both teams' progress deltas."""
    team0_delta = progress_delta(
        team0_before,
        team0_after,
        required_level_plan,
    )
    team1_delta = progress_delta(
        team1_before,
        team1_after,
        required_level_plan,
    )
    reward = float(team0_delta - team1_delta)
    return TeamReward(team0=reward, team1=-reward)
