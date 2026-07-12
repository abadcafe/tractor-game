"""Zero-sum training reward from required-level progress."""

from __future__ import annotations

from dataclasses import dataclass

from server.game.rules.cards import Rank
from server.game.rules.required_progress import (
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
class RoundScore:
    """Completed-round score context for continuous training rewards."""

    declarer_team: int
    total_defender_points: int

    def __post_init__(self) -> None:
        assert self.declarer_team in (0, 1)
        assert self.total_defender_points >= 0


@dataclass(frozen=True, slots=True)
class TeamReward:
    """Zero-sum reward pair for one completed round."""

    team0: float
    team1: float


def progress_delta(
    before: TeamProgress,
    after: TeamProgress,
) -> int:
    """
    Return this team's stage progress delta for one round.

    A team that was off-stage and becomes declarer gets one control
    step. Level movement is clipped to the stage target determined at
    the beginning of the round. WIN is a valid terminal progress level,
    but only from a round the team started as declarer.
    """
    assert isinstance(before.level, Rank)
    target = stage_target(before.level)
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
    score: RoundScore,
) -> TeamReward:
    """Return zero-sum rewards from clipped continuous progress."""
    assert team0_before.is_declarer != team1_before.is_declarer
    assert team0_after.is_declarer != team1_after.is_declarer
    assert team0_before.is_declarer == (score.declarer_team == 0)
    assert team1_before.is_declarer == (score.declarer_team == 1)
    team0_delta = continuous_progress_delta(
        team=0,
        before=team0_before,
        after=team0_after,
        score=score,
    )
    team1_delta = continuous_progress_delta(
        team=1,
        before=team1_before,
        after=team1_after,
        score=score,
    )
    reward = team0_delta - team1_delta
    return TeamReward(team0=reward, team1=-reward)


def continuous_progress_delta(
    *,
    team: int,
    before: TeamProgress,
    after: TeamProgress,
    score: RoundScore,
) -> float:
    """
    Return one team's clipped continuous progress for training.

    The score-derived level component is clipped to the team's current
    required-level target. A non-declarer already at ACE can gain stage
    control, but cannot make progress toward WIN in the same round.
    """
    assert team in (0, 1)
    assert before.is_declarer == (score.declarer_team == team)
    if after.level == TerminalProgress.WIN:
        assert before.is_declarer
        assert after.is_declarer
    raw_gain = (
        _declarer_continuous_level_gain(score.total_defender_points)
        if team == score.declarer_team
        else _defender_continuous_level_gain(
            score.total_defender_points
        )
    )
    clipped_level_gain = min(raw_gain, _level_gain_limit(before))
    control_delta = (
        1.0 if not before.is_declarer and after.is_declarer else 0.0
    )
    return control_delta + clipped_level_gain


def _level_gain_limit(before: TeamProgress) -> float:
    assert isinstance(before.level, Rank)
    target = stage_target(before.level)
    if target == TerminalProgress.WIN and not before.is_declarer:
        return 0.0
    before_index = progress_target_index(before.level)
    target_index = progress_target_index(target)
    return float(target_index - before_index)


def _declarer_continuous_level_gain(
    total_defender_points: int,
) -> float:
    assert total_defender_points >= 0
    if total_defender_points < 40:
        return 3.0 - (total_defender_points / 40.0)
    if total_defender_points < 80:
        return 2.0 - (total_defender_points / 40.0)
    return 0.0


def _defender_continuous_level_gain(
    total_defender_points: int,
) -> float:
    assert total_defender_points >= 0
    if total_defender_points < 80:
        return 0.0
    return (total_defender_points - 80) / 40.0
