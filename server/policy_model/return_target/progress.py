"""Zero-sum terminal returns from required-level progress."""

from __future__ import annotations

from dataclasses import dataclass

from server.game import Partnership
from server.game.rules.cards import Rank
from server.game.rules.progression import (
    TerminalProgress,
    progress_target_index,
    stage_target,
)

type ProgressLevel = Rank | TerminalProgress


@dataclass(frozen=True, slots=True)
class PartnershipProgress:
    """One partnership's public progress at a round boundary."""

    level: ProgressLevel
    is_declarer: bool


@dataclass(frozen=True, slots=True)
class RoundScore:
    """Completed-round score context for continuous returns."""

    declarer_partnership: Partnership
    total_defender_points: int
    mandatory_levels: tuple[Rank, ...]

    def __post_init__(self) -> None:
        assert self.total_defender_points >= 0
        assert self.mandatory_levels


@dataclass(frozen=True, slots=True)
class PartnershipReward:
    """Zero-sum reward pair for one completed round."""

    first_partnership: float
    second_partnership: float


def progress_delta(
    before: PartnershipProgress,
    after: PartnershipProgress,
    mandatory_levels: tuple[Rank, ...],
) -> int:
    """
    Return this team's stage progress delta for one round.

    A team that was off-stage and becomes declarer gets one control
    step. Level movement is clipped to the stage target determined at
    the beginning of the round. WIN is a valid terminal progress level,
    but only from a round the team started as declarer.
    """
    assert isinstance(before.level, Rank)
    target = stage_target(before.level, mandatory_levels)
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
    first_before: PartnershipProgress,
    second_before: PartnershipProgress,
    first_after: PartnershipProgress,
    second_after: PartnershipProgress,
    score: RoundScore,
) -> PartnershipReward:
    """Return zero-sum rewards from clipped continuous progress."""
    assert first_before.is_declarer != second_before.is_declarer
    assert first_after.is_declarer != second_after.is_declarer
    assert first_before.is_declarer == (
        score.declarer_partnership == Partnership.FIRST
    )
    assert second_before.is_declarer == (
        score.declarer_partnership == Partnership.SECOND
    )
    first_delta = continuous_progress_delta(
        partnership=Partnership.FIRST,
        before=first_before,
        after=first_after,
        score=score,
    )
    second_delta = continuous_progress_delta(
        partnership=Partnership.SECOND,
        before=second_before,
        after=second_after,
        score=score,
    )
    reward = first_delta - second_delta
    return PartnershipReward(
        first_partnership=reward,
        second_partnership=-reward,
    )


def continuous_progress_delta(
    *,
    partnership: Partnership,
    before: PartnershipProgress,
    after: PartnershipProgress,
    score: RoundScore,
) -> float:
    """
    Return one team's clipped continuous progress value.

    The score-derived level component is clipped to the team's current
    required-level target. A non-declarer already at ACE can gain stage
    control, but cannot make progress toward WIN in the same round.
    """
    assert before.is_declarer == (
        score.declarer_partnership == partnership
    )
    if after.level == TerminalProgress.WIN:
        assert before.is_declarer
        assert after.is_declarer
    raw_gain = (
        _declarer_continuous_level_gain(score.total_defender_points)
        if partnership == score.declarer_partnership
        else _defender_continuous_level_gain(
            score.total_defender_points
        )
    )
    clipped_level_gain = min(
        raw_gain,
        _level_gain_limit(before, score.mandatory_levels),
    )
    control_delta = (
        1.0 if not before.is_declarer and after.is_declarer else 0.0
    )
    return control_delta + clipped_level_gain


def _level_gain_limit(
    before: PartnershipProgress,
    mandatory_levels: tuple[Rank, ...],
) -> float:
    assert isinstance(before.level, Rank)
    target = stage_target(before.level, mandatory_levels)
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
