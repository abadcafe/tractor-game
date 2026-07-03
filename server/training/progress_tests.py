"""Black-box tests for required-level training rewards."""

from __future__ import annotations

from server.rules.cards import Rank
from server.sm.required_progress import (
    RequiredLevelPlan,
    TerminalProgress,
)
from server.training.progress import (
    TeamProgress,
    progress_delta,
    zero_sum_rewards,
)


def test_progress_delta_clips_at_required_level() -> None:
    plan = RequiredLevelPlan(required_levels=(Rank.JACK, Rank.ACE))
    before = TeamProgress(level=Rank.TEN, is_declarer=True)
    after = TeamProgress(level=Rank.KING, is_declarer=True)

    assert progress_delta(before, after, plan) == 1


def test_progress_delta_counts_j_to_ace_as_three_steps() -> None:
    plan = RequiredLevelPlan(required_levels=(Rank.JACK, Rank.ACE))
    before = TeamProgress(level=Rank.JACK, is_declarer=True)
    after = TeamProgress(level=Rank.ACE, is_declarer=True)

    assert progress_delta(before, after, plan) == 3


def test_progress_delta_counts_taking_stage_control() -> None:
    plan = RequiredLevelPlan(required_levels=(Rank.JACK, Rank.ACE))
    before = TeamProgress(level=Rank.TEN, is_declarer=False)
    after = TeamProgress(level=Rank.JACK, is_declarer=True)

    assert progress_delta(before, after, plan) == 2


def test_progress_delta_counts_declarer_ace_to_win() -> None:
    before = TeamProgress(level=Rank.ACE, is_declarer=True)
    after = TeamProgress(level=TerminalProgress.WIN, is_declarer=True)

    assert progress_delta(before, after) == 1


def test_progress_delta_counts_non_declarer_ace_taking_stage() -> None:
    before = TeamProgress(level=Rank.ACE, is_declarer=False)
    after = TeamProgress(level=Rank.ACE, is_declarer=True)

    assert progress_delta(before, after) == 1


def test_zero_sum_rewards_are_opposites() -> None:
    reward = zero_sum_rewards(
        team0_before=TeamProgress(level=Rank.TEN, is_declarer=True),
        team1_before=TeamProgress(level=Rank.TEN, is_declarer=False),
        team0_after=TeamProgress(level=Rank.JACK, is_declarer=True),
        team1_after=TeamProgress(level=Rank.TEN, is_declarer=False),
        required_level_plan=RequiredLevelPlan(
            required_levels=(Rank.JACK, Rank.ACE)
        ),
    )

    assert reward.team0 == 1.0
    assert reward.team1 == -1.0
