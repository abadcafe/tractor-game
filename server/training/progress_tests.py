"""Black-box tests for required-level progress rewards."""

from __future__ import annotations

from server.rules.cards import Rank
from server.training.progress import (
    ProgressConfig,
    TeamProgress,
    distance_to_target,
    progress_delta,
    stage_target,
    zero_sum_rewards,
)


def test_stage_target_current_rules_targets_ace_then_win() -> None:
    config = ProgressConfig(required_levels=(Rank.ACE,))
    assert stage_target(Rank.TEN, config) == Rank.ACE
    assert stage_target(Rank.ACE, config) == "WIN"


def test_stage_target_j_rule_moves_from_j_to_ace() -> None:
    config = ProgressConfig(required_levels=(Rank.JACK, Rank.ACE))
    assert stage_target(Rank.TEN, config) == Rank.JACK
    assert stage_target(Rank.JACK, config) == Rank.ACE


def test_progress_delta_clips_at_required_level() -> None:
    config = ProgressConfig(required_levels=(Rank.JACK, Rank.ACE))
    before = TeamProgress(level=Rank.TEN, is_declarer=True)
    after = TeamProgress(level=Rank.KING, is_declarer=True)
    assert progress_delta(before, after, config) == 1


def test_progress_delta_counts_j_to_ace_as_three_steps() -> None:
    config = ProgressConfig(required_levels=(Rank.JACK, Rank.ACE))
    before = TeamProgress(level=Rank.JACK, is_declarer=True)
    after = TeamProgress(level=Rank.ACE, is_declarer=True)
    assert progress_delta(before, after, config) == 3


def test_progress_delta_counts_taking_stage_control() -> None:
    config = ProgressConfig(required_levels=(Rank.JACK, Rank.ACE))
    before = TeamProgress(level=Rank.TEN, is_declarer=False)
    after = TeamProgress(level=Rank.JACK, is_declarer=True)
    assert progress_delta(before, after, config) == 2


def test_zero_sum_rewards_are_opposites() -> None:
    reward = zero_sum_rewards(
        team0_before=TeamProgress(level=Rank.TEN, is_declarer=True),
        team1_before=TeamProgress(level=Rank.TEN, is_declarer=False),
        team0_after=TeamProgress(level=Rank.JACK, is_declarer=True),
        team1_after=TeamProgress(level=Rank.TEN, is_declarer=False),
        config=ProgressConfig(required_levels=(Rank.JACK, Rank.ACE)),
    )
    assert reward.team0 == 1.0
    assert reward.team1 == -1.0


def test_distance_to_win_from_ace_is_one() -> None:
    assert distance_to_target(Rank.ACE, "WIN") == 1
