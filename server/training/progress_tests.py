"""Black-box tests for required-level training rewards."""

from __future__ import annotations

from server.game.rules.cards import Rank
from server.game.rules.required_progress import TerminalProgress
from server.training.progress import (
    RoundScore,
    TeamProgress,
    continuous_progress_delta,
    progress_delta,
    zero_sum_rewards,
)


def test_progress_delta_stays_zero_without_progress() -> None:
    before = TeamProgress(level=Rank.KING, is_declarer=True)
    after = TeamProgress(level=Rank.KING, is_declarer=True)

    assert progress_delta(before, after) == 0


def test_progress_delta_counts_j_to_ace_as_three_steps() -> None:
    before = TeamProgress(level=Rank.JACK, is_declarer=True)
    after = TeamProgress(level=Rank.ACE, is_declarer=True)

    assert progress_delta(before, after) == 3


def test_progress_delta_counts_taking_stage_control() -> None:
    before = TeamProgress(level=Rank.KING, is_declarer=False)
    after = TeamProgress(level=Rank.ACE, is_declarer=True)

    assert progress_delta(before, after) == 2


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
        score=RoundScore(declarer_team=0, total_defender_points=40),
    )

    assert reward.team0 == 1.0
    assert reward.team1 == -1.0


def test_zero_sum_rewards_distinguishes_declarer_buckets() -> None:
    reward35 = _declarer_team_reward(total_defender_points=35)
    reward40 = _declarer_team_reward(total_defender_points=40)
    reward79 = _declarer_team_reward(total_defender_points=79)

    _assert_close(reward35, 2.125)
    _assert_close(reward40, 1.0)
    _assert_close(reward79, 0.025)
    assert reward35 > reward40 > reward79


def test_zero_sum_rewards_distinguishes_defender_progress() -> None:
    reward120 = _defender_team_reward(
        before_level=Rank.TEN,
        total_defender_points=120,
    )
    reward135 = _defender_team_reward(
        before_level=Rank.TEN,
        total_defender_points=135,
    )
    reward160 = _defender_team_reward(
        before_level=Rank.TEN,
        total_defender_points=160,
    )

    _assert_close(reward120, 2.0)
    _assert_close(reward135, 2.375)
    _assert_close(reward160, 3.0)
    _assert_close(reward135 - reward120, 0.375)
    _assert_close(reward160 - reward135, 0.625)


def test_zero_sum_rewards_clips_defender_to_target() -> None:
    reward120 = _defender_team_reward(
        before_level=Rank.KING,
        total_defender_points=120,
    )
    reward135 = _defender_team_reward(
        before_level=Rank.KING,
        total_defender_points=135,
    )
    reward160 = _defender_team_reward(
        before_level=Rank.KING,
        total_defender_points=160,
    )

    _assert_close(reward120, 2.0)
    _assert_close(reward135, 2.0)
    _assert_close(reward160, 2.0)


def test_continuous_delta_blocks_non_declarer_win_gain() -> None:
    delta = continuous_progress_delta(
        team=1,
        before=TeamProgress(level=Rank.ACE, is_declarer=False),
        after=TeamProgress(level=Rank.ACE, is_declarer=True),
        score=RoundScore(declarer_team=0, total_defender_points=160),
    )

    assert delta == 1.0


def test_continuous_progress_delta_allows_declarer_win_gain() -> None:
    delta = continuous_progress_delta(
        team=0,
        before=TeamProgress(level=Rank.ACE, is_declarer=True),
        after=TeamProgress(
            level=TerminalProgress.WIN,
            is_declarer=True,
        ),
        score=RoundScore(declarer_team=0, total_defender_points=0),
    )

    assert delta == 1.0


def _declarer_team_reward(*, total_defender_points: int) -> float:
    reward = zero_sum_rewards(
        team0_before=TeamProgress(level=Rank.TEN, is_declarer=True),
        team1_before=TeamProgress(level=Rank.TEN, is_declarer=False),
        team0_after=TeamProgress(level=Rank.TEN, is_declarer=True),
        team1_after=TeamProgress(level=Rank.TEN, is_declarer=False),
        score=RoundScore(
            declarer_team=0,
            total_defender_points=total_defender_points,
        ),
    )
    return reward.team0


def _defender_team_reward(
    *,
    before_level: Rank,
    total_defender_points: int,
) -> float:
    reward = zero_sum_rewards(
        team0_before=TeamProgress(level=Rank.TEN, is_declarer=True),
        team1_before=TeamProgress(
            level=before_level,
            is_declarer=False,
        ),
        team0_after=TeamProgress(level=Rank.TEN, is_declarer=False),
        team1_after=TeamProgress(level=before_level, is_declarer=True),
        score=RoundScore(
            declarer_team=0,
            total_defender_points=total_defender_points,
        ),
    )
    return reward.team1


def _assert_close(actual: float, expected: float) -> None:
    assert abs(actual - expected) <= 0.000000000001
