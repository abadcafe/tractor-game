"""Black-box tests for required-level training rewards."""

from __future__ import annotations

from server.game import Partnership
from server.game.rules.cards import Rank
from server.game.rules.progression import TerminalProgress
from server.policy_model.return_target.progress import (
    PartnershipProgress,
    RoundScore,
    continuous_progress_delta,
    progress_delta,
    zero_sum_rewards,
)

_MANDATORY_LEVELS = (Rank.ACE,)


def test_progress_delta_stays_zero_without_progress() -> None:
    before = PartnershipProgress(level=Rank.KING, is_declarer=True)
    after = PartnershipProgress(level=Rank.KING, is_declarer=True)

    assert progress_delta(before, after, _MANDATORY_LEVELS) == 0


def test_progress_delta_counts_j_to_ace_as_three_steps() -> None:
    before = PartnershipProgress(level=Rank.JACK, is_declarer=True)
    after = PartnershipProgress(level=Rank.ACE, is_declarer=True)

    assert progress_delta(before, after, _MANDATORY_LEVELS) == 3


def test_progress_delta_counts_taking_stage_control() -> None:
    before = PartnershipProgress(level=Rank.KING, is_declarer=False)
    after = PartnershipProgress(level=Rank.ACE, is_declarer=True)

    assert progress_delta(before, after, _MANDATORY_LEVELS) == 2


def test_progress_delta_counts_declarer_ace_to_win() -> None:
    before = PartnershipProgress(level=Rank.ACE, is_declarer=True)
    after = PartnershipProgress(
        level=TerminalProgress.WIN, is_declarer=True
    )

    assert progress_delta(before, after, _MANDATORY_LEVELS) == 1


def test_progress_delta_counts_non_declarer_ace_taking_stage() -> None:
    before = PartnershipProgress(level=Rank.ACE, is_declarer=False)
    after = PartnershipProgress(level=Rank.ACE, is_declarer=True)

    assert progress_delta(before, after, _MANDATORY_LEVELS) == 1


def test_zero_sum_rewards_are_opposites() -> None:
    reward = zero_sum_rewards(
        first_before=PartnershipProgress(
            level=Rank.TEN, is_declarer=True
        ),
        second_before=PartnershipProgress(
            level=Rank.TEN, is_declarer=False
        ),
        first_after=PartnershipProgress(
            level=Rank.JACK, is_declarer=True
        ),
        second_after=PartnershipProgress(
            level=Rank.TEN, is_declarer=False
        ),
        score=RoundScore(
            declarer_partnership=Partnership.FIRST,
            total_defender_points=40,
            mandatory_levels=_MANDATORY_LEVELS,
        ),
    )

    assert reward.first_partnership == 1.0
    assert reward.second_partnership == -1.0


def test_zero_sum_rewards_distinguishes_declarer_buckets() -> None:
    reward35 = _declarer_partnership_reward(total_defender_points=35)
    reward40 = _declarer_partnership_reward(total_defender_points=40)
    reward79 = _declarer_partnership_reward(total_defender_points=79)

    _assert_close(reward35, 2.125)
    _assert_close(reward40, 1.0)
    _assert_close(reward79, 0.025)
    assert reward35 > reward40 > reward79


def test_zero_sum_rewards_distinguishes_defender_progress() -> None:
    reward120 = _defender_partnership_reward(
        before_level=Rank.TEN,
        total_defender_points=120,
    )
    reward135 = _defender_partnership_reward(
        before_level=Rank.TEN,
        total_defender_points=135,
    )
    reward160 = _defender_partnership_reward(
        before_level=Rank.TEN,
        total_defender_points=160,
    )

    _assert_close(reward120, 2.0)
    _assert_close(reward135, 2.375)
    _assert_close(reward160, 3.0)
    _assert_close(reward135 - reward120, 0.375)
    _assert_close(reward160 - reward135, 0.625)


def test_zero_sum_rewards_clips_defender_to_target() -> None:
    reward120 = _defender_partnership_reward(
        before_level=Rank.KING,
        total_defender_points=120,
    )
    reward135 = _defender_partnership_reward(
        before_level=Rank.KING,
        total_defender_points=135,
    )
    reward160 = _defender_partnership_reward(
        before_level=Rank.KING,
        total_defender_points=160,
    )

    _assert_close(reward120, 2.0)
    _assert_close(reward135, 2.0)
    _assert_close(reward160, 2.0)


def test_continuous_delta_blocks_non_declarer_win_gain() -> None:
    delta = continuous_progress_delta(
        partnership=Partnership.SECOND,
        before=PartnershipProgress(level=Rank.ACE, is_declarer=False),
        after=PartnershipProgress(level=Rank.ACE, is_declarer=True),
        score=RoundScore(
            declarer_partnership=Partnership.FIRST,
            total_defender_points=160,
            mandatory_levels=_MANDATORY_LEVELS,
        ),
    )

    assert delta == 1.0


def test_continuous_progress_delta_allows_declarer_win_gain() -> None:
    delta = continuous_progress_delta(
        partnership=Partnership.FIRST,
        before=PartnershipProgress(level=Rank.ACE, is_declarer=True),
        after=PartnershipProgress(
            level=TerminalProgress.WIN,
            is_declarer=True,
        ),
        score=RoundScore(
            declarer_partnership=Partnership.FIRST,
            total_defender_points=0,
            mandatory_levels=_MANDATORY_LEVELS,
        ),
    )

    assert delta == 1.0


def _declarer_partnership_reward(
    *, total_defender_points: int
) -> float:
    reward = zero_sum_rewards(
        first_before=PartnershipProgress(
            level=Rank.TEN, is_declarer=True
        ),
        second_before=PartnershipProgress(
            level=Rank.TEN, is_declarer=False
        ),
        first_after=PartnershipProgress(
            level=Rank.TEN, is_declarer=True
        ),
        second_after=PartnershipProgress(
            level=Rank.TEN, is_declarer=False
        ),
        score=RoundScore(
            declarer_partnership=Partnership.FIRST,
            total_defender_points=total_defender_points,
            mandatory_levels=_MANDATORY_LEVELS,
        ),
    )
    return reward.first_partnership


def _defender_partnership_reward(
    *,
    before_level: Rank,
    total_defender_points: int,
) -> float:
    reward = zero_sum_rewards(
        first_before=PartnershipProgress(
            level=Rank.TEN, is_declarer=True
        ),
        second_before=PartnershipProgress(
            level=before_level,
            is_declarer=False,
        ),
        first_after=PartnershipProgress(
            level=Rank.TEN, is_declarer=False
        ),
        second_after=PartnershipProgress(
            level=before_level, is_declarer=True
        ),
        score=RoundScore(
            declarer_partnership=Partnership.FIRST,
            total_defender_points=total_defender_points,
            mandatory_levels=_MANDATORY_LEVELS,
        ),
    )
    return reward.second_partnership


def _assert_close(actual: float, expected: float) -> None:
    assert abs(actual - expected) <= 0.000000000001
