"""Black-box tests for required-level training rewards."""

from __future__ import annotations

from server.rules.cards import Rank
from server.rules.required_progress import TerminalProgress
from server.training.progress import (
    TeamProgress,
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
    )

    assert reward.team0 == 1.0
    assert reward.team1 == -1.0
