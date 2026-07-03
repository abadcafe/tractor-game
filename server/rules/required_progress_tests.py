"""Black-box tests for required-level progress rules."""

from __future__ import annotations

from server.rules.cards import Rank
from server.rules.required_progress import (
    MANDATORY_LEVELS,
    TerminalProgress,
    advance_team_progress,
    distance_to_target,
    stage_target,
)


def test_stage_target_default_rules_targets_ace_then_win() -> None:
    assert MANDATORY_LEVELS == (Rank.ACE,)
    assert stage_target(Rank.TEN) == Rank.ACE
    assert stage_target(Rank.ACE) == TerminalProgress.WIN


def test_distance_to_win_from_ace_is_one() -> None:
    assert distance_to_target(Rank.ACE, TerminalProgress.WIN) == 1


def test_advance_team_progress_clips_at_ace() -> None:
    result = advance_team_progress(
        level=Rank.KING,
        raw_gain=3,
        was_declarer=True,
    )

    assert result.level == Rank.ACE
    assert result.won_game is False


def test_advance_team_progress_declarer_ace_gain_wins() -> None:
    result = advance_team_progress(
        level=Rank.ACE,
        raw_gain=1,
        was_declarer=True,
    )

    assert result.level == Rank.ACE
    assert result.won_game is True


def test_advance_team_progress_non_declarer_ace_gain_takes_stage() -> (
    None
):
    result = advance_team_progress(
        level=Rank.ACE,
        raw_gain=1,
        was_declarer=False,
    )

    assert result.level == Rank.ACE
    assert result.won_game is False
