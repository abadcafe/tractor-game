"""Black-box tests for required-level progress rules."""

from __future__ import annotations

from server.rules.cards import Rank
from server.sm.required_progress import (
    DEFAULT_REQUIRED_LEVEL_PLAN,
    RequiredLevelPlan,
    TerminalProgress,
    advance_team_progress,
    distance_to_target,
    stage_target,
)


def test_stage_target_default_rules_targets_ace_then_win() -> None:
    assert (
        stage_target(Rank.TEN, DEFAULT_REQUIRED_LEVEL_PLAN) == Rank.ACE
    )
    assert (
        stage_target(Rank.ACE, DEFAULT_REQUIRED_LEVEL_PLAN)
        == TerminalProgress.WIN
    )


def test_stage_target_multiple_required_levels_are_ordered() -> None:
    plan = RequiredLevelPlan(required_levels=(Rank.JACK, Rank.ACE))

    assert stage_target(Rank.TEN, plan) == Rank.JACK
    assert stage_target(Rank.JACK, plan) == Rank.ACE
    assert stage_target(Rank.ACE, plan) == TerminalProgress.WIN


def test_distance_to_win_from_ace_is_one() -> None:
    assert distance_to_target(Rank.ACE, TerminalProgress.WIN) == 1


def test_advance_team_progress_clips_at_next_required_level() -> None:
    plan = RequiredLevelPlan(required_levels=(Rank.JACK, Rank.ACE))

    result = advance_team_progress(
        level=Rank.TEN,
        raw_gain=3,
        was_declarer=True,
        plan=plan,
    )

    assert result.level == Rank.JACK
    assert result.won_game is False


def test_advance_team_progress_declarer_ace_gain_wins() -> None:
    result = advance_team_progress(
        level=Rank.ACE,
        raw_gain=1,
        was_declarer=True,
        plan=DEFAULT_REQUIRED_LEVEL_PLAN,
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
        plan=DEFAULT_REQUIRED_LEVEL_PLAN,
    )

    assert result.level == Rank.ACE
    assert result.won_game is False
