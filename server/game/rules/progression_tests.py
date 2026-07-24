"""Black-box tests for required-level progress rules."""

from __future__ import annotations

from server.game.rules.cards import Rank
from server.game.rules.progression import (
    TerminalProgress,
    advance_team,
    distance_to_target,
    stage_target,
)

_MANDATORY_LEVELS = (Rank.ACE,)


def test_stage_target_default_rules_targets_ace_then_win() -> None:
    assert stage_target(Rank.TEN, _MANDATORY_LEVELS) == Rank.ACE
    assert (
        stage_target(Rank.ACE, _MANDATORY_LEVELS)
        == TerminalProgress.WIN
    )


def test_distance_to_win_from_ace_is_one() -> None:
    assert distance_to_target(Rank.ACE, TerminalProgress.WIN) == 1


def test_advance_team_progress_clips_at_ace() -> None:
    result = advance_team(
        level=Rank.KING,
        raw_gain=3,
        was_declarer=True,
        mandatory_levels=_MANDATORY_LEVELS,
    )

    assert result.level == Rank.ACE
    assert result.won_game is False


def test_advance_team_progress_declarer_ace_gain_wins() -> None:
    result = advance_team(
        level=Rank.ACE,
        raw_gain=1,
        was_declarer=True,
        mandatory_levels=_MANDATORY_LEVELS,
    )

    assert result.level == Rank.ACE
    assert result.won_game is True


def test_advance_team_progress_non_declarer_ace_gain_takes_stage() -> (
    None
):
    result = advance_team(
        level=Rank.ACE,
        raw_gain=1,
        was_declarer=False,
        mandatory_levels=_MANDATORY_LEVELS,
    )

    assert result.level == Rank.ACE
    assert result.won_game is False


def test_custom_mandatory_levels_are_reached_in_order() -> None:
    mandatory = (Rank.FIVE, Rank.TEN, Rank.ACE)

    assert stage_target(Rank.TWO, mandatory) == Rank.FIVE
    assert stage_target(Rank.FIVE, mandatory) == Rank.TEN
    assert stage_target(Rank.TEN, mandatory) == Rank.ACE
    assert stage_target(Rank.ACE, mandatory) == TerminalProgress.WIN


def test_large_gain_cannot_skip_a_mandatory_level() -> None:
    result = advance_team(
        level=Rank.TWO,
        raw_gain=20,
        was_declarer=True,
        mandatory_levels=(Rank.FIVE, Rank.TEN, Rank.ACE),
    )

    assert result.level == Rank.FIVE
    assert not result.won_game


def test_zero_gain_preserves_level_and_never_wins() -> None:
    result = advance_team(
        level=Rank.ACE,
        raw_gain=0,
        was_declarer=True,
        mandatory_levels=_MANDATORY_LEVELS,
    )

    assert result.level == Rank.ACE
    assert not result.won_game


def test_distance_is_zero_after_reaching_target() -> None:
    assert distance_to_target(Rank.ACE, Rank.TEN) == 0
