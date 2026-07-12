"""Black-box tests for level progression rules."""

from __future__ import annotations

from server.game.rules.cards import Rank
from server.game.rules.level_progression import LEVELS, advance_level


def test_levels_order() -> None:
    assert LEVELS[0] == Rank.TWO
    assert LEVELS[-1] == Rank.ACE
    assert len(LEVELS) == 13


def test_advance_level_forward() -> None:
    assert advance_level(Rank.TWO, 3) == Rank.FIVE


def test_advance_level_backward() -> None:
    assert advance_level(Rank.FIVE, -2) == Rank.THREE


def test_advance_level_clamp_lower() -> None:
    assert advance_level(Rank.TWO, -1) == Rank.TWO


def test_advance_level_clamp_upper() -> None:
    assert advance_level(Rank.ACE, 1) == Rank.ACE
