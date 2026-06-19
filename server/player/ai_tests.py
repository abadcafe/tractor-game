"""Tests for AIPlayer type boundaries."""

from __future__ import annotations

from . import ai, base


def test_ai_player_is_player() -> None:
    player = ai.AIPlayer(index=0)
    assert isinstance(player, base.Player)


def test_ai_player_is_distinct_exported_type() -> None:
    player = ai.AIPlayer(index=0)
    assert isinstance(player, ai.AIPlayer)
    assert type(player) is ai.AIPlayer
