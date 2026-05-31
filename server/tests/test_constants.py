"""Tests for engine.constants module."""
import pytest
from server.engine.card import Rank
from server.engine.constants import (
    DECK_COUNT, TOTAL_CARDS, PLAYER_COUNT, BOTTOM_CARD_COUNT,
    CARDS_PER_PLAYER, HUMAN_PLAYER_INDEX,
    TEAM_0, TEAM_1, NEXT_PLAYER,
    LEVELS, START_LEVEL,
)


class TestConstants:
    def test_constants_values(self):
        assert DECK_COUNT == 2
        assert TOTAL_CARDS == 108
        assert PLAYER_COUNT == 4
        assert BOTTOM_CARD_COUNT == 8
        assert CARDS_PER_PLAYER == 25
        assert HUMAN_PLAYER_INDEX == 3

    def test_teams(self):
        assert TEAM_0 == [0, 3]
        assert TEAM_1 == [1, 2]

    def test_next_player_map(self):
        assert NEXT_PLAYER[0] == 2  # North -> East
        assert NEXT_PLAYER[1] == 0  # West -> North
        assert NEXT_PLAYER[2] == 3  # East -> South
        assert NEXT_PLAYER[3] == 1  # South -> West

    def test_levels_order(self):
        """LEVELS must be 13 ranks from TWO to ACE."""
        assert len(LEVELS) == 13
        assert LEVELS[0] == Rank.TWO
        assert LEVELS[-1] == Rank.ACE
        # No duplicates
        assert len(set(LEVELS)) == 13

    def test_start_level(self):
        assert START_LEVEL == Rank.TWO
