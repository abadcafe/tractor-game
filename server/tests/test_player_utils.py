"""Tests for engine.player_utils module."""
import pytest
from server.engine.player_utils import (
    next_player, clockwise_distance, get_team_index, get_partner_index,
)


class TestNextPlayer:
    def test_next_player_values(self):
        assert next_player(0) == 2  # North -> East
        assert next_player(1) == 0  # West -> North
        assert next_player(2) == 3  # East -> South
        assert next_player(3) == 1  # South -> West

    def test_next_player_cycle(self):
        current = 0
        visited = [current]
        for _ in range(3):
            current = next_player(current)
            visited.append(current)
        # Clockwise: N(0) -> E(2) -> S(3) -> W(1)
        assert visited == [0, 2, 3, 1]


class TestClockwiseDistance:
    def test_clockwise_distance_same(self):
        assert clockwise_distance(0, 0) == 0

    def test_clockwise_distance_adjacent(self):
        assert clockwise_distance(0, 2) == 1  # N->E
        assert clockwise_distance(2, 3) == 1  # E->S
        assert clockwise_distance(3, 1) == 1  # S->W

    def test_clockwise_distance_full(self):
        assert clockwise_distance(0, 3) == 2  # N->E->S
        assert clockwise_distance(3, 0) == 2  # S->W->N (2 steps)
        assert clockwise_distance(0, 1) == 3  # N->E->S->W


class TestGetTeamIndex:
    def test_get_team_index_team0(self):
        assert get_team_index(0) == 0
        assert get_team_index(3) == 0

    def test_get_team_index_team1(self):
        assert get_team_index(1) == 1
        assert get_team_index(2) == 1


class TestGetPartnerIndex:
    def test_get_partner_index(self):
        """Partners sit opposite: N(0)<->S(3), W(1)<->E(2)."""
        assert get_partner_index(0) == 3  # North's partner is South
        assert get_partner_index(3) == 0  # South's partner is North
        assert get_partner_index(1) == 2  # West's partner is East
        assert get_partner_index(2) == 1  # East's partner is West
