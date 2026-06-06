"""Tests for sm.constants module."""
import pytest
from server.sm.card_model import Rank
from server.sm.constants import (
    PLAYER_COUNT, BOTTOM_CARD_COUNT, TOTAL_CARDS,
    TEAM_0, TEAM_1, CCW_NEXT, LEVELS, TOTAL_POINTS,
    get_team_index, get_partner_index, next_player_ccw,
    advance_level, ScoreThreshold, SCORE_THRESHOLDS,
    HUMAN_PLAYER_INDEX,
)


class TestPlayerPositioning:
    def test_player_count(self) -> None:
        assert PLAYER_COUNT == 4

    def test_ccw_next_cycle(self) -> None:
        """Counterclockwise: 0→1→3→2→0."""
        assert next_player_ccw(0) == 1
        assert next_player_ccw(1) == 3
        assert next_player_ccw(3) == 2
        assert next_player_ccw(2) == 0

    def test_ccw_next_starts_from_zero(self) -> None:
        """Starting from player 0, full cycle returns to 0."""
        p = 0
        for _ in range(4):
            p = next_player_ccw(p)
        assert p == 0

    def test_team0_members(self) -> None:
        """Team 0: North(0) + South(3)."""
        assert TEAM_0 == [0, 3]

    def test_team1_members(self) -> None:
        """Team 1: West(1) + East(2)."""
        assert TEAM_1 == [1, 2]


class TestTeamUtils:
    def test_get_team_index_team0(self) -> None:
        assert get_team_index(0) == 0
        assert get_team_index(3) == 0

    def test_get_team_index_team1(self) -> None:
        assert get_team_index(1) == 1
        assert get_team_index(2) == 1

    def test_get_partner_index_team0(self) -> None:
        """N(0) partner is S(3), S(3) partner is N(0)."""
        assert get_partner_index(0) == 3
        assert get_partner_index(3) == 0

    def test_get_partner_index_team1(self) -> None:
        """W(1) partner is E(2), E(2) partner is W(1)."""
        assert get_partner_index(1) == 2
        assert get_partner_index(2) == 1


class TestLevelProgression:
    def test_levels_order(self) -> None:
        """Levels go 2→3→4→...→A."""
        assert LEVELS[0] == Rank.TWO
        assert LEVELS[-1] == Rank.ACE
        assert len(LEVELS) == 13

    def test_advance_level_forward(self) -> None:
        """Advance TWO by 3 = FIVE."""
        assert advance_level(Rank.TWO, 3) == Rank.FIVE

    def test_advance_level_backward(self) -> None:
        """Advance FIVE by -2 = THREE."""
        assert advance_level(Rank.FIVE, -2) == Rank.THREE

    def test_advance_level_clamp_lower(self) -> None:
        """Cannot go below TWO."""
        assert advance_level(Rank.TWO, -1) == Rank.TWO

    def test_advance_level_clamp_upper(self) -> None:
        """Cannot go above ACE."""
        assert advance_level(Rank.ACE, 1) == Rank.ACE


class TestScoringConstants:
    def test_bottom_card_count(self) -> None:
        assert BOTTOM_CARD_COUNT == 8

    def test_total_cards(self) -> None:
        assert TOTAL_CARDS == 108

    def test_total_points(self) -> None:
        """2 decks × (4 suits × 3 scoring ranks) = 200 points total."""
        assert TOTAL_POINTS == 200

    def test_score_thresholds_ordering(self) -> None:
        """Thresholds must be strictly increasing in max_points."""
        for i in range(len(SCORE_THRESHOLDS) - 1):
            assert SCORE_THRESHOLDS[i].max_points < SCORE_THRESHOLDS[i + 1].max_points

    def test_human_player_index(self) -> None:
        assert HUMAN_PLAYER_INDEX == 3
