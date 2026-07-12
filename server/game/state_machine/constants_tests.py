"""Tests for sm.constants module."""

from types import MappingProxyType

from .constants import (
    BOTTOM_CARD_COUNT,
    CCW_NEXT,
    PLAYER_COUNT,
    SCORE_THRESHOLDS,
    TEAM_0,
    TEAM_1,
    TOTAL_CARDS,
    TOTAL_POINTS,
    get_partner_index,
    get_team_index,
    next_player_ccw,
)


class TestPlayerPositioning:
    def test_player_count(self) -> None:
        assert PLAYER_COUNT == 4

    def test_ccw_next_cycle(self) -> None:
        """Counterclockwise: 0→1→2→3→0."""
        assert next_player_ccw(0) == 1
        assert next_player_ccw(1) == 2
        assert next_player_ccw(2) == 3
        assert next_player_ccw(3) == 0

    def test_ccw_next_starts_from_zero(self) -> None:
        """Starting from player 0, full cycle returns to 0."""
        p = 0
        for _ in range(4):
            p = next_player_ccw(p)
        assert p == 0

    def test_team0_members(self) -> None:
        """Team 0: North(0) + South(2)."""
        assert TEAM_0 == (0, 2)

    def test_team1_members(self) -> None:
        """Team 1: West(1) + East(3)."""
        assert TEAM_1 == (1, 3)


class TestTeamUtils:
    def test_get_team_index_team0(self) -> None:
        assert get_team_index(0) == 0
        assert get_team_index(2) == 0

    def test_get_team_index_team1(self) -> None:
        assert get_team_index(1) == 1
        assert get_team_index(3) == 1

    def test_get_partner_index_team0(self) -> None:
        """N(0) partner is S(2), S(2) partner is N(0)."""
        assert get_partner_index(0) == 2
        assert get_partner_index(2) == 0

    def test_get_partner_index_team1(self) -> None:
        """W(1) partner is E(3), E(3) partner is W(1)."""
        assert get_partner_index(1) == 3
        assert get_partner_index(3) == 1


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
            assert (
                SCORE_THRESHOLDS[i].max_points
                < SCORE_THRESHOLDS[i + 1].max_points
            )


class TestImmutability:
    def test_team0_is_tuple(self) -> None:
        assert isinstance(TEAM_0, tuple)

    def test_team1_is_tuple(self) -> None:
        assert isinstance(TEAM_1, tuple)

    def test_ccw_next_is_frozen(self) -> None:
        assert isinstance(CCW_NEXT, MappingProxyType)

    def test_score_thresholds_is_tuple(self) -> None:
        assert isinstance(SCORE_THRESHOLDS, tuple)

    def test_score_threshold_is_frozen_dataclass(self) -> None:
        from dataclasses import FrozenInstanceError

        st = SCORE_THRESHOLDS[0]
        try:
            setattr(st, "max_points", 999)
            raise AssertionError(
                "Should have raised FrozenInstanceError"
            )
        except FrozenInstanceError:
            pass


class TestInputValidation:
    def test_next_player_ccw_invalid(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Invalid player index"):
            next_player_ccw(99)

    def test_get_team_index_invalid(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Invalid player index"):
            get_team_index(-1)

    def test_get_partner_index_invalid(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Invalid player index"):
            get_partner_index(4)
