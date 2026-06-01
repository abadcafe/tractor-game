"""Tests for engine.scoring module."""
from server.engine.card import Card, Suit, Rank
from server.engine.types import PlayType
from server.engine.scoring import (
    calculate_score, is_game_over, ScoreResult, ScoreThreshold,
    TARGET_LEVEL, SCORE_TABLE, DEFAULT_SETTINGS,
)

import pytest


def _card(suit: Suit, rank: Rank, deck: int = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit, rank=rank,
        is_joker=(suit == Suit.JOKER),
        is_big_joker=(rank == Rank.BIG_JOKER),
        points=0, deck=deck,
    )


class TestCalculateScore:
    def test_calculate_score_big_light(self):
        """Defender 0 points -> declarer +3."""
        result = calculate_score(
            defender_points=0,
            bottom_cards=[],
            last_trick_winner_team=0,
            last_trick_play_type=PlayType.SINGLE,
            declarer_team_index=0,
            declarer_team_level=Rank.TWO,
            defender_team_level=Rank.TWO,
        )
        assert result.declarer_level_change == 3
        assert result.switch_declarer is False

    def test_calculate_score_small_light(self):
        """Defender 1-39 points -> declarer +2."""
        result = calculate_score(
            defender_points=35,
            bottom_cards=[],
            last_trick_winner_team=0,
            last_trick_play_type=PlayType.SINGLE,
            declarer_team_index=0,
            declarer_team_level=Rank.TWO,
            defender_team_level=Rank.TWO,
        )
        assert result.declarer_level_change == 2

    def test_calculate_score_declarer_plus1(self):
        """Defender 40-79 points -> declarer +1."""
        result = calculate_score(
            defender_points=50,
            bottom_cards=[],
            last_trick_winner_team=0,
            last_trick_play_type=PlayType.SINGLE,
            declarer_team_index=0,
            declarer_team_level=Rank.TWO,
            defender_team_level=Rank.TWO,
        )
        assert result.declarer_level_change == 1

    def test_calculate_score_switch(self):
        """Defender 80-119 points -> switch declarer."""
        result = calculate_score(
            defender_points=100,
            bottom_cards=[],
            last_trick_winner_team=1,
            last_trick_play_type=PlayType.SINGLE,
            declarer_team_index=0,
            declarer_team_level=Rank.TWO,
            defender_team_level=Rank.TWO,
        )
        assert result.declarer_level_change == 0
        assert result.switch_declarer is True

    def test_calculate_score_defender_plus1(self):
        """Defender 120-159 points -> defender +1."""
        result = calculate_score(
            defender_points=130,
            bottom_cards=[],
            last_trick_winner_team=1,
            last_trick_play_type=PlayType.SINGLE,
            declarer_team_index=0,
            declarer_team_level=Rank.TWO,
            defender_team_level=Rank.TWO,
        )
        assert result.declarer_level_change == -1
        assert result.switch_declarer is True

    def test_calculate_score_defender_plus2(self):
        """Defender 160-199 points -> defender +2."""
        result = calculate_score(
            defender_points=180,
            bottom_cards=[],
            last_trick_winner_team=1,
            last_trick_play_type=PlayType.SINGLE,
            declarer_team_index=0,
            declarer_team_level=Rank.TWO,
            defender_team_level=Rank.TWO,
        )
        assert result.declarer_level_change == -2

    def test_calculate_score_defender_plus3(self):
        """Defender 200 points -> defender +3."""
        result = calculate_score(
            defender_points=200,
            bottom_cards=[],
            last_trick_winner_team=1,
            last_trick_play_type=PlayType.SINGLE,
            declarer_team_index=0,
            declarer_team_level=Rank.TWO,
            defender_team_level=Rank.TWO,
        )
        assert result.declarer_level_change == -3

    def test_calculate_score_ambush_single_x2(self):
        """Bug #4 fix: deduction with single play -> x2 multiplier."""
        bottom = [
            _card(Suit.SPADES, Rank.FIVE),   # 5 points
            _card(Suit.SPADES, Rank.TEN),    # 10 points
        ]
        result = calculate_score(
            defender_points=10,
            bottom_cards=bottom,
            last_trick_winner_team=1,
            last_trick_play_type=PlayType.SINGLE,
            declarer_team_index=0,
            declarer_team_level=Rank.TWO,
            defender_team_level=Rank.TWO,
        )
        # 10 + (5+10)*2 = 10 + 30 = 40
        assert result.bottom_card_bonus == 30
        assert result.total_defender_points == 40

    def test_calculate_score_ambush_pair_x4(self):
        """Bug #4 fix: deduction with pair play -> x4 multiplier."""
        bottom = [
            _card(Suit.SPADES, Rank.FIVE, 1),
            _card(Suit.SPADES, Rank.FIVE, 2),
        ]
        result = calculate_score(
            defender_points=10,
            bottom_cards=bottom,
            last_trick_winner_team=1,
            last_trick_play_type=PlayType.PAIR,
            declarer_team_index=0,
            declarer_team_level=Rank.TWO,
            defender_team_level=Rank.TWO,
        )
        # 10 + (5+5)*4 = 10 + 40 = 50
        assert result.bottom_card_bonus == 40
        assert result.total_defender_points == 50

    def test_calculate_score_ambush_tractor_x8(self):
        """Bug #4 fix: deduction with tractor play -> x8 multiplier."""
        bottom = [
            _card(Suit.SPADES, Rank.FIVE, 1),
            _card(Suit.SPADES, Rank.FIVE, 2),
            _card(Suit.SPADES, Rank.TEN, 1),
            _card(Suit.SPADES, Rank.TEN, 2),
        ]
        result = calculate_score(
            defender_points=10,
            bottom_cards=bottom,
            last_trick_winner_team=1,
            last_trick_play_type=PlayType.TRACTOR,
            declarer_team_index=0,
            declarer_team_level=Rank.TWO,
            defender_team_level=Rank.TWO,
        )
        # (5+5+10+10)*8 = 240
        assert result.bottom_card_bonus == 240
        # 10 + 240 = 250
        assert result.total_defender_points == 250

    def test_calculate_score_ambush_throw_x2(self):
        """CR-006: THROW play type ambush multiplier -> x2."""
        bottom = [
            _card(Suit.SPADES, Rank.FIVE),
            _card(Suit.SPADES, Rank.TEN),
        ]
        result = calculate_score(
            defender_points=10,
            bottom_cards=bottom,
            last_trick_winner_team=1,
            last_trick_play_type=PlayType.THROW,
            declarer_team_index=0,
            declarer_team_level=Rank.TWO,
            defender_team_level=Rank.TWO,
        )
        # (5+10)*2 = 30
        assert result.bottom_card_bonus == 30
        assert result.total_defender_points == 40

    def test_calculate_score_no_ambush_declarer_wins_last(self):
        """No ambush bonus when declarer wins the last trick."""
        bottom = [
            _card(Suit.SPADES, Rank.FIVE),
            _card(Suit.SPADES, Rank.TEN),
        ]
        result = calculate_score(
            defender_points=10,
            bottom_cards=bottom,
            last_trick_winner_team=0,
            last_trick_play_type=PlayType.SINGLE,
            declarer_team_index=0,
            declarer_team_level=Rank.TWO,
            defender_team_level=Rank.TWO,
        )
        assert result.bottom_card_bonus == 0
        assert result.total_defender_points == 10

    def test_calculate_score_independent_team_levels(self):
        """Bug #3 fix: each team's level changes independently."""
        result = calculate_score(
            defender_points=0,
            bottom_cards=[],
            last_trick_winner_team=0,
            last_trick_play_type=PlayType.SINGLE,
            declarer_team_index=0,
            declarer_team_level=Rank.FIVE,
            defender_team_level=Rank.THREE,
        )
        assert result.team0_new_level == Rank.EIGHT
        assert result.team1_new_level == Rank.THREE

    def test_calculate_score_declarer_index_1(self):
        """When team 1 is declarer, team 1 advances on win."""
        result = calculate_score(
            defender_points=0,
            bottom_cards=[],
            last_trick_winner_team=1,
            last_trick_play_type=PlayType.SINGLE,
            declarer_team_index=1,
            declarer_team_level=Rank.FIVE,
            defender_team_level=Rank.THREE,
        )
        # team 1 is declarer, wins with 0 points -> +3
        assert result.declarer_level_change == 3
        assert result.team0_new_level == Rank.THREE
        assert result.team1_new_level == Rank.EIGHT

    def test_calculate_score_defender_advances_on_win(self):
        """When defender wins, defender advances by abs(declarer_change)."""
        result = calculate_score(
            defender_points=130,
            bottom_cards=[],
            last_trick_winner_team=1,
            last_trick_play_type=PlayType.SINGLE,
            declarer_team_index=0,
            declarer_team_level=Rank.FIVE,
            defender_team_level=Rank.THREE,
        )
        # 130 points -> defender +1, switch declarer
        assert result.declarer_level_change == -1
        assert result.switch_declarer is True
        # declarer (team0) drops from FIVE to FOUR
        assert result.team0_new_level == Rank.FOUR
        # defender (team1) advances from THREE to FOUR
        assert result.team1_new_level == Rank.FOUR

    def test_calculate_score_defender_advances_plus2(self):
        """When defender wins big, defender advances by abs(declarer_change)."""
        result = calculate_score(
            defender_points=180,
            bottom_cards=[],
            last_trick_winner_team=1,
            last_trick_play_type=PlayType.SINGLE,
            declarer_team_index=0,
            declarer_team_level=Rank.FIVE,
            defender_team_level=Rank.THREE,
        )
        # 180 points -> defender +2
        assert result.declarer_level_change == -2
        # declarer (team0) drops from FIVE to THREE
        assert result.team0_new_level == Rank.THREE
        # defender (team1) advances from THREE to FIVE
        assert result.team1_new_level == Rank.FIVE


    # CR-005: Boundary value tests for scoring tier thresholds
    @pytest.mark.parametrize(
        "points,expected_change,expected_switch",
        [
            (0, 3, False),      # exact 0 -> big light
            (1, 2, False),      # just above 0 -> small light
            (39, 2, False),     # upper bound of small light
            (40, 1, False),     # lower bound of declarer +1
            (79, 1, False),     # upper bound of declarer +1
            (80, 0, True),      # lower bound of switch
            (119, 0, True),     # upper bound of switch
            (120, -1, True),    # lower bound of defender +1
            (159, -1, True),    # upper bound of defender +1
            (160, -2, True),    # lower bound of defender +2
            (199, -2, True),    # upper bound of defender +2
            (200, -3, True),    # exact 200 -> defender +3
        ],
    )
    def test_calculate_score_boundary_values(
        self, points, expected_change, expected_switch
    ):
        """CR-005: Verify exact boundary values for each scoring tier."""
        result = calculate_score(
            defender_points=points,
            bottom_cards=[],
            last_trick_winner_team=0,
            last_trick_play_type=PlayType.SINGLE,
            declarer_team_index=0,
            declarer_team_level=Rank.TWO,
            defender_team_level=Rank.TWO,
        )
        assert result.declarer_level_change == expected_change
        assert result.switch_declarer is expected_switch


class TestIsGameOver:
    def test_is_game_over_at_A(self):
        """Bug #6 fix: reaching A level means game over (>= not >)."""
        assert is_game_over(Rank.ACE, Rank.ACE) is True

    def test_is_game_over_below_A(self):
        assert is_game_over(Rank.KING, Rank.ACE) is False

    def test_is_game_over_past_A(self):
        assert is_game_over(Rank.ACE, Rank.KING) is True


class TestScoringConstants:
    def test_score_table_ordering(self):
        assert len(SCORE_TABLE) >= 6
        for i in range(len(SCORE_TABLE) - 1):
            assert SCORE_TABLE[i].max_points < SCORE_TABLE[i + 1].max_points
