"""Tests for sm.game_sm module."""
from server.sm.card_model import Rank
from server.sm.scoring import RoundResult
from server.sm.game_sm import (
    create_game, start_game, process_round_result,
)


class TestCreateGame:
    def test_create_game_initial_state(self) -> None:
        """Game starts in IDLE phase with both teams at level TWO."""
        state = create_game()
        assert state.phase == "IDLE"
        assert state.team0_level == Rank.TWO
        assert state.team1_level == Rank.TWO

    def test_start_game_enters_in_round(self) -> None:
        """Starting the game transitions to IN_ROUND."""
        state = create_game()
        state = start_game(state)
        assert state.phase == "IN_ROUND"

    def test_start_game_initial_levels(self) -> None:
        """Both teams start at level TWO."""
        state = create_game()
        state = start_game(state)
        assert state.team0_level == Rank.TWO
        assert state.team1_level == Rank.TWO


class TestProcessRoundResult:
    def test_process_round_result_updates_levels(self) -> None:
        """Round result updates team levels."""
        state = create_game()
        state = start_game(state)
        result = RoundResult(
            team0_new_level=Rank.FIVE,
            team1_new_level=Rank.THREE,
            next_declarer_team=0,
            next_declarer_player=3,
            total_defender_points=20,
            declarer_level_change=2,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        state = process_round_result(state, result)
        assert state.team0_level == Rank.FIVE
        assert state.team1_level == Rank.THREE

    def test_process_round_result_declarer_stays(self) -> None:
        """When declarer stays, next round uses partner as declarer."""
        state = create_game()
        state = start_game(state)
        result = RoundResult(
            team0_new_level=Rank.FOUR,
            team1_new_level=Rank.TWO,
            next_declarer_team=0,
            next_declarer_player=3,
            total_defender_points=50,
            declarer_level_change=1,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        state = process_round_result(state, result)
        assert state.declarer_team == 0
        assert state.last_declarer_player == 3

    def test_process_round_result_declarer_switches(self) -> None:
        """When declarer switches, next round uses opposite team."""
        state = create_game()
        state = start_game(state)
        result = RoundResult(
            team0_new_level=Rank.TWO,
            team1_new_level=Rank.THREE,
            next_declarer_team=1,
            next_declarer_player=1,
            total_defender_points=100,
            declarer_level_change=0,
            switch_declarer=True,
            bottom_card_bonus=0,
        )
        state = process_round_result(state, result)
        assert state.declarer_team == 1
        assert state.last_declarer_player == 1


class TestGameOver:
    def test_game_over_team0(self) -> None:
        """Game over when team 0 reaches ACE."""
        state = create_game()
        state = start_game(state)
        result = RoundResult(
            team0_new_level=Rank.ACE,
            team1_new_level=Rank.TEN,
            next_declarer_team=0,
            next_declarer_player=3,
            total_defender_points=0,
            declarer_level_change=3,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        state = process_round_result(state, result)
        assert state.phase == "GAME_OVER"
        assert state.winning_team == 0

    def test_game_over_team1(self) -> None:
        """Game over when team 1 reaches ACE."""
        state = create_game()
        state = start_game(state)
        result = RoundResult(
            team0_new_level=Rank.QUEEN,
            team1_new_level=Rank.ACE,
            next_declarer_team=1,
            next_declarer_player=2,
            total_defender_points=150,
            declarer_level_change=-2,
            switch_declarer=True,
            bottom_card_bonus=0,
        )
        state = process_round_result(state, result)
        assert state.phase == "GAME_OVER"
        assert state.winning_team == 1

    def test_game_not_over_mid_game(self) -> None:
        """Game continues when neither team has reached ACE."""
        state = create_game()
        state = start_game(state)
        result = RoundResult(
            team0_new_level=Rank.FIVE,
            team1_new_level=Rank.THREE,
            next_declarer_team=0,
            next_declarer_player=3,
            total_defender_points=30,
            declarer_level_change=2,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        state = process_round_result(state, result)
        assert state.phase == "IN_ROUND"

    def test_game_multiple_rounds(self) -> None:
        """Multiple rounds can be processed."""
        state = create_game()
        state = start_game(state)
        # Round 1: team 0 wins big
        r1 = RoundResult(
            team0_new_level=Rank.FIVE,
            team1_new_level=Rank.TWO,
            next_declarer_team=0,
            next_declarer_player=3,
            total_defender_points=20,
            declarer_level_change=2,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        state = process_round_result(state, r1)
        assert state.phase == "IN_ROUND"
        assert state.team0_level == Rank.FIVE
        # Round 2: team 1 wins
        r2 = RoundResult(
            team0_new_level=Rank.FIVE,
            team1_new_level=Rank.FIVE,
            next_declarer_team=1,
            next_declarer_player=1,
            total_defender_points=120,
            declarer_level_change=-1,
            switch_declarer=True,
            bottom_card_bonus=0,
        )
        state = process_round_result(state, r2)
        assert state.phase == "IN_ROUND"
        assert state.team1_level == Rank.FIVE


class TestInvalidTransitions:
    def test_start_game_when_in_round(self) -> None:
        """Cannot start game when already in round."""
        state = create_game()
        state = start_game(state)
        try:
            start_game(state)
            raise AssertionError("Expected ValueError for invalid phase transition")
        except ValueError:
            pass

    def test_start_game_when_game_over(self) -> None:
        """Cannot start game when game is over."""
        state = create_game()
        state = start_game(state)
        result = RoundResult(
            team0_new_level=Rank.ACE,
            team1_new_level=Rank.TWO,
            next_declarer_team=0,
            next_declarer_player=0,
            total_defender_points=0,
            declarer_level_change=3,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        state = process_round_result(state, result)
        try:
            start_game(state)
            raise AssertionError("Expected ValueError for invalid phase transition")
        except ValueError:
            pass

    def test_process_round_result_when_idle(self) -> None:
        """Cannot process round result when game is idle."""
        state = create_game()
        result = RoundResult(
            team0_new_level=Rank.FIVE,
            team1_new_level=Rank.TWO,
            next_declarer_team=0,
            next_declarer_player=0,
            total_defender_points=0,
            declarer_level_change=2,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        try:
            process_round_result(state, result)
            raise AssertionError("Expected ValueError for invalid phase transition")
        except ValueError:
            pass

    def test_process_round_result_when_game_over(self) -> None:
        """Cannot process round result when game is over."""
        state = create_game()
        state = start_game(state)
        result = RoundResult(
            team0_new_level=Rank.ACE,
            team1_new_level=Rank.TWO,
            next_declarer_team=0,
            next_declarer_player=0,
            total_defender_points=0,
            declarer_level_change=3,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        state = process_round_result(state, result)
        assert state.phase == "GAME_OVER"
        result2 = RoundResult(
            team0_new_level=Rank.ACE,
            team1_new_level=Rank.TWO,
            next_declarer_team=0,
            next_declarer_player=0,
            total_defender_points=0,
            declarer_level_change=3,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        try:
            process_round_result(state, result2)
            raise AssertionError("Expected ValueError for invalid phase transition")
        except ValueError:
            pass


class TestEdgeCases:
    def test_game_over_both_teams_ace(self) -> None:
        """When both teams reach ACE, team 0 wins (tie-breaker)."""
        state = create_game()
        state = start_game(state)
        result = RoundResult(
            team0_new_level=Rank.ACE,
            team1_new_level=Rank.ACE,
            next_declarer_team=0,
            next_declarer_player=0,
            total_defender_points=0,
            declarer_level_change=3,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        state = process_round_result(state, result)
        assert state.phase == "GAME_OVER"
        assert state.winning_team == 0
        assert state.team0_level == Rank.ACE
        assert state.team1_level == Rank.ACE

    def test_game_over_resets_declarer_fields(self) -> None:
        """On game over, declarer_team and last_declarer_player are reset to None."""
        state = create_game()
        state = start_game(state)
        # First round to set declarer fields
        r1 = RoundResult(
            team0_new_level=Rank.FOUR,
            team1_new_level=Rank.TWO,
            next_declarer_team=0,
            next_declarer_player=3,
            total_defender_points=20,
            declarer_level_change=1,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        state = process_round_result(state, r1)
        assert state.declarer_team == 0
        assert state.last_declarer_player == 3
        # Second round ends the game
        r2 = RoundResult(
            team0_new_level=Rank.ACE,
            team1_new_level=Rank.TWO,
            next_declarer_team=0,
            next_declarer_player=3,
            total_defender_points=0,
            declarer_level_change=3,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        state = process_round_result(state, r2)
        assert state.phase == "GAME_OVER"
        assert state.declarer_team is None
        assert state.last_declarer_player is None
