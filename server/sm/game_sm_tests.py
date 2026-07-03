"""Tests for sm.game_sm module."""

import subprocess
import sys
from itertools import combinations

from server.result import Ok, Rejected
from server.rules.cards import Rank

from .game_sm import (
    create_game,
    process_round_result,
    start_game,
)
from .round_sm import (
    RoundInput,
    RoundState,
    create_round,
    get_round_result,
    is_round_complete,
)
from .round_sm import (
    deal_next_card as rn_deal,
)
from .round_sm import (
    finalize_deal_bid as rn_finalize,
)
from .round_sm import (
    pass_stir as rn_pass,
)
from .round_sm import (
    play as rn_play,
)
from .round_sm import (
    stir_discard as rn_stir_discard,
)
from .scoring import RoundResult
from .types import CompletedTrick


class TestCreateGame:
    def test_create_game_initial_state(self) -> None:
        """
        Game starts waiting for the first round with both teams at level
        TWO.
        """
        state = create_game()
        assert state.round_number == 0
        assert state.winning_team is None
        assert state.team0_level == Rank.TWO
        assert state.team1_level == Rank.TWO

    def test_start_game_enters_in_round(self) -> None:
        """Starting the game records the first round number."""
        state = create_game()
        result = start_game(state)
        assert isinstance(result, Ok)
        assert result.value.round_number == 1
        assert result.value.winning_team is None

    def test_start_game_initial_levels(self) -> None:
        """Both teams start at level TWO."""
        state = create_game()
        result = start_game(state)
        assert isinstance(result, Ok)
        state = result.value
        assert state.team0_level == Rank.TWO
        assert state.team1_level == Rank.TWO


class TestProcessRoundResult:
    def test_process_round_result_updates_levels(self) -> None:
        """Round result updates team levels."""
        state = create_game()
        result = start_game(state)
        assert isinstance(result, Ok)
        state = result.value
        rr = RoundResult(
            declarer_team=0,
            round_winning_team=0,
            next_declarer_player=2,
            total_defender_points=20,
            declarer_level_gain=2,
            defender_level_gain=0,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        result = process_round_result(state, rr)
        assert isinstance(result, Ok)
        state = result.value
        assert state.team0_level == Rank.FOUR
        assert state.team1_level == Rank.TWO

    def test_process_round_result_declarer_stays(self) -> None:
        """When declarer stays, next round uses partner as declarer."""
        state = create_game()
        result = start_game(state)
        assert isinstance(result, Ok)
        state = result.value
        rr = RoundResult(
            declarer_team=0,
            round_winning_team=0,
            next_declarer_player=2,
            total_defender_points=50,
            declarer_level_gain=1,
            defender_level_gain=0,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        result = process_round_result(state, rr)
        assert isinstance(result, Ok)
        state = result.value
        assert state.declarer_team == 0
        assert state.next_declarer_player == 2

    def test_process_round_result_declarer_switches(self) -> None:
        """When declarer switches, next round uses opposite team."""
        state = create_game()
        result = start_game(state)
        assert isinstance(result, Ok)
        state = result.value
        rr = RoundResult(
            declarer_team=0,
            round_winning_team=1,
            next_declarer_player=1,
            total_defender_points=100,
            declarer_level_gain=0,
            defender_level_gain=0,
            switch_declarer=True,
            bottom_card_bonus=0,
        )
        result = process_round_result(state, rr)
        assert isinstance(result, Ok)
        state = result.value
        assert state.declarer_team == 1
        assert state.next_declarer_player == 1

    def test_process_round_result_crashes_on_declarer_mismatch(
        self,
    ) -> None:
        """RoundResult declarer must match a known game declarer."""
        completed: subprocess.CompletedProcess[str] = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from server.result import Ok\n"
                    "from server.sm.game_sm import (\n"
                    "    create_game,\n"
                    "    process_round_result,\n"
                    "    start_game,\n"
                    ")\n"
                    "from server.sm.scoring import RoundResult\n"
                    "started = start_game(create_game())\n"
                    "assert isinstance(started, Ok)\n"
                    "state = started.value.model_copy(\n"
                    "    update={'declarer_team': 0}\n"
                    ")\n"
                    "process_round_result(\n"
                    "    state,\n"
                    "    RoundResult(\n"
                    "        declarer_team=1,\n"
                    "        round_winning_team=0,\n"
                    "        next_declarer_player=0,\n"
                    "        total_defender_points=80,\n"
                    "        declarer_level_gain=0,\n"
                    "        defender_level_gain=0,\n"
                    "        switch_declarer=True,\n"
                    "        bottom_card_bonus=0,\n"
                    "    ),\n"
                    ")\n"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode != 0
        assert "AssertionError" in completed.stderr

    def test_process_round_result_crashes_on_copied_round_result(
        self,
    ) -> None:
        """model_copy cannot bypass RoundResult invariants."""
        completed: subprocess.CompletedProcess[str] = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from server.result import Ok\n"
                    "from server.sm.game_sm import (\n"
                    "    create_game,\n"
                    "    process_round_result,\n"
                    "    start_game,\n"
                    ")\n"
                    "from server.sm.scoring import RoundResult\n"
                    "started = start_game(create_game())\n"
                    "assert isinstance(started, Ok)\n"
                    "round_result = RoundResult(\n"
                    "    declarer_team=0,\n"
                    "    round_winning_team=0,\n"
                    "    next_declarer_player=2,\n"
                    "    total_defender_points=40,\n"
                    "    declarer_level_gain=1,\n"
                    "    defender_level_gain=0,\n"
                    "    switch_declarer=False,\n"
                    "    bottom_card_bonus=0,\n"
                    ")\n"
                    "invalid = round_result.model_copy(\n"
                    "    update={\n"
                    "        'switch_declarer': True,\n"
                    "        'defender_level_gain': 1,\n"
                    "    }\n"
                    ")\n"
                    "process_round_result(started.value, invalid)\n"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode != 0
        assert "AssertionError" in completed.stderr

    def test_process_round_result_clips_at_required_level(self) -> None:
        """Raw gains cannot skip over the next required level."""
        state = create_game()
        result = start_game(state)
        assert isinstance(result, Ok)
        state = result.value.model_copy(
            update={"team0_level": Rank.KING}
        )
        rr = RoundResult(
            declarer_team=0,
            round_winning_team=0,
            next_declarer_player=2,
            total_defender_points=0,
            declarer_level_gain=3,
            defender_level_gain=0,
            switch_declarer=False,
            bottom_card_bonus=0,
        )

        result = process_round_result(state, rr)

        assert isinstance(result, Ok)
        assert result.value.team0_level == Rank.ACE
        assert result.value.winning_team is None


class TestGameOver:
    def test_reaching_ace_does_not_end_game_team0(self) -> None:
        """
        Reaching ACE schedules an ACE round; it does not end the game.
        """
        state = create_game()
        result = start_game(state)
        assert isinstance(result, Ok)
        state = result.value.model_copy(
            update={"team0_level": Rank.KING}
        )
        rr = RoundResult(
            declarer_team=0,
            round_winning_team=0,
            next_declarer_player=2,
            total_defender_points=0,
            declarer_level_gain=3,
            defender_level_gain=0,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        result = process_round_result(state, rr)
        assert isinstance(result, Ok)
        state = result.value
        assert state.winning_team is None
        assert state.team0_level == Rank.ACE

    def test_jumping_from_queen_to_ace_does_not_end_game(self) -> None:
        """
        A big level jump that lands on ACE still requires playing ACE.
        """
        state = create_game()
        result = start_game(state)
        assert isinstance(result, Ok)
        state = result.value.model_copy(
            update={"team0_level": Rank.QUEEN}
        )
        rr = RoundResult(
            declarer_team=0,
            round_winning_team=0,
            next_declarer_player=2,
            total_defender_points=0,
            declarer_level_gain=3,
            defender_level_gain=0,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        result = process_round_result(state, rr)
        assert isinstance(result, Ok)
        state = result.value
        assert state.winning_team is None
        assert state.team0_level == Rank.ACE

    def test_reaching_ace_does_not_end_game_team1(self) -> None:
        """
        A defender team that reaches ACE still has to pass ACE later.
        """
        state = create_game()
        result = start_game(state)
        assert isinstance(result, Ok)
        state = result.value.model_copy(
            update={"team1_level": Rank.KING}
        )
        rr = RoundResult(
            declarer_team=0,
            round_winning_team=1,
            next_declarer_player=3,
            total_defender_points=150,
            declarer_level_gain=0,
            defender_level_gain=2,
            switch_declarer=True,
            bottom_card_bonus=0,
        )
        result = process_round_result(state, rr)
        assert isinstance(result, Ok)
        state = result.value
        assert state.winning_team is None
        assert state.team1_level == Rank.ACE

    def test_game_over_team0_after_passing_ace(self) -> None:
        """Game over when team 0 is already at ACE and gains again."""
        state = create_game()
        result = start_game(state)
        assert isinstance(result, Ok)
        state = result.value.model_copy(
            update={"team0_level": Rank.ACE}
        )
        rr = RoundResult(
            declarer_team=0,
            round_winning_team=0,
            next_declarer_player=2,
            total_defender_points=0,
            declarer_level_gain=3,
            defender_level_gain=0,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        result = process_round_result(state, rr)
        assert isinstance(result, Ok)
        state = result.value
        assert state.winning_team == 0

    def test_non_declarer_ace_gain_does_not_win(self) -> None:
        """A team at ACE must be declarer to pass the WIN target."""
        state = create_game()
        result = start_game(state)
        assert isinstance(result, Ok)
        state = result.value.model_copy(
            update={"team1_level": Rank.ACE}
        )
        rr = RoundResult(
            declarer_team=0,
            round_winning_team=1,
            next_declarer_player=3,
            total_defender_points=150,
            declarer_level_gain=0,
            defender_level_gain=2,
            switch_declarer=True,
            bottom_card_bonus=0,
        )
        result = process_round_result(state, rr)
        assert isinstance(result, Ok)
        state = result.value
        assert state.winning_team is None
        assert state.team1_level == Rank.ACE
        assert state.declarer_team == 1

    def test_game_over_team1_after_passing_ace(self) -> None:
        """Game over when team 1 is declarer at ACE and gains again."""
        state = create_game()
        result = start_game(state)
        assert isinstance(result, Ok)
        state = result.value.model_copy(
            update={"team1_level": Rank.ACE}
        )
        rr = RoundResult(
            declarer_team=1,
            round_winning_team=1,
            next_declarer_player=3,
            total_defender_points=0,
            declarer_level_gain=2,
            defender_level_gain=0,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        result = process_round_result(state, rr)
        assert isinstance(result, Ok)
        state = result.value
        assert state.winning_team == 1

    def test_ace_without_level_gain_does_not_end_game(self) -> None:
        """
        Playing an ACE round without gaining a level does not pass ACE.
        """
        state = create_game()
        result = start_game(state)
        assert isinstance(result, Ok)
        state = result.value.model_copy(
            update={"team1_level": Rank.ACE}
        )
        rr = RoundResult(
            declarer_team=0,
            round_winning_team=1,
            next_declarer_player=1,
            total_defender_points=80,
            declarer_level_gain=0,
            defender_level_gain=0,
            switch_declarer=True,
            bottom_card_bonus=0,
        )
        result = process_round_result(state, rr)
        assert isinstance(result, Ok)
        assert result.value.winning_team is None

    def test_game_not_over_mid_game(self) -> None:
        """Game continues when neither team has reached ACE."""
        state = create_game()
        result = start_game(state)
        assert isinstance(result, Ok)
        state = result.value
        rr = RoundResult(
            declarer_team=0,
            round_winning_team=0,
            next_declarer_player=2,
            total_defender_points=30,
            declarer_level_gain=2,
            defender_level_gain=0,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        result = process_round_result(state, rr)
        assert isinstance(result, Ok)
        assert result.value.winning_team is None

    def test_game_multiple_rounds(self) -> None:
        """Multiple rounds can be processed."""
        state = create_game()
        result = start_game(state)
        assert isinstance(result, Ok)
        state = result.value
        # Round 1: team 0 wins big
        r1 = RoundResult(
            declarer_team=0,
            round_winning_team=0,
            next_declarer_player=2,
            total_defender_points=20,
            declarer_level_gain=2,
            defender_level_gain=0,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        result = process_round_result(state, r1)
        assert isinstance(result, Ok)
        state = result.value
        assert state.team0_level == Rank.FOUR
        # Round 2: team 1 wins
        r2 = RoundResult(
            declarer_team=0,
            round_winning_team=1,
            next_declarer_player=1,
            total_defender_points=120,
            declarer_level_gain=0,
            defender_level_gain=1,
            switch_declarer=True,
            bottom_card_bonus=0,
        )
        result = process_round_result(state, r2)
        assert isinstance(result, Ok)
        state = result.value
        assert state.winning_team is None
        assert state.team1_level == Rank.THREE


class TestInvalidTransitions:
    def test_start_game_when_in_round(self) -> None:
        """Cannot start game when already in round."""
        state = create_game()
        result = start_game(state)
        assert isinstance(result, Ok)
        state = result.value
        result = start_game(state)
        assert isinstance(result, Rejected)

    def test_start_game_when_game_over(self) -> None:
        """Cannot start game when game is over."""
        state = create_game()
        result = start_game(state)
        assert isinstance(result, Ok)
        state = result.value.model_copy(
            update={"team0_level": Rank.ACE}
        )
        rr = RoundResult(
            declarer_team=0,
            round_winning_team=0,
            next_declarer_player=0,
            total_defender_points=0,
            declarer_level_gain=3,
            defender_level_gain=0,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        result = process_round_result(state, rr)
        assert isinstance(result, Ok)
        state = result.value
        result = start_game(state)
        assert isinstance(result, Rejected)

    def test_process_round_result_when_idle(self) -> None:
        """Cannot process round result when game is idle."""
        state = create_game()
        rr = RoundResult(
            declarer_team=0,
            round_winning_team=0,
            next_declarer_player=0,
            total_defender_points=0,
            declarer_level_gain=2,
            defender_level_gain=0,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        result = process_round_result(state, rr)
        assert isinstance(result, Rejected)

    def test_process_round_result_when_game_over(self) -> None:
        """Cannot process round result when game is over."""
        state = create_game()
        result = start_game(state)
        assert isinstance(result, Ok)
        state = result.value.model_copy(
            update={"team0_level": Rank.ACE}
        )
        rr = RoundResult(
            declarer_team=0,
            round_winning_team=0,
            next_declarer_player=0,
            total_defender_points=0,
            declarer_level_gain=3,
            defender_level_gain=0,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        result = process_round_result(state, rr)
        assert isinstance(result, Ok)
        state = result.value
        assert state.winning_team == 0
        rr2 = RoundResult(
            declarer_team=0,
            round_winning_team=0,
            next_declarer_player=0,
            total_defender_points=0,
            declarer_level_gain=3,
            defender_level_gain=0,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        result = process_round_result(state, rr2)
        assert isinstance(result, Rejected)


class TestEdgeCases:
    def test_non_declarer_ace_steal_from_ace_declarer_does_not_win(
        self,
    ) -> None:
        """
        A defender at ACE can take the stage from an ACE declarer, but
        does not pass WIN until it wins an ACE round as declarer.
        """
        state = create_game()
        result = start_game(state)
        assert isinstance(result, Ok)
        state = result.value.model_copy(
            update={"team0_level": Rank.ACE, "team1_level": Rank.ACE}
        )
        rr = RoundResult(
            declarer_team=0,
            round_winning_team=1,
            next_declarer_player=1,
            total_defender_points=120,
            declarer_level_gain=0,
            defender_level_gain=1,
            switch_declarer=True,
            bottom_card_bonus=0,
        )
        result = process_round_result(state, rr)
        assert isinstance(result, Ok)
        state = result.value
        assert state.winning_team is None
        assert state.team0_level == Rank.ACE
        assert state.team1_level == Rank.ACE
        assert state.declarer_team == 1

    def test_game_over_resets_declarer_fields(self) -> None:
        """
        On game over, declarer_team and next_declarer_player are reset
        to None.
        """
        state = create_game()
        result = start_game(state)
        assert isinstance(result, Ok)
        state = result.value
        # First round to set declarer fields
        r1 = RoundResult(
            declarer_team=0,
            round_winning_team=0,
            next_declarer_player=2,
            total_defender_points=20,
            declarer_level_gain=1,
            defender_level_gain=0,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        result = process_round_result(state, r1)
        assert isinstance(result, Ok)
        state = result.value.model_copy(
            update={"team0_level": Rank.ACE}
        )
        assert state.declarer_team == 0
        assert state.next_declarer_player == 2
        # Second round passes ACE and ends the game
        r2 = RoundResult(
            declarer_team=0,
            round_winning_team=0,
            next_declarer_player=2,
            total_defender_points=0,
            declarer_level_gain=3,
            defender_level_gain=0,
            switch_declarer=False,
            bottom_card_bonus=0,
        )
        result = process_round_result(state, r2)
        assert isinstance(result, Ok)
        state = result.value
        assert state.winning_team == 0
        assert state.declarer_team is None
        assert state.next_declarer_player is None


# ---- Integration tests with real round_sm ----

type CompletedTrickKey = tuple[
    int, int, int, tuple[tuple[int, tuple[str, ...]], ...]
]


def _completed_trick_key(
    trick: CompletedTrick | None,
) -> CompletedTrickKey | None:
    if trick is None:
        return None
    return (
        trick.lead_player,
        trick.winner,
        trick.points,
        tuple(
            (slot.player, tuple(card.id for card in slot.cards))
            for slot in trick.slots
        ),
    )


def _unwrap_round(result: Ok[RoundState] | Rejected) -> RoundState:
    """Unwrap an Ok[RoundState] | Rejected result, asserting Ok."""
    assert isinstance(result, Ok), (
        f"Expected Ok, got Rejected: {result.reason}"
    )
    return result.value


def _play_first_legal(round_state: RoundState) -> RoundState:
    """Play the first legal play for the current player in the trick."""
    trick = round_state.trick_state
    assert trick is not None
    cur = trick.cur
    hand = trick.hands[cur]
    if not hand:
        return round_state

    if trick.phase == "LEADING":
        candidate_sizes = [1]
    else:
        lead_slot = trick.slots[trick.lead_player]
        assert lead_slot is not None
        candidate_sizes = [len(lead_slot.cards)]

    for size in candidate_sizes:
        for candidate in combinations(hand, size):
            result = rn_play(
                round_state, player_index=cur, cards=list(candidate)
            )
            if isinstance(result, Ok):
                return result.value
    raise AssertionError(f"No accepted play for player {cur}")


def _complete_round_no_bid(round_state: RoundState) -> RoundState:
    """
    Drive a round through all phases with no bids, all pass stirring.
    """
    while round_state.phase == "DEAL_BID":
        if (
            round_state.deal_bid_state is None
            or round_state.deal_bid_state.phase != "DEALING"
        ):
            break
        if round_state.deal_bid_state.all_dealt:
            round_state = _unwrap_round(rn_finalize(round_state))
            break
        round_state = _unwrap_round(rn_deal(round_state))

    # STIRRING phase: handle EXCHANGING sub-phase and WAITING sub-phase
    max_stir_iterations = 20
    for _ in range(max_stir_iterations):
        if round_state.phase != "STIRRING":
            break
        stirring = round_state.stirring_state
        if stirring is None:
            break
        if stirring.phase == "EXCHANGING":
            # Discard bottom cards for the exchanging player
            assert stirring.exchange_state is not None
            assert stirring.exchanging_player is not None
            discards = stirring.exchange_state.hand_after_pickup[
                : stirring.exchange_state.count
            ]
            round_state = _unwrap_round(
                rn_stir_discard(
                    round_state,
                    player_index=stirring.exchanging_player,
                    cards=discards,
                )
            )
        elif stirring.phase == "WAITING":
            cur = stirring.current_player
            round_state = _unwrap_round(
                rn_pass(round_state, player_index=cur)
            )
        else:
            break  # COMPLETE or unknown

    prev_completed_trick_key: CompletedTrickKey | None = None
    max_iterations = 30
    for _ in range(max_iterations):
        if round_state.phase != "PLAYING":
            break
        trick = round_state.trick_state
        if trick is None:
            break
        for _ in range(4):
            if trick.phase == "RESOLVED":
                break
            round_state = _play_first_legal(round_state)
            trick = round_state.trick_state
            if trick is None:
                break
        completed_trick_key = _completed_trick_key(
            round_state.last_completed_trick
        )
        if completed_trick_key == prev_completed_trick_key:
            break
        prev_completed_trick_key = completed_trick_key

    return round_state


class TestMultipleRoundsWithRealRoundSm:
    def test_multiple_rounds_with_real_round_sm(self) -> None:
        """
        Drive multiple rounds through the game state machine using real
        round results.
        """
        game = create_game()
        result = start_game(game)
        assert isinstance(result, Ok)
        game = result.value

        for _ in range(6):
            if game.winning_team is not None:
                break

            round_state = create_round(
                RoundInput(
                    declarer_team=game.declarer_team,
                    trump_rank=game.team0_level
                    if (game.declarer_team or 0) == 0
                    else game.team1_level,
                    next_declarer_player=game.next_declarer_player,
                    team0_level=game.team0_level,
                    team1_level=game.team1_level,
                )
            )
            round_state = _complete_round_no_bid(round_state)

            if is_round_complete(round_state):
                round_result = get_round_result(round_state)
                assert round_result is not None
                result = process_round_result(game, round_result)
                assert isinstance(result, Ok)
                game = result.value
            else:
                break

        assert game.round_number >= 1

    def test_full_game_with_real_round_sm(self) -> None:
        """
        Fast game: use real RoundResults from completed rounds to drive
        game_sm.
        """
        game = create_game()
        result = start_game(game)
        assert isinstance(result, Ok)
        game = result.value

        max_rounds = 20
        for _ in range(max_rounds):
            if game.winning_team is not None:
                break

            round_state = create_round(
                RoundInput(
                    declarer_team=game.declarer_team,
                    trump_rank=game.team0_level
                    if (game.declarer_team or 0) == 0
                    else game.team1_level,
                    next_declarer_player=game.next_declarer_player,
                    team0_level=game.team0_level,
                    team1_level=game.team1_level,
                )
            )
            round_state = _complete_round_no_bid(round_state)

            if is_round_complete(round_state):
                round_result = get_round_result(round_state)
                assert round_result is not None
                result = process_round_result(game, round_result)
                assert isinstance(result, Ok)
                game = result.value
            else:
                break

        assert game.round_number >= 1
