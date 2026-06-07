"""Tests for sm.stirring module."""
import pytest
from server.sm.card_model import Card, Suit, Rank
from server.sm.stirring import (
    StirringState, StirInput, StirResult,
    create_stirring, pass_stir, stir, get_stir_result,
)


def _card(suit: Suit, rank: Rank, deck: int = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit, rank=rank,
        is_joker=(suit == Suit.JOKER),
        is_big_joker=(rank == Rank.BIG_JOKER),
        points=0, deck=deck,
    )


class TestCreateStirring:
    def test_create_stirring_initial_state(self) -> None:
        """Initial state: WAITING, current_player = CCW_next(declarer), empty pass_set."""
        state = create_stirring(StirInput(
            trump_suit=Suit.HEARTS,
            trump_rank=Rank.TWO,
            declarer_player=0,
        ))
        assert state.phase == "WAITING"
        assert state.current_player == 1  # CCW next of 0
        assert len(state.pass_set) == 0
        assert len(state.actions) == 0


class TestPassStir:
    def test_pass_stir_adds_to_pass_set(self) -> None:
        state = create_stirring(StirInput(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        state = pass_stir(state, player=1)
        assert 1 in state.pass_set

    def test_pass_stir_advances_player(self) -> None:
        state = create_stirring(StirInput(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        state = pass_stir(state, player=1)
        assert state.current_player == 3  # CCW next of 1

    def test_pass_stir_all_pass_complete(self) -> None:
        """When all 4 players pass, phase becomes COMPLETE."""
        state = create_stirring(StirInput(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        # Player 1 passes
        state = pass_stir(state, player=state.current_player)
        # Player 3 passes
        state = pass_stir(state, player=state.current_player)
        # Player 2 passes
        state = pass_stir(state, player=state.current_player)
        # Player 0 passes
        state = pass_stir(state, player=state.current_player)
        assert state.phase == "COMPLETE"

    def test_pass_stir_wrong_player_rejected(self) -> None:
        """Pass from a player who is not current_player is rejected."""
        state = create_stirring(StirInput(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        # current_player is 1; pass from player 0 should be rejected
        old_phase = state.phase
        old_current = state.current_player
        state = pass_stir(state, player=0)
        assert state.phase == old_phase
        assert state.current_player == old_current


class TestStir:
    def test_stir_pair_accepted(self) -> None:
        """Pair of trump rank in higher-priority suit is accepted."""
        state = create_stirring(StirInput(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        cards = [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)]
        state = stir(state, player=state.current_player, cards=cards)
        assert state.trump_suit == Suit.SPADES

    def test_stir_pair_changes_trump(self) -> None:
        """Stir changes trump suit to the pair's suit."""
        state = create_stirring(StirInput(
            trump_suit=Suit.CLUBS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        cards = [_card(Suit.HEARTS, Rank.TWO, 1), _card(Suit.HEARTS, Rank.TWO, 2)]
        state = stir(state, player=state.current_player, cards=cards)
        assert state.trump_suit == Suit.HEARTS

    def test_stir_pair_resets_pass_set(self) -> None:
        """After a stir, pass_set is reset (others can counter-stir)."""
        state = create_stirring(StirInput(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        # Player 1 passes first
        state = pass_stir(state, player=state.current_player)
        assert len(state.pass_set) == 1
        # Player 3 stirs
        cards = [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)]
        state = stir(state, player=state.current_player, cards=cards)
        assert len(state.pass_set) == 0

    def test_stir_single_rejected(self) -> None:
        """Single card stir is rejected (must be pair)."""
        state = create_stirring(StirInput(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        cards = [_card(Suit.SPADES, Rank.TWO, 1)]
        old_trump = state.trump_suit
        state = stir(state, player=state.current_player, cards=cards)
        assert state.trump_suit == old_trump  # unchanged

    def test_stir_same_suit_rejected(self) -> None:
        """Stirring with the same suit as current trump is rejected."""
        state = create_stirring(StirInput(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        cards = [_card(Suit.HEARTS, Rank.TWO, 1), _card(Suit.HEARTS, Rank.TWO, 2)]
        old_trump = state.trump_suit
        state = stir(state, player=state.current_player, cards=cards)
        assert state.trump_suit == old_trump

    def test_stir_lower_priority_rejected(self) -> None:
        """Pair with lower priority than current trump suit is rejected."""
        state = create_stirring(StirInput(
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO, declarer_player=0,
        ))
        # ♣ pair has lower priority than current ♠ trump
        cards = [_card(Suit.CLUBS, Rank.TWO, 1), _card(Suit.CLUBS, Rank.TWO, 2)]
        old_trump = state.trump_suit
        state = stir(state, player=state.current_player, cards=cards)
        assert state.trump_suit == old_trump

    def test_stir_joker_pair_accepted(self) -> None:
        """Joker pair stir is always accepted (highest priority)."""
        state = create_stirring(StirInput(
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO, declarer_player=0,
        ))
        cards = [_card(Suit.JOKER, Rank.BIG_JOKER, 1), _card(Suit.JOKER, Rank.BIG_JOKER, 2)]
        state = stir(state, player=state.current_player, cards=cards)
        assert state.trump_suit is None  # 无主

    def test_stir_joker_pair_sets_no_trump(self) -> None:
        """Joker pair stir sets trump_suit=None."""
        state = create_stirring(StirInput(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        cards = [_card(Suit.JOKER, Rank.SMALL_JOKER, 1), _card(Suit.JOKER, Rank.SMALL_JOKER, 2)]
        state = stir(state, player=state.current_player, cards=cards)
        assert state.trump_suit is None

    def test_stir_empty_trump_diamond_pair_accepted(self) -> None:
        """空主: even ♦ pair can stir (lowest priority still beats no trump)."""
        state = create_stirring(StirInput(
            trump_suit=None, trump_rank=Rank.TWO, declarer_player=0,
        ))
        cards = [_card(Suit.DIAMONDS, Rank.TWO, 1), _card(Suit.DIAMONDS, Rank.TWO, 2)]
        state = stir(state, player=state.current_player, cards=cards)
        assert state.trump_suit == Suit.DIAMONDS

    def test_stir_empty_trump_joker_pair_no_change(self) -> None:
        """空主: joker pair doesn't change trump (already None per SI-006)."""
        state = create_stirring(StirInput(
            trump_suit=None, trump_rank=Rank.TWO, declarer_player=0,
        ))
        cards = [_card(Suit.JOKER, Rank.BIG_JOKER, 1), _card(Suit.JOKER, Rank.BIG_JOKER, 2)]
        state = stir(state, player=state.current_player, cards=cards)
        assert state.trump_suit is None  # stays None

    def test_stir_wrong_player_rejected(self) -> None:
        """Stir from a player who is not current_player is rejected."""
        state = create_stirring(StirInput(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        cards = [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)]
        wrong_player = 0  # current is 1
        old_trump = state.trump_suit
        state = stir(state, player=wrong_player, cards=cards)
        assert state.trump_suit == old_trump


class TestStirFullFlow:
    def test_stir_full_flow_multiple_stirs(self) -> None:
        """Multiple stirs: each higher-priority pair overrides."""
        state = create_stirring(StirInput(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        # Player 1 stirs with ♣ pair (beats ♦)
        cards_club = [_card(Suit.CLUBS, Rank.TWO, 1), _card(Suit.CLUBS, Rank.TWO, 2)]
        state = stir(state, player=state.current_player, cards=cards_club)
        assert state.trump_suit == Suit.CLUBS
        # Player 3 stirs with ♠ pair (beats ♣)
        cards_spade = [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)]
        state = stir(state, player=state.current_player, cards=cards_spade)
        assert state.trump_suit == Suit.SPADES
        # Remaining players pass
        state = pass_stir(state, player=state.current_player)
        state = pass_stir(state, player=state.current_player)
        state = pass_stir(state, player=state.current_player)
        state = pass_stir(state, player=state.current_player)
        assert state.phase == "COMPLETE"

    def test_stir_complete_result(self) -> None:
        """COMPLETE state produces correct StirResult."""
        state = create_stirring(StirInput(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        # All pass
        for _ in range(4):
            state = pass_stir(state, player=state.current_player)
        assert state.phase == "COMPLETE"
        # Use get_stir_result to extract StirResult
        result = get_stir_result(state)
        assert result.final_trump_suit == Suit.HEARTS
        assert result.stir_count == 0
