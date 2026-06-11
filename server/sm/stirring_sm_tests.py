"""Tests for sm.stirring_sm module."""
from typing import Literal

from server.sm.card_model import Card, Suit, Rank
from server.sm.result import Ok, Rejected
from server.sm.stirring_sm import (
    StirInput,
    create_stirring, pass_stir, stir, get_stir_result,
)


def _card(suit: Suit, rank: Rank, deck: Literal[1, 2] = 1) -> Card:
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
        result = pass_stir(state, player=1)
        assert isinstance(result, Ok)
        assert 1 in result.value.pass_set

    def test_pass_stir_advances_player(self) -> None:
        state = create_stirring(StirInput(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        result = pass_stir(state, player=1)
        assert isinstance(result, Ok)
        assert result.value.current_player == 3  # CCW next of 1

    def test_pass_stir_all_pass_complete(self) -> None:
        """When all 4 players pass, phase becomes COMPLETE."""
        state = create_stirring(StirInput(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        # Player 1 passes
        result = pass_stir(state, player=state.current_player)
        assert isinstance(result, Ok)
        state = result.value
        # Player 3 passes
        result = pass_stir(state, player=state.current_player)
        assert isinstance(result, Ok)
        state = result.value
        # Player 2 passes
        result = pass_stir(state, player=state.current_player)
        assert isinstance(result, Ok)
        state = result.value
        # Player 0 passes
        result = pass_stir(state, player=state.current_player)
        assert isinstance(result, Ok)
        assert result.value.phase == "COMPLETE"

    def test_pass_stir_wrong_player_rejected(self) -> None:
        """Pass from a player who is not current_player is rejected."""
        state = create_stirring(StirInput(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        # current_player is 1; pass from player 0 should be rejected
        result = pass_stir(state, player=0)
        assert isinstance(result, Rejected)


class TestStir:
    def test_stir_pair_accepted(self) -> None:
        """Pair of trump rank in higher-priority suit is accepted."""
        state = create_stirring(StirInput(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        cards = [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        assert result.value.trump_suit == Suit.SPADES

    def test_stir_pair_changes_trump(self) -> None:
        """Stir changes trump suit to the pair's suit."""
        state = create_stirring(StirInput(
            trump_suit=Suit.CLUBS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        cards = [_card(Suit.HEARTS, Rank.TWO, 1), _card(Suit.HEARTS, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        assert result.value.trump_suit == Suit.HEARTS

    def test_stir_pair_resets_pass_set(self) -> None:
        """After a stir, pass_set is reset (others can counter-stir)."""
        state = create_stirring(StirInput(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        # Player 1 passes first
        result = pass_stir(state, player=state.current_player)
        assert isinstance(result, Ok)
        state = result.value
        assert len(state.pass_set) == 1
        # Player 3 stirs
        cards = [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        assert len(result.value.pass_set) == 0

    def test_stir_single_rejected(self) -> None:
        """Single card stir is rejected (must be pair)."""
        state = create_stirring(StirInput(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        cards = [_card(Suit.SPADES, Rank.TWO, 1)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Rejected)
        assert "对子" in result.reason

    def test_stir_same_suit_rejected(self) -> None:
        """Stirring with the same suit as current trump is rejected."""
        state = create_stirring(StirInput(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        cards = [_card(Suit.HEARTS, Rank.TWO, 1), _card(Suit.HEARTS, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Rejected)
        assert "优先级" in result.reason

    def test_stir_lower_priority_rejected(self) -> None:
        """Pair with lower priority than current trump suit is rejected."""
        state = create_stirring(StirInput(
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO, declarer_player=0,
        ))
        # ♣ pair has lower priority than current ♠ trump
        cards = [_card(Suit.CLUBS, Rank.TWO, 1), _card(Suit.CLUBS, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Rejected)
        assert "优先级" in result.reason

    def test_stir_joker_pair_accepted(self) -> None:
        """Joker pair stir is always accepted (highest priority)."""
        state = create_stirring(StirInput(
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO, declarer_player=0,
        ))
        cards = [_card(Suit.JOKER, Rank.BIG_JOKER, 1), _card(Suit.JOKER, Rank.BIG_JOKER, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        assert result.value.trump_suit is None  # 无主

    def test_stir_joker_pair_sets_no_trump(self) -> None:
        """Joker pair stir sets trump_suit=None."""
        state = create_stirring(StirInput(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        cards = [_card(Suit.JOKER, Rank.SMALL_JOKER, 1), _card(Suit.JOKER, Rank.SMALL_JOKER, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        assert result.value.trump_suit is None

    def test_stir_empty_trump_diamond_pair_accepted(self) -> None:
        """空主: even ♦ pair can stir (lowest priority still beats no trump)."""
        state = create_stirring(StirInput(
            trump_suit=None, trump_rank=Rank.TWO, declarer_player=0,
        ))
        cards = [_card(Suit.DIAMONDS, Rank.TWO, 1), _card(Suit.DIAMONDS, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        assert result.value.trump_suit == Suit.DIAMONDS

    def test_stir_empty_trump_joker_pair_accepted(self) -> None:
        """空主: joker pair is accepted, sets trump to None (per SI-006)."""
        state = create_stirring(StirInput(
            trump_suit=None, trump_rank=Rank.TWO, declarer_player=0,
        ))
        cards = [_card(Suit.JOKER, Rank.BIG_JOKER, 1), _card(Suit.JOKER, Rank.BIG_JOKER, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        assert result.value.trump_suit is None  # stays None

    def test_stir_empty_trump_small_joker_then_big_joker(self) -> None:
        """空主: small joker pair → big joker pair accepted (higher priority)."""
        state = create_stirring(StirInput(
            trump_suit=None, trump_rank=Rank.TWO, declarer_player=0,
        ))
        # Player 1 stirs with small joker pair
        small_jokers = [_card(Suit.JOKER, Rank.SMALL_JOKER, 1), _card(Suit.JOKER, Rank.SMALL_JOKER, 2)]
        result = stir(state, player=state.current_player, cards=small_jokers)
        assert isinstance(result, Ok)
        assert result.value.trump_suit is None
        assert result.value.current_priority == 204
        state = result.value

        # Player 3 (next player CCW) stirs with big joker pair
        big_jokers = [_card(Suit.JOKER, Rank.BIG_JOKER, 1), _card(Suit.JOKER, Rank.BIG_JOKER, 2)]
        result = stir(state, player=state.current_player, cards=big_jokers)
        assert isinstance(result, Ok)
        assert result.value.trump_suit is None
        assert result.value.current_priority == 205

    def test_stir_empty_trump_big_joker_then_small_joker_rejected(self) -> None:
        """空主: after big joker pair, small joker pair is rejected (lower priority)."""
        state = create_stirring(StirInput(
            trump_suit=None, trump_rank=Rank.TWO, declarer_player=0,
        ))
        # Player 1 stirs with big joker pair
        big_jokers = [_card(Suit.JOKER, Rank.BIG_JOKER, 1), _card(Suit.JOKER, Rank.BIG_JOKER, 2)]
        result = stir(state, player=state.current_player, cards=big_jokers)
        assert isinstance(result, Ok)
        assert result.value.current_priority == 205
        state = result.value

        # Player 3 tries small joker pair — rejected (204 <= 205)
        small_jokers = [_card(Suit.JOKER, Rank.SMALL_JOKER, 1), _card(Suit.JOKER, Rank.SMALL_JOKER, 2)]
        result = stir(state, player=state.current_player, cards=small_jokers)
        assert isinstance(result, Rejected)
        assert "优先级" in result.reason

    def test_stir_empty_trump_joker_no_infinite_loop(self) -> None:
        """空主: two players with joker pairs cannot alternate infinitely."""
        state = create_stirring(StirInput(
            trump_suit=None, trump_rank=Rank.TWO, declarer_player=0,
        ))
        # Player 1 stirs with big joker pair
        big_jokers = [_card(Suit.JOKER, Rank.BIG_JOKER, 1), _card(Suit.JOKER, Rank.BIG_JOKER, 2)]
        result = stir(state, player=state.current_player, cards=big_jokers)
        assert isinstance(result, Ok)
        state = result.value
        assert state.current_priority == 205

        # Others pass, then the same player passes (last_stir_player)
        for _ in range(4):
            result = pass_stir(state, player=state.current_player)
            assert isinstance(result, Ok)
            state = result.value

        # Phase should be COMPLETE — no infinite loop possible
        assert state.phase == "COMPLETE"

    def test_stir_wrong_player_rejected(self) -> None:
        """Stir from a player who is not current_player is rejected."""
        state = create_stirring(StirInput(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        cards = [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)]
        wrong_player = 0  # current is 1
        result = stir(state, player=wrong_player, cards=cards)
        assert isinstance(result, Rejected)
        assert "回合" in result.reason

    def test_stir_last_stir_player_rejected(self) -> None:
        """Player who just stirred cannot immediately stir again."""
        state = create_stirring(StirInput(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        cards = [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)]
        # Player 1 stirs
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        state = result.value
        # Others pass until it's player 1's turn again
        for _ in range(3):
            result = pass_stir(state, player=state.current_player)
            assert isinstance(result, Ok)
            state = result.value
        # Player 1 tries to stir again — should be rejected
        cards_hearts = [_card(Suit.HEARTS, Rank.TWO, 1), _card(Suit.HEARTS, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards_hearts)
        assert isinstance(result, Rejected)
        assert "连续" in result.reason

    def test_stir_updates_current_priority(self) -> None:
        """Successful stir updates current_priority to the new bid_value."""
        state = create_stirring(StirInput(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        # Initial: ♦ pair priority = 200
        assert state.current_priority == 200

        # Stir with ♠ pair (priority 203)
        cards = [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        assert result.value.current_priority == 203


class TestStirFullFlow:
    def test_stir_full_flow_multiple_stirs(self) -> None:
        """Multiple stirs: each higher-priority pair overrides."""
        state = create_stirring(StirInput(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        # Player 1 stirs with ♣ pair (beats ♦)
        cards_club = [_card(Suit.CLUBS, Rank.TWO, 1), _card(Suit.CLUBS, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards_club)
        assert isinstance(result, Ok)
        state = result.value
        assert state.trump_suit == Suit.CLUBS
        # Player 3 stirs with ♠ pair (beats ♣)
        cards_spade = [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards_spade)
        assert isinstance(result, Ok)
        state = result.value
        assert state.trump_suit == Suit.SPADES
        # Remaining players pass
        for _ in range(4):
            result = pass_stir(state, player=state.current_player)
            assert isinstance(result, Ok)
            state = result.value
        assert state.phase == "COMPLETE"

    def test_stir_complete_result(self) -> None:
        """COMPLETE state produces correct StirResult."""
        state = create_stirring(StirInput(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        # All pass
        for _ in range(4):
            result = pass_stir(state, player=state.current_player)
            assert isinstance(result, Ok)
            state = result.value
        assert state.phase == "COMPLETE"
        # Use get_stir_result to extract StirResult
        result = get_stir_result(state)
        assert result.final_trump_suit == Suit.HEARTS
        assert result.stir_count == 0
