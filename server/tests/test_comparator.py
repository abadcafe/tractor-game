"""Tests for rules.comparator module."""
import pytest
from server.engine.card import Card, Suit, Rank
from server.rules.comparator import (
    trump_order, non_trump_order, compare_cards, is_equal_in_trump,
    effective_suit, compare_plays, sort_by_trump_order,
)


def _card(suit: Suit, rank: Rank, deck: int = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit, rank=rank,
        is_joker=(suit == Suit.JOKER),
        is_big_joker=(rank == Rank.BIG_JOKER),
        points=0, deck=deck,
    )


class TestTrumpOrder:
    def test_trump_order_big_joker(self):
        c = _card(Suit.JOKER, Rank.BIG_JOKER)
        assert trump_order(c, Suit.HEARTS, Rank.TWO) == 100

    def test_trump_order_small_joker(self):
        c = _card(Suit.JOKER, Rank.SMALL_JOKER)
        assert trump_order(c, Suit.HEARTS, Rank.TWO) == 90

    def test_trump_order_trump_rank_trump_suit(self):
        c = _card(Suit.HEARTS, Rank.TWO)
        assert trump_order(c, Suit.HEARTS, Rank.TWO) == 80

    def test_trump_order_trump_rank_other_suit(self):
        c = _card(Suit.SPADES, Rank.TWO)
        order = trump_order(c, Suit.HEARTS, Rank.TWO)
        assert 70 <= order <= 79  # Sub-trump rank

    def test_trump_order_trump_suit_other(self):
        c = _card(Suit.HEARTS, Rank.ACE)
        order = trump_order(c, Suit.HEARTS, Rank.TWO)
        assert 47 <= order <= 59  # Trump suit, non-trump-rank

    def test_trump_order_non_trump(self):
        c = _card(Suit.SPADES, Rank.ACE)
        order = trump_order(c, Suit.HEARTS, Rank.TWO)
        assert 0 <= order <= 12  # Non-trump card

    def test_trump_order_sub_trump_outranks_trump_suit_non_rank(self):
        """Sub-trump rank cards must outrank all trump-suit non-rank cards."""
        sub_trump = _card(Suit.CLUBS, Rank.TWO)  # lowest sub-trump
        trump_suit_ace = _card(Suit.HEARTS, Rank.ACE)  # highest trump-suit non-rank
        assert trump_order(sub_trump, Suit.HEARTS, Rank.TWO) > \
               trump_order(trump_suit_ace, Suit.HEARTS, Rank.TWO)


class TestEffectiveSuit:
    def test_effective_suit_trump_joker(self):
        c = _card(Suit.JOKER, Rank.BIG_JOKER)
        assert effective_suit(c, Suit.HEARTS, Rank.TWO) == "trump"

    def test_effective_suit_trump_rank(self):
        c = _card(Suit.SPADES, Rank.TWO)
        assert effective_suit(c, Suit.HEARTS, Rank.TWO) == "trump"

    def test_effective_suit_trump_suit(self):
        c = _card(Suit.HEARTS, Rank.ACE)
        assert effective_suit(c, Suit.HEARTS, Rank.TWO) == "trump"

    def test_effective_suit_non_trump(self):
        c = _card(Suit.SPADES, Rank.ACE)
        assert effective_suit(c, Suit.HEARTS, Rank.TWO) == Suit.SPADES

    def test_effective_suit_sub_trump_rank(self):
        """A trump-rank card of a non-trump suit is still trump."""
        c = _card(Suit.CLUBS, Rank.TWO)
        assert effective_suit(c, Suit.HEARTS, Rank.TWO) == "trump"


class TestComparePlays:
    def test_compare_plays_trump_beats_non_trump(self):
        """Trump play beats non-trump play."""
        trump_card = _card(Suit.HEARTS, Rank.ACE)
        non_trump_card = _card(Suit.SPADES, Rank.ACE)
        result = compare_plays(
            [trump_card], [non_trump_card],
            Suit.HEARTS, Rank.TWO, Suit.SPADES,
        )
        assert result > 0

    def test_compare_plays_same_suit_highest_wins(self):
        """Same suit: higher rank wins."""
        ace = _card(Suit.SPADES, Rank.ACE)
        king = _card(Suit.SPADES, Rank.KING)
        result = compare_plays(
            [ace], [king],
            Suit.HEARTS, Rank.TWO, Suit.SPADES,
        )
        assert result > 0

    def test_compare_plays_different_non_trump_lead_wins(self):
        """Different non-trump suits: lead suit wins."""
        spade_a = _card(Suit.SPADES, Rank.ACE)
        club_a = _card(Suit.CLUBS, Rank.ACE)
        result = compare_plays(
            [spade_a], [club_a],
            Suit.HEARTS, Rank.TWO, Suit.SPADES,
        )
        assert result > 0  # Spade leads, spade wins

    def test_compare_plays_pair_vs_pair(self):
        """Pair vs pair: higher pair wins."""
        ace1 = _card(Suit.SPADES, Rank.ACE, deck=1)
        ace2 = _card(Suit.SPADES, Rank.ACE, deck=2)
        king1 = _card(Suit.SPADES, Rank.KING, deck=1)
        king2 = _card(Suit.SPADES, Rank.KING, deck=2)
        result = compare_plays(
            [ace1, ace2], [king1, king2],
            Suit.HEARTS, Rank.TWO, Suit.SPADES,
        )
        assert result > 0

    def test_compare_plays_tractor_vs_tractor(self):
        """Tractor vs tractor: higher wins."""
        aa = [_card(Suit.SPADES, Rank.ACE, d) for d in (1, 2)]
        kk = [_card(Suit.SPADES, Rank.KING, d) for d in (1, 2)]
        qq = [_card(Suit.SPADES, Rank.QUEEN, d) for d in (1, 2)]
        jj = [_card(Suit.SPADES, Rank.JACK, d) for d in (1, 2)]
        result = compare_plays(
            aa + kk, qq + jj,
            Suit.HEARTS, Rank.TWO, Suit.SPADES,
        )
        assert result > 0

    def test_compare_plays_empty_lists(self):
        """Empty inputs return 0."""
        c = _card(Suit.SPADES, Rank.ACE)
        assert compare_plays([], [c], Suit.HEARTS, Rank.TWO, Suit.SPADES) == 0
        assert compare_plays([c], [], Suit.HEARTS, Rank.TWO, Suit.SPADES) == 0

    def test_compare_plays_different_lengths(self):
        """Different-length plays return length difference."""
        c = _card(Suit.SPADES, Rank.ACE)
        result = compare_plays([c, c], [c], Suit.HEARTS, Rank.TWO, Suit.SPADES)
        assert result > 0  # longer list wins

    def test_compare_plays_both_off_suit_tie(self):
        """Both non-trump, different suits, neither is lead: tie."""
        spade = _card(Suit.SPADES, Rank.ACE)
        club = _card(Suit.CLUBS, Rank.ACE)
        result = compare_plays(
            [spade], [club],
            Suit.HEARTS, Rank.TWO, Suit.DIAMONDS,
        )
        assert result == 0


class TestNonTrumpOrder:
    def test_non_trump_order_trump_rank_returns_negative(self):
        """Trump rank cards return -1 in non-trump ordering."""
        c = _card(Suit.SPADES, Rank.TWO)
        assert non_trump_order(c, Rank.TWO) == -1

    def test_non_trump_order_normal_card(self):
        """Non-trump cards return RANK_ORDER value."""
        c = _card(Suit.SPADES, Rank.ACE)
        assert non_trump_order(c, Rank.TWO) == 14

    def test_non_trump_order_joker(self):
        """Joker cards return their RANK_ORDER value."""
        c = _card(Suit.JOKER, Rank.BIG_JOKER)
        assert non_trump_order(c, Rank.TWO) == 16


class TestIsEqualInTrump:
    def test_is_equal_in_trump_same_card_type(self):
        """Two cards with same trump_order are equal in trump context."""
        a = _card(Suit.SPADES, Rank.ACE, deck=1)
        b = _card(Suit.SPADES, Rank.ACE, deck=2)
        assert is_equal_in_trump(a, b, Suit.HEARTS, Rank.TWO) is True

    def test_is_equal_in_trump_different_tiers(self):
        """Cards in different trump tiers are not equal."""
        joker = _card(Suit.JOKER, Rank.BIG_JOKER)
        ace = _card(Suit.SPADES, Rank.ACE)
        assert is_equal_in_trump(joker, ace, Suit.HEARTS, Rank.TWO) is False

    def test_is_equal_in_trump_sub_trump_rank_same_order(self):
        """All sub-trump rank cards of same suit have same trump_order."""
        a = _card(Suit.SPADES, Rank.TWO, deck=1)
        b = _card(Suit.SPADES, Rank.TWO, deck=2)
        assert is_equal_in_trump(a, b, Suit.HEARTS, Rank.TWO) is True


class TestCompareCards:
    def test_compare_cards_greater(self):
        ace = _card(Suit.HEARTS, Rank.ACE)
        king = _card(Suit.HEARTS, Rank.KING)
        assert compare_cards(ace, king, Suit.HEARTS, Rank.TWO) > 0

    def test_compare_cards_lesser(self):
        king = _card(Suit.HEARTS, Rank.KING)
        ace = _card(Suit.HEARTS, Rank.ACE)
        assert compare_cards(king, ace, Suit.HEARTS, Rank.TWO) < 0

    def test_compare_cards_equal(self):
        """Same trump_order returns 0."""
        a = _card(Suit.SPADES, Rank.ACE, deck=1)
        b = _card(Suit.SPADES, Rank.ACE, deck=2)
        assert compare_cards(a, b, Suit.HEARTS, Rank.TWO) == 0


class TestSortByTrumpOrder:
    def test_sort_by_trump_order(self):
        cards = [
            _card(Suit.SPADES, Rank.ACE),
            _card(Suit.JOKER, Rank.BIG_JOKER),
            _card(Suit.HEARTS, Rank.KING),
            _card(Suit.JOKER, Rank.SMALL_JOKER),
        ]
        sorted_cards = sort_by_trump_order(cards, Suit.HEARTS, Rank.TWO)
        assert sorted_cards[0].rank == Rank.BIG_JOKER
        assert sorted_cards[1].rank == Rank.SMALL_JOKER
