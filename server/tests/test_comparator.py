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
        assert 60 <= order < 80  # Trump suit, non-trump-rank

    def test_trump_order_non_trump(self):
        c = _card(Suit.SPADES, Rank.ACE)
        order = trump_order(c, Suit.HEARTS, Rank.TWO)
        assert order < 60  # Non-trump card


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


class TestCompareCards:
    def test_compare_cards_greater(self):
        ace = _card(Suit.HEARTS, Rank.ACE)
        king = _card(Suit.HEARTS, Rank.KING)
        assert compare_cards(ace, king, Suit.HEARTS, Rank.TWO) > 0

    def test_compare_cards_lesser(self):
        king = _card(Suit.HEARTS, Rank.KING)
        ace = _card(Suit.HEARTS, Rank.ACE)
        assert compare_cards(king, ace, Suit.HEARTS, Rank.TWO) < 0


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
