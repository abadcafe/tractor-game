"""Tests for engine.card_utils module."""
import pytest
from server.engine.card import Card, Suit, Rank
from server.engine.card_utils import (
    natural_rank, is_pair, group_by_suit, sort_hand,
    RANK_ORDER, SUITED_RANKS, SUITS, POINTS_MAP, TOTAL_POINTS,
)


class TestNaturalRank:
    def test_natural_rank_values(self):
        assert natural_rank(Rank.TWO) == 2
        assert natural_rank(Rank.ACE) == 14
        assert natural_rank(Rank.SMALL_JOKER) == 15
        assert natural_rank(Rank.BIG_JOKER) == 16


class TestIsPair:
    def test_is_pair_true(self):
        a = Card(id="D1-hearts-A", suit=Suit.HEARTS, rank=Rank.ACE,
                 is_joker=False, is_big_joker=False, points=0, deck=1)
        b = Card(id="D2-hearts-A", suit=Suit.HEARTS, rank=Rank.ACE,
                 is_joker=False, is_big_joker=False, points=0, deck=2)
        assert is_pair(a, b) is True

    def test_is_pair_different_suit(self):
        a = Card(id="D1-hearts-A", suit=Suit.HEARTS, rank=Rank.ACE,
                 is_joker=False, is_big_joker=False, points=0, deck=1)
        b = Card(id="D1-spades-A", suit=Suit.SPADES, rank=Rank.ACE,
                 is_joker=False, is_big_joker=False, points=0, deck=1)
        assert is_pair(a, b) is False

    def test_is_pair_different_rank(self):
        a = Card(id="D1-hearts-A", suit=Suit.HEARTS, rank=Rank.ACE,
                 is_joker=False, is_big_joker=False, points=0, deck=1)
        b = Card(id="D1-hearts-K", suit=Suit.HEARTS, rank=Rank.KING,
                 is_joker=False, is_big_joker=False, points=0, deck=1)
        assert is_pair(a, b) is False

    def test_is_pair_same_card(self):
        a = Card(id="D1-hearts-A", suit=Suit.HEARTS, rank=Rank.ACE,
                 is_joker=False, is_big_joker=False, points=0, deck=1)
        assert is_pair(a, a) is False

    def test_is_pair_joker_pair(self):
        """Two small jokers from different decks form a pair."""
        a = Card(id="D1-joker-SJ", suit=Suit.JOKER, rank=Rank.SMALL_JOKER,
                 is_joker=True, is_big_joker=False, points=0, deck=1)
        b = Card(id="D2-joker-SJ", suit=Suit.JOKER, rank=Rank.SMALL_JOKER,
                 is_joker=True, is_big_joker=False, points=0, deck=2)
        assert is_pair(a, b) is True

    def test_is_pair_joker_not_pair(self):
        """Small joker and big joker are NOT a pair (different rank)."""
        a = Card(id="D1-joker-SJ", suit=Suit.JOKER, rank=Rank.SMALL_JOKER,
                 is_joker=True, is_big_joker=False, points=0, deck=1)
        b = Card(id="D1-joker-BJ", suit=Suit.JOKER, rank=Rank.BIG_JOKER,
                 is_joker=True, is_big_joker=True, points=0, deck=1)
        assert is_pair(a, b) is False


class TestGroupBySuit:
    def test_group_by_suit(self):
        cards = [
            Card(id="D1-hearts-A", suit=Suit.HEARTS, rank=Rank.ACE,
                 is_joker=False, is_big_joker=False, points=0, deck=1),
            Card(id="D1-spades-K", suit=Suit.SPADES, rank=Rank.KING,
                 is_joker=False, is_big_joker=False, points=0, deck=1),
            Card(id="D2-hearts-K", suit=Suit.HEARTS, rank=Rank.KING,
                 is_joker=False, is_big_joker=False, points=0, deck=2),
        ]
        groups = group_by_suit(cards)
        assert len(groups[Suit.HEARTS]) == 2
        assert len(groups[Suit.SPADES]) == 1

    def test_group_by_suit_includes_jokers(self):
        """Jokers should appear under the JOKER suit key."""
        cards = [
            Card(id="D1-joker-SJ", suit=Suit.JOKER, rank=Rank.SMALL_JOKER,
                 is_joker=True, is_big_joker=False, points=0, deck=1),
            Card(id="D2-joker-BJ", suit=Suit.JOKER, rank=Rank.BIG_JOKER,
                 is_joker=True, is_big_joker=True, points=0, deck=2),
            Card(id="D1-hearts-A", suit=Suit.HEARTS, rank=Rank.ACE,
                 is_joker=False, is_big_joker=False, points=0, deck=1),
        ]
        groups = group_by_suit(cards)
        assert len(groups[Suit.JOKER]) == 2
        assert len(groups[Suit.HEARTS]) == 1

    def test_group_by_suit_empty(self):
        """group_by_suit([]) should return an empty dict."""
        assert group_by_suit([]) == {}


class TestSortHand:
    def test_sort_hand_by_suit(self):
        cards = [
            Card(id="D1-spades-K", suit=Suit.SPADES, rank=Rank.KING,
                 is_joker=False, is_big_joker=False, points=0, deck=1),
            Card(id="D1-hearts-A", suit=Suit.HEARTS, rank=Rank.ACE,
                 is_joker=False, is_big_joker=False, points=0, deck=1),
            Card(id="D2-hearts-K", suit=Suit.HEARTS, rank=Rank.KING,
                 is_joker=False, is_big_joker=False, points=0, deck=2),
        ]
        sorted_cards = sort_hand(cards)
        # Hearts should come before spades in sort order
        suit_order = [c.suit for c in sorted_cards]
        # Verify grouped: no suit should appear non-contiguously
        seen = set()
        prev = None
        for s in suit_order:
            if s != prev:
                assert s not in seen
                seen.add(s)
                prev = s

    def test_sort_hand_descending_rank(self):
        cards = [
            Card(id="D1-hearts-K", suit=Suit.HEARTS, rank=Rank.KING,
                 is_joker=False, is_big_joker=False, points=0, deck=1),
            Card(id="D1-hearts-A", suit=Suit.HEARTS, rank=Rank.ACE,
                 is_joker=False, is_big_joker=False, points=0, deck=1),
        ]
        sorted_cards = sort_hand(cards)
        assert sorted_cards[0].rank == Rank.ACE
        assert sorted_cards[1].rank == Rank.KING

    def test_sort_hand_jokers_first(self):
        """Jokers should sort before any suited card."""
        cards = [
            Card(id="D1-hearts-A", suit=Suit.HEARTS, rank=Rank.ACE,
                 is_joker=False, is_big_joker=False, points=0, deck=1),
            Card(id="D1-joker-BJ", suit=Suit.JOKER, rank=Rank.BIG_JOKER,
                 is_joker=True, is_big_joker=True, points=0, deck=1),
            Card(id="D1-spades-K", suit=Suit.SPADES, rank=Rank.KING,
                 is_joker=False, is_big_joker=False, points=0, deck=1),
        ]
        sorted_cards = sort_hand(cards)
        assert sorted_cards[0].suit == Suit.JOKER

    def test_sort_hand_empty(self):
        """sort_hand([]) should return an empty list."""
        assert sort_hand([]) == []


class TestTotalPoints:
    def test_total_points(self):
        assert TOTAL_POINTS == 200
