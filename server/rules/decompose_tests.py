"""Tests for rules.decompose public interface."""

from typing import Literal

from server.rules.cards import Card, POINTS_MAP, Suit, Rank
from server.rules.decompose import decompose


def _card(suit: Suit, rank: Rank, deck: Literal[1, 2] = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit, rank=rank,
        points=POINTS_MAP[rank],
    )


class TestDecompose:
    # --- Singles ---
    def test_decompose_single_card(self) -> None:
        """Single card -> one SubPlay with pair_count=0."""
        c = _card(Suit.HEARTS, Rank.ACE)
        subs = decompose([c], Suit.SPADES, Rank.TWO)
        assert len(subs) == 1
        assert subs[0].pair_count == 0
        assert subs[0].cards == [c]
        assert subs[0].suit == Suit.HEARTS

    def test_decompose_two_different_singles(self) -> None:
        """Two different cards -> two SubPlay singles."""
        c1 = _card(Suit.HEARTS, Rank.ACE)
        c2 = _card(Suit.HEARTS, Rank.KING)
        subs = decompose([c1, c2], Suit.SPADES, Rank.TWO)
        assert len(subs) == 2
        assert all(s.pair_count == 0 for s in subs)

    # --- Pairs ---
    def test_decompose_pair(self) -> None:
        """Two same-rank same-suit cards -> one SubPlay pair."""
        c1 = _card(Suit.HEARTS, Rank.ACE, 1)
        c2 = _card(Suit.HEARTS, Rank.ACE, 2)
        subs = decompose([c1, c2], Suit.SPADES, Rank.TWO)
        assert len(subs) == 1
        assert subs[0].pair_count == 1
        assert len(subs[0].cards) == 2

    def test_decompose_pair_plus_single(self) -> None:
        """Pair + extra single -> one pair SubPlay + one single SubPlay."""
        c1 = _card(Suit.HEARTS, Rank.ACE, 1)
        c2 = _card(Suit.HEARTS, Rank.ACE, 2)
        c3 = _card(Suit.HEARTS, Rank.KING, 1)
        subs = decompose([c1, c2, c3], Suit.SPADES, Rank.TWO)
        pair_subs = [s for s in subs if s.pair_count == 1]
        single_subs = [s for s in subs if s.pair_count == 0]
        assert len(pair_subs) == 1
        assert len(single_subs) == 1

    # --- Tractors ---
    def test_decompose_tractor_2_pairs(self) -> None:
        """Two consecutive pairs -> one tractor SubPlay (pair_count=2)."""
        cards = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
        ]
        subs = decompose(cards, Suit.SPADES, Rank.TWO)
        assert len(subs) == 1
        assert subs[0].pair_count == 2
        assert len(subs[0].cards) == 4

    def test_decompose_tractor_3_pairs(self) -> None:
        """Three consecutive pairs -> one tractor SubPlay (pair_count=3)."""
        cards = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
            _card(Suit.HEARTS, Rank.FIVE, 1), _card(Suit.HEARTS, Rank.FIVE, 2),
        ]
        subs = decompose(cards, Suit.SPADES, Rank.TWO)
        assert len(subs) == 1
        assert subs[0].pair_count == 3

    def test_decompose_tractor_skips_trump_rank(self) -> None:
        """Non-trump suit tractor: trump_rank is skipped in consecutive check.

        trump_rank=5: 4-4 + 6-6 are consecutive (5 is skipped).
        """
        cards = [
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
            _card(Suit.HEARTS, Rank.SIX, 1), _card(Suit.HEARTS, Rank.SIX, 2),
        ]
        subs = decompose(cards, Suit.SPADES, Rank.FIVE)
        assert len(subs) == 1
        assert subs[0].pair_count == 2

    def test_decompose_tractor_non_consecutive_pairs(self) -> None:
        """Non-consecutive pairs -> separate pair SubPlays, not a tractor."""
        cards = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2),
        ]
        subs = decompose(cards, Suit.SPADES, Rank.TWO)
        pair_subs = [s for s in subs if s.pair_count >= 1]
        assert len(pair_subs) == 2
        assert all(s.pair_count == 1 for s in pair_subs)

    # --- Mixed ---
    def test_decompose_tractor_plus_pair_plus_singles(self) -> None:
        """spA spK sp10-10 sp7-7-6-6 -> [single spA, single spK, pair sp10-10, tractor sp7-7-6-6]."""
        cards = [
            _card(Suit.SPADES, Rank.ACE),
            _card(Suit.SPADES, Rank.KING),
            _card(Suit.SPADES, Rank.TEN, 1), _card(Suit.SPADES, Rank.TEN, 2),
            _card(Suit.SPADES, Rank.SEVEN, 1), _card(Suit.SPADES, Rank.SEVEN, 2),
            _card(Suit.SPADES, Rank.SIX, 1), _card(Suit.SPADES, Rank.SIX, 2),
        ]
        subs = decompose(cards, Suit.HEARTS, Rank.TWO)
        tractor_subs = [s for s in subs if s.pair_count >= 2]
        pair_subs = [s for s in subs if s.pair_count == 1]
        single_subs = [s for s in subs if s.pair_count == 0]
        assert len(tractor_subs) == 1
        assert tractor_subs[0].pair_count == 2
        assert len(pair_subs) == 1
        assert len(single_subs) == 2

    def test_decompose_pair_and_tractor_and_singles(self) -> None:
        """spA spA spK spQ spQ sp9 sp9 sp8 sp8 -> [pair spA-A, single spK, pair spQ-Q, tractor sp9-9-8-8]."""
        cards = [
            _card(Suit.SPADES, Rank.ACE, 1), _card(Suit.SPADES, Rank.ACE, 2),
            _card(Suit.SPADES, Rank.KING),
            _card(Suit.SPADES, Rank.QUEEN, 1), _card(Suit.SPADES, Rank.QUEEN, 2),
            _card(Suit.SPADES, Rank.NINE, 1), _card(Suit.SPADES, Rank.NINE, 2),
            _card(Suit.SPADES, Rank.EIGHT, 1), _card(Suit.SPADES, Rank.EIGHT, 2),
        ]
        subs = decompose(cards, Suit.HEARTS, Rank.TWO)
        tractor_subs = [s for s in subs if s.pair_count >= 2]
        pair_subs = [s for s in subs if s.pair_count == 1]
        single_subs = [s for s in subs if s.pair_count == 0]
        assert len(tractor_subs) == 1
        assert tractor_subs[0].pair_count == 2
        assert len(pair_subs) == 2
        assert len(single_subs) == 1

    # --- Trump group ---
    def test_decompose_trump_group_uses_trump_rank_order(self) -> None:
        """Trump group: tractor detection uses trump_rank_order, not non_trump_rank_order.

        With heart trump, rank=5: h3 h3 h4 h4 -> tractor (consecutive in trump ordering).
        """
        cards = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
        ]
        subs = decompose(cards, Suit.HEARTS, Rank.FIVE)
        assert len(subs) == 1
        assert subs[0].pair_count == 2

    def test_decompose_trump_group_cross_sub_type_tractor(self) -> None:
        """Trump group cross-sub-type tractor: hA hA + d5 d5 is tractor when heart trump, rank=5.

        hA is at position 45+14=59, d5 is at 70. They are adjacent in the
        trump_rank_order sequence because no trump card has a position value between
        59 and 70. Adjacent means "consecutive in the sorted list of position values"
        -- not "position values differ by 1".
        """
        cards = [
            _card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.DIAMONDS, Rank.FIVE, 1), _card(Suit.DIAMONDS, Rank.FIVE, 2),
        ]
        subs = decompose(cards, Suit.HEARTS, Rank.FIVE)
        assert len(subs) == 1
        assert subs[0].pair_count == 2, f"Expected tractor (pair_count=2), got {subs[0].pair_count}"

    def test_decompose_trump_group_joker_tractor(self) -> None:
        """Small joker pair + big joker pair -> tractor (adjacent in trump ordering)."""
        cards = [
            _card(Suit.JOKER, Rank.SMALL_JOKER, 1), _card(Suit.JOKER, Rank.SMALL_JOKER, 2),
            _card(Suit.JOKER, Rank.BIG_JOKER, 1), _card(Suit.JOKER, Rank.BIG_JOKER, 2),
        ]
        subs = decompose(cards, Suit.HEARTS, Rank.TWO)
        assert len(subs) == 1
        assert subs[0].pair_count == 2

    def test_decompose_trump_group_suit_specific_rank_pairs(self) -> None:
        """c5 c5 + sp5 sp5 -> tractor by same-rank suit structure.

        This structural tractor rule still uses SUIT_OFFSET to decide same-rank
        suit adjacency. It does not mean c5 and sp5 have different trick-winning
        strength; they are equal during comparison.
        """
        cards = [
            _card(Suit.CLUBS, Rank.FIVE, 1), _card(Suit.CLUBS, Rank.FIVE, 2),
            _card(Suit.SPADES, Rank.FIVE, 1), _card(Suit.SPADES, Rank.FIVE, 2),
        ]
        subs = decompose(cards, Suit.HEARTS, Rank.FIVE)
        assert len(subs) == 1
        assert subs[0].pair_count == 2

    # --- Bug regression ---
    def test_decompose_trump_single_no_duplicate(self) -> None:
        """Regression: trump group with single card must not produce duplicate SubPlay entries."""
        c = _card(Suit.HEARTS, Rank.THREE)
        subs = decompose([c], Suit.HEARTS, Rank.FIVE)
        assert len(subs) == 1
        assert subs[0].pair_count == 0
        assert subs[0].cards == [c]

    # --- Edge cases ---
    def test_decompose_empty(self) -> None:
        """Empty cards -> empty list."""
        subs = decompose([], Suit.SPADES, Rank.TWO)
        assert subs == []

    def test_decompose_four_of_a_kind(self) -> None:
        """4 cards of same rank in trump group -> 2 separate pairs (same rank, not tractor)."""
        cards = [
            _card(Suit.DIAMONDS, Rank.TWO, 1), _card(Suit.DIAMONDS, Rank.TWO, 2),
            _card(Suit.CLUBS, Rank.TWO, 1), _card(Suit.CLUBS, Rank.TWO, 2),
        ]
        subs = decompose(cards, Suit.HEARTS, Rank.TWO)
        pair_subs = [s for s in subs if s.pair_count >= 1]
        assert len(pair_subs) == 2
        assert all(s.pair_count == 1 for s in pair_subs)

    def test_decompose_longest_tractor_first(self) -> None:
        """Longest tractor extracted first: 3 pairs -> tractor(3), not tractor(2)+pair."""
        cards = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
            _card(Suit.HEARTS, Rank.FIVE, 1), _card(Suit.HEARTS, Rank.FIVE, 2),
        ]
        subs = decompose(cards, Suit.SPADES, Rank.TWO)
        tractor_subs = [s for s in subs if s.pair_count >= 2]
        assert len(tractor_subs) == 1
        assert tractor_subs[0].pair_count == 3
