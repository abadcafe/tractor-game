"""Tests for sm.comparator module."""
from typing import Literal

from server.sm.card_model import Card, Suit, Rank
from server.sm.comparator import (
    trump_order, effective_suit, compare_plays, sort_by_trump_order,
    is_trump_card, bid_value, trump_rank_order,
)


def _card(suit: Suit, rank: Rank, deck: Literal[1, 2] = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit, rank=rank,
        is_joker=(suit == Suit.JOKER),
        is_big_joker=(rank == Rank.BIG_JOKER),
        points=0, deck=deck,
    )


class TestTrumpOrder:
    def test_trump_order_big_joker(self) -> None:
        """Big Joker = 100."""
        c = _card(Suit.JOKER, Rank.BIG_JOKER)
        assert trump_order(c, Suit.HEARTS, Rank.TWO) == 100

    def test_trump_order_small_joker(self) -> None:
        """Small Joker = 90."""
        c = _card(Suit.JOKER, Rank.SMALL_JOKER)
        assert trump_order(c, Suit.HEARTS, Rank.TWO) == 90

    def test_trump_order_trump_rank_trump_suit(self) -> None:
        """Trump rank + trump suit = 80."""
        c = _card(Suit.HEARTS, Rank.TWO)
        assert trump_order(c, Suit.HEARTS, Rank.TWO) == 80

    def test_trump_order_trump_rank_other_suit(self) -> None:
        """Trump rank + other suit = 70 + suit_offset."""
        c = _card(Suit.SPADES, Rank.TWO)
        # SPADES offset = 3
        assert trump_order(c, Suit.HEARTS, Rank.TWO) == 73

    def test_trump_order_trump_suit_non_rank(self) -> None:
        """Trump suit non-rank = 45 + RANK_ORDER."""
        c = _card(Suit.HEARTS, Rank.ACE)
        # ACE RANK_ORDER = 14
        assert trump_order(c, Suit.HEARTS, Rank.TWO) == 45 + 14

    def test_trump_order_non_trump(self) -> None:
        """Non-trump suit card = RANK_ORDER - 2."""
        c = _card(Suit.SPADES, Rank.ACE)
        # ACE = 14, minus 2 = 12
        assert trump_order(c, Suit.HEARTS, Rank.TWO) == 12

    def test_trump_order_no_trump_suit(self) -> None:
        """When trump_suit is None, only jokers and trump_rank are trump."""
        bj = _card(Suit.JOKER, Rank.BIG_JOKER)
        sj = _card(Suit.JOKER, Rank.SMALL_JOKER)
        tr = _card(Suit.HEARTS, Rank.TWO)
        non_trump = _card(Suit.HEARTS, Rank.ACE)
        # Big/Small joker and trump rank cards still get high order
        assert trump_order(bj, None, Rank.TWO) == 100
        assert trump_order(sj, None, Rank.TWO) == 90
        assert trump_order(tr, None, Rank.TWO) >= 70  # trump rank
        # Non-trump cards get low order
        assert trump_order(non_trump, None, Rank.TWO) < 50


class TestEffectiveSuit:
    def test_effective_suit_trump_joker(self) -> None:
        """Jokers are always trump."""
        c = _card(Suit.JOKER, Rank.BIG_JOKER)
        assert effective_suit(c, Suit.HEARTS, Rank.TWO) == "trump"

    def test_effective_suit_trump_rank(self) -> None:
        """Trump rank cards are always trump."""
        c = _card(Suit.SPADES, Rank.TWO)
        assert effective_suit(c, Suit.HEARTS, Rank.TWO) == "trump"

    def test_effective_suit_trump_suit(self) -> None:
        """Trump suit cards are trump."""
        c = _card(Suit.HEARTS, Rank.ACE)
        assert effective_suit(c, Suit.HEARTS, Rank.TWO) == "trump"

    def test_effective_suit_non_trump(self) -> None:
        """Non-trump cards keep their suit."""
        c = _card(Suit.SPADES, Rank.ACE)
        assert effective_suit(c, Suit.HEARTS, Rank.TWO) == Suit.SPADES

    def test_effective_suit_none_trump_suit(self) -> None:
        """When trump_suit=None, only jokers and trump rank are trump."""
        c = _card(Suit.HEARTS, Rank.ACE)
        assert effective_suit(c, None, Rank.TWO) == Suit.HEARTS


class TestComparePlays:
    def test_compare_plays_trump_beats_non_trump(self) -> None:
        """Any trump play beats any non-trump play."""
        trump_card = [_card(Suit.HEARTS, Rank.TWO)]
        non_trump_card = [_card(Suit.SPADES, Rank.ACE)]
        assert compare_plays(trump_card, non_trump_card, Suit.HEARTS, Rank.TWO, Suit.SPADES) > 0

    def test_compare_plays_both_trump(self) -> None:
        """Higher trump order wins among trump plays."""
        big_joker = [_card(Suit.JOKER, Rank.BIG_JOKER)]
        small_joker = [_card(Suit.JOKER, Rank.SMALL_JOKER)]
        assert compare_plays(big_joker, small_joker, Suit.HEARTS, Rank.TWO, None) > 0

    def test_compare_plays_same_suit(self) -> None:
        """Same non-trump suit: higher rank wins."""
        ace = [_card(Suit.SPADES, Rank.ACE)]
        king = [_card(Suit.SPADES, Rank.KING)]
        assert compare_plays(ace, king, Suit.HEARTS, Rank.TWO, Suit.SPADES) > 0

    def test_compare_plays_different_suit_lead_wins(self) -> None:
        """Different non-trump suits: lead suit wins."""
        spade = [_card(Suit.SPADES, Rank.ACE)]
        diamond = [_card(Suit.DIAMONDS, Rank.ACE)]
        assert compare_plays(spade, diamond, Suit.HEARTS, Rank.TWO, Suit.SPADES) > 0

    def test_compare_plays_different_suit_off_suit(self) -> None:
        """Different non-trump suits: non-lead suit loses."""
        spade = [_card(Suit.SPADES, Rank.ACE)]
        diamond = [_card(Suit.DIAMONDS, Rank.ACE)]
        assert compare_plays(diamond, spade, Suit.HEARTS, Rank.TWO, Suit.SPADES) < 0


class TestIsTrumpCard:
    def test_is_trump_card_joker(self) -> None:
        c = _card(Suit.JOKER, Rank.BIG_JOKER)
        assert is_trump_card(c, Suit.HEARTS, Rank.TWO) is True

    def test_is_trump_card_trump_rank(self) -> None:
        c = _card(Suit.SPADES, Rank.TWO)
        assert is_trump_card(c, Suit.HEARTS, Rank.TWO) is True

    def test_is_trump_card_trump_suit(self) -> None:
        c = _card(Suit.HEARTS, Rank.ACE)
        assert is_trump_card(c, Suit.HEARTS, Rank.TWO) is True

    def test_is_trump_card_non_trump(self) -> None:
        c = _card(Suit.SPADES, Rank.ACE)
        assert is_trump_card(c, Suit.HEARTS, Rank.TWO) is False


class TestBidValue:
    def test_bid_value_single_diamond(self) -> None:
        """Single ♦ rank = 1*100 + 0 = 100."""
        c = _card(Suit.DIAMONDS, Rank.TWO)
        assert bid_value(cards=[c], trump_rank=Rank.TWO) == 100

    def test_bid_value_single_spade(self) -> None:
        """Single ♠ rank = 1*100 + 3 = 103."""
        c = _card(Suit.SPADES, Rank.TWO)
        assert bid_value(cards=[c], trump_rank=Rank.TWO) == 103

    def test_bid_value_pair_heart(self) -> None:
        """Pair ♥ rank = 2*100 + 2 = 202."""
        c1 = _card(Suit.HEARTS, Rank.TWO, 1)
        c2 = _card(Suit.HEARTS, Rank.TWO, 2)
        assert bid_value(cards=[c1, c2], trump_rank=Rank.TWO) == 202

    def test_bid_value_pair_big_joker(self) -> None:
        """Pair big joker = 2*100 + 5 = 205."""
        c1 = _card(Suit.JOKER, Rank.BIG_JOKER, 1)
        c2 = _card(Suit.JOKER, Rank.BIG_JOKER, 2)
        assert bid_value(cards=[c1, c2], trump_rank=Rank.TWO) == 205

    def test_bid_value_single_joker_invalid(self) -> None:
        """Single joker cannot bid — returns 0 (invalid)."""
        c = _card(Suit.JOKER, Rank.BIG_JOKER)
        assert bid_value(cards=[c], trump_rank=Rank.TWO) == 0

    def test_bid_value_non_trump_rank_invalid(self) -> None:
        """Non-trump-rank card cannot bid — returns 0."""
        c = _card(Suit.HEARTS, Rank.THREE)
        assert bid_value(cards=[c], trump_rank=Rank.TWO) == 0


class TestSortByTrumpOrder:
    def test_sort_by_trump_order(self) -> None:
        """Cards sorted in descending trump order."""
        cards = [
            _card(Suit.SPADES, Rank.ACE),
            _card(Suit.HEARTS, Rank.TWO),
            _card(Suit.JOKER, Rank.BIG_JOKER),
        ]
        sorted_cards = sort_by_trump_order(cards, Suit.HEARTS, Rank.TWO)
        assert sorted_cards[0].rank == Rank.BIG_JOKER
        assert sorted_cards[1].rank == Rank.TWO
        assert sorted_cards[2].rank == Rank.ACE


from server.sm.comparator import trump_rank_order


class TestSUITOffset:
    def test_suit_offset_matches_spec(self) -> None:
        """SUIT_OFFSET must match spec section 2.3: diamond=0, club=1, heart=2, spade=3."""
        from server.sm.comparator import SUIT_OFFSET
        assert SUIT_OFFSET[Suit.DIAMONDS] == 0
        assert SUIT_OFFSET[Suit.CLUBS] == 1
        assert SUIT_OFFSET[Suit.HEARTS] == 2
        assert SUIT_OFFSET[Suit.SPADES] == 3


class TestTrumpRankOrder:
    def test_trump_rank_order_big_joker(self) -> None:
        """Big Joker = 100 in trump_rank_order."""
        c = _card(Suit.JOKER, Rank.BIG_JOKER)
        assert trump_rank_order(c, Suit.HEARTS, Rank.TWO) == 100

    def test_trump_rank_order_small_joker(self) -> None:
        """Small Joker = 90 in trump_rank_order."""
        c = _card(Suit.JOKER, Rank.SMALL_JOKER)
        assert trump_rank_order(c, Suit.HEARTS, Rank.TWO) == 90

    def test_trump_rank_order_trump_rank_trump_suit(self) -> None:
        """Trump rank in trump suit = 80."""
        c = _card(Suit.HEARTS, Rank.TWO)
        assert trump_rank_order(c, Suit.HEARTS, Rank.TWO) == 80

    def test_trump_rank_order_trump_rank_other_suit_diamonds(self) -> None:
        """Trump rank in diamond = 70 + 0 = 70."""
        c = _card(Suit.DIAMONDS, Rank.TWO)
        assert trump_rank_order(c, Suit.HEARTS, Rank.TWO) == 70

    def test_trump_rank_order_trump_rank_other_suit_clubs(self) -> None:
        """Trump rank in club = 70 + 1 = 71."""
        c = _card(Suit.CLUBS, Rank.TWO)
        assert trump_rank_order(c, Suit.HEARTS, Rank.TWO) == 71

    def test_trump_rank_order_trump_rank_other_suit_spades(self) -> None:
        """Trump rank in spade = 70 + 3 = 73."""
        c = _card(Suit.SPADES, Rank.TWO)
        assert trump_rank_order(c, Suit.HEARTS, Rank.TWO) == 73

    def test_trump_rank_order_trump_suit_non_rank_low(self) -> None:
        """Trump suit non-rank low card: 45 + RANK_ORDER[3] = 45 + 3 = 48."""
        c = _card(Suit.HEARTS, Rank.THREE)
        assert trump_rank_order(c, Suit.HEARTS, Rank.TWO) == 48

    def test_trump_rank_order_trump_suit_non_rank_ace(self) -> None:
        """Trump suit non-rank ACE: 45 + RANK_ORDER[14] = 45 + 14 = 59."""
        c = _card(Suit.HEARTS, Rank.ACE)
        assert trump_rank_order(c, Suit.HEARTS, Rank.TWO) == 59

    def test_trump_rank_order_non_trump_fallback(self) -> None:
        """Non-trump card fallback: RANK_ORDER - 2."""
        c = _card(Suit.SPADES, Rank.ACE)
        # ACE RANK_ORDER = 14, minus 2 = 12
        assert trump_rank_order(c, Suit.HEARTS, Rank.TWO) == 12

    def test_trump_rank_order_full_sequence(self) -> None:
        """Full trump ordering for heart trump, rank=5: h3<h4<h6<...<hA<d5<c5<h5<SJ<BJ."""
        ts, tr = Suit.HEARTS, Rank.FIVE
        h3 = _card(Suit.HEARTS, Rank.THREE)
        h4 = _card(Suit.HEARTS, Rank.FOUR)
        h6 = _card(Suit.HEARTS, Rank.SIX)
        hA = _card(Suit.HEARTS, Rank.ACE)
        d5 = _card(Suit.DIAMONDS, Rank.FIVE)
        c5 = _card(Suit.CLUBS, Rank.FIVE)
        h5 = _card(Suit.HEARTS, Rank.FIVE)
        sj = _card(Suit.JOKER, Rank.SMALL_JOKER)
        bj = _card(Suit.JOKER, Rank.BIG_JOKER)

        vals = [
            trump_rank_order(h3, ts, tr),
            trump_rank_order(h4, ts, tr),
            trump_rank_order(h6, ts, tr),
            trump_rank_order(hA, ts, tr),
            trump_rank_order(d5, ts, tr),
            trump_rank_order(c5, ts, tr),
            trump_rank_order(h5, ts, tr),
            trump_rank_order(sj, ts, tr),
            trump_rank_order(bj, ts, tr),
        ]
        # Must be strictly increasing
        for i in range(1, len(vals)):
            assert vals[i] > vals[i - 1], f"Order violation at index {i}: {vals[i-1]} vs {vals[i]}"

    def test_trump_rank_order_no_trump_suit(self) -> None:
        """When trump_suit=None, trump rank cards still get 70+suit_offset."""
        c = _card(Suit.DIAMONDS, Rank.TWO)
        assert trump_rank_order(c, None, Rank.TWO) == 70
        c2 = _card(Suit.SPADES, Rank.TWO)
        assert trump_rank_order(c2, None, Rank.TWO) == 73
