"""Tests for rules.bidding module."""
import pytest
from server.engine.card import Suit, Rank
from server.engine.types import BidAction, StirAction
from server.rules.bidding import (
    is_valid_bid, get_valid_bid_levels, is_bidding_over, get_winning_bid,
    is_valid_stir, get_valid_stir_options, is_stirring_over, get_next_bidder,
)


class TestIsValidBid:
    def test_is_valid_bid_pass(self):
        assert is_valid_bid(None, True, None, Rank.TWO) is True

    def test_is_valid_bid_first_bid_must_meet_level(self):
        assert is_valid_bid(Rank.TWO, False, None, Rank.TWO) is True
        assert is_valid_bid(Rank.ACE, False, None, Rank.TWO) is True

    def test_is_valid_bid_must_be_higher(self):
        assert is_valid_bid(Rank.FIVE, False, Rank.THREE, Rank.TWO) is True
        assert is_valid_bid(Rank.THREE, False, Rank.THREE, Rank.TWO) is False
        assert is_valid_bid(Rank.TWO, False, Rank.THREE, Rank.TWO) is False

    def test_is_valid_bid_too_low(self):
        assert is_valid_bid(Rank.TWO, False, None, Rank.FIVE) is False


class TestGetValidBidLevels:
    def test_get_valid_bid_levels_initial(self):
        levels = get_valid_bid_levels(None, Rank.FIVE)
        assert Rank.FIVE in levels
        assert Rank.ACE in levels
        assert Rank.TWO not in levels
        assert Rank.THREE not in levels

    def test_get_valid_bid_levels_after_bid(self):
        levels = get_valid_bid_levels(Rank.SEVEN, Rank.TWO)
        assert Rank.EIGHT in levels
        assert Rank.SEVEN not in levels


class TestIsBiddingOver:
    def test_is_bidding_over_no_bids(self):
        assert is_bidding_over([], 4) is False

    def test_is_bidding_over_all_pass(self):
        bids = [
            BidAction(player_index=i, level=None, pass_=True)
            for i in range(4)
        ]
        assert is_bidding_over(bids, 4) is True

    def test_is_bidding_over_three_consecutive_passes(self):
        bids = [
            BidAction(player_index=0, level=Rank.THREE, pass_=False),
            BidAction(player_index=1, level=None, pass_=True),
            BidAction(player_index=2, level=None, pass_=True),
            BidAction(player_index=3, level=None, pass_=True),
        ]
        assert is_bidding_over(bids, 4) is True

    def test_is_bidding_over_still_going(self):
        bids = [
            BidAction(player_index=0, level=Rank.THREE, pass_=False),
            BidAction(player_index=1, level=None, pass_=True),
        ]
        assert is_bidding_over(bids, 4) is False


class TestGetWinningBid:
    def test_get_winning_bid(self):
        bids = [
            BidAction(player_index=0, level=Rank.THREE, pass_=False),
            BidAction(player_index=1, level=Rank.FIVE, pass_=False),
        ]
        winner = get_winning_bid(bids)
        assert winner is not None
        assert winner.player_index == 1

    def test_get_winning_bid_all_pass(self):
        bids = [
            BidAction(player_index=0, level=None, pass_=True),
            BidAction(player_index=1, level=None, pass_=True),
        ]
        winner = get_winning_bid(bids)
        assert winner is None


class TestIsValidStir:
    def test_is_valid_stir_same_level_different_suit(self):
        stir = StirAction(player_index=1, new_trump_suit=Suit.SPADES, level=Rank.THREE)
        assert is_valid_stir(stir, Suit.HEARTS, Rank.THREE, [], 1) is True

    def test_is_valid_stir_same_level_same_suit_invalid(self):
        stir = StirAction(player_index=1, new_trump_suit=Suit.HEARTS, level=Rank.THREE)
        assert is_valid_stir(stir, Suit.HEARTS, Rank.THREE, [], 1) is False

    def test_is_valid_stir_higher_level(self):
        stir = StirAction(player_index=1, new_trump_suit=Suit.SPADES, level=Rank.FIVE)
        assert is_valid_stir(stir, Suit.HEARTS, Rank.THREE, [], 1) is True

    def test_is_valid_stir_consecutive_same_player_invalid(self):
        """Bug #5 fix: same person cannot stir consecutively."""
        prev_stir = StirAction(player_index=1, new_trump_suit=Suit.SPADES, level=Rank.THREE)
        stir = StirAction(player_index=1, new_trump_suit=Suit.DIAMONDS, level=Rank.FIVE)
        assert is_valid_stir(stir, Suit.SPADES, Rank.THREE, [prev_stir], 1) is False

    def test_is_valid_stir_different_player_after_consecutive(self):
        """Different player CAN stir after someone else stirred."""
        prev_stir = StirAction(player_index=0, new_trump_suit=Suit.SPADES, level=Rank.THREE)
        stir = StirAction(player_index=1, new_trump_suit=Suit.DIAMONDS, level=Rank.FIVE)
        assert is_valid_stir(stir, Suit.SPADES, Rank.THREE, [prev_stir], 1) is True

    def test_is_valid_stir_lower_level_invalid(self):
        stir = StirAction(player_index=1, new_trump_suit=Suit.SPADES, level=Rank.TWO)
        assert is_valid_stir(stir, Suit.HEARTS, Rank.THREE, [], 1) is False


class TestGetValidStirOptions:
    def test_get_valid_stir_options(self):
        options = get_valid_stir_options(Suit.HEARTS, Rank.THREE, 0, [])
        assert len(options) > 0
        same_level = [o for o in options if o.level == Rank.THREE]
        for o in same_level:
            assert o.new_trump_suit != Suit.HEARTS
        higher = [o for o in options if o.level != Rank.THREE]
        assert len(higher) > 0


class TestIsStirringOver:
    def test_is_stirring_over(self):
        assert is_stirring_over(4, 4) is True
        assert is_stirring_over(3, 4) is False


class TestGetNextBidder:
    def test_get_next_bidder(self):
        assert get_next_bidder(0) == 2
        assert get_next_bidder(1) == 0
        assert get_next_bidder(2) == 3
        assert get_next_bidder(3) == 1
