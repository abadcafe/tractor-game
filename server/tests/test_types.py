"""Tests for engine.types module."""
import pytest
from server.engine.types import Phase, PlayType, PlayAction, BidAction, StirAction
from server.engine.card import Suit, Rank, Card


class TestPhaseEnum:
    def test_phase_enum_values(self):
        assert Phase.DEALING == "dealing"
        assert Phase.BIDDING == "bidding"
        assert Phase.STIRRING == "stirring"
        assert Phase.EXCHANGE == "exchange"
        assert Phase.PLAYING == "playing"
        assert Phase.SCORING == "scoring"
        assert Phase.GAME_OVER == "game_over"


class TestPlayTypeEnum:
    def test_play_type_enum_values(self):
        assert PlayType.SINGLE == "single"
        assert PlayType.PAIR == "pair"
        assert PlayType.TRACTOR == "tractor"
        assert PlayType.THROW == "throw"


class TestPlayAction:
    def test_play_action_creation(self):
        c1 = Card(id="D1-hearts-A", suit=Suit.HEARTS, rank=Rank.ACE,
                   is_joker=False, is_big_joker=False, points=0, deck=1)
        action = PlayAction(type=PlayType.SINGLE, cards=[c1])
        assert action.type == PlayType.SINGLE
        assert len(action.cards) == 1


class TestBidAction:
    def test_bid_action_creation(self):
        bid = BidAction(player_index=0, level=Rank.TWO, pass_=False)
        assert bid.player_index == 0
        assert bid.level == Rank.TWO
        assert bid.pass_ is False

    def test_bid_action_pass_alias(self):
        bid = BidAction(player_index=1, level=None, pass_=True)
        assert bid.pass_ is True
        assert bid.level is None
        # JSON serialization should use "pass" not "pass_"
        data = bid.model_dump(by_alias=True)
        assert "pass" in data
        assert "pass_" not in data


class TestStirAction:
    def test_stir_action_creation(self):
        stir = StirAction(player_index=0, new_trump_suit=Suit.HEARTS, level=Rank.TWO)
        assert stir.new_trump_suit == Suit.HEARTS
        assert stir.level == Rank.TWO
