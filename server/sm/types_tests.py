"""Tests for sm.types module."""
import pytest
from server.sm.card_model import Card, Suit, Rank
from server.sm.types import (
    PlayType, PlayAction, BidEvent, StirAction, Player,
    CompletedTrick, CompletedTrickSlot,
)


def _card(suit: Suit, rank: Rank, deck: int = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit, rank=rank,
        is_joker=(suit == Suit.JOKER),
        is_big_joker=(rank == Rank.BIG_JOKER),
        points=0, deck=deck,
    )


class TestPlayType:
    def test_play_type_values(self) -> None:
        assert PlayType.SINGLE.value == "single"
        assert PlayType.PAIR.value == "pair"
        assert PlayType.TRACTOR.value == "tractor"
        assert PlayType.THROW.value == "throw"


class TestPlayAction:
    def test_play_action_single(self) -> None:
        """PlayAction for a single card."""
        card = _card(Suit.HEARTS, Rank.ACE)
        action = PlayAction(type=PlayType.SINGLE, cards=[card])
        assert action.type == PlayType.SINGLE
        assert len(action.cards) == 1
        assert action.cards[0] == card

    def test_play_action_pair(self) -> None:
        """PlayAction for a pair."""
        c1 = _card(Suit.HEARTS, Rank.ACE, 1)
        c2 = _card(Suit.HEARTS, Rank.ACE, 2)
        action = PlayAction(type=PlayType.PAIR, cards=[c1, c2])
        assert action.type == PlayType.PAIR
        assert len(action.cards) == 2

    def test_play_action_tractor(self) -> None:
        """PlayAction for a tractor."""
        cards = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
        ]
        action = PlayAction(type=PlayType.TRACTOR, cards=cards)
        assert action.type == PlayType.TRACTOR
        assert len(action.cards) == 4


class TestBidEvent:
    def test_bid_event_creation_trump_rank(self) -> None:
        """BidEvent for revealing trump rank cards."""
        cards = [_card(Suit.HEARTS, Rank.TWO)]
        event = BidEvent(
            player=0,
            cards=cards,
            kind="trump_rank",
            suit=Suit.HEARTS,
            joker_type=None,
            count=1,
        )
        assert event.player == 0
        assert event.kind == "trump_rank"
        assert event.suit == Suit.HEARTS
        assert event.count == 1

    def test_bid_event_creation_joker(self) -> None:
        """BidEvent for revealing joker pair."""
        cards = [
            _card(Suit.JOKER, Rank.BIG_JOKER, 1),
            _card(Suit.JOKER, Rank.BIG_JOKER, 2),
        ]
        event = BidEvent(
            player=2,
            cards=cards,
            kind="joker",
            suit=None,
            joker_type="big",
            count=2,
        )
        assert event.kind == "joker"
        assert event.suit is None
        assert event.joker_type == "big"
        assert event.count == 2

    def test_bid_event_suit_none_for_joker(self) -> None:
        """Joker bid event has suit=None."""
        cards = [_card(Suit.JOKER, Rank.SMALL_JOKER, 1), _card(Suit.JOKER, Rank.SMALL_JOKER, 2)]
        event = BidEvent(
            player=1, cards=cards, kind="joker",
            suit=None, joker_type="small", count=2,
        )
        assert event.suit is None


class TestStirAction:
    def test_stir_action_creation(self) -> None:
        """StirAction records a player's stir or pass."""
        action = StirAction(player=1, kind="stir", new_suit=Suit.SPADES)
        assert action.player == 1
        assert action.kind == "stir"
        assert action.new_suit == Suit.SPADES

    def test_stir_action_pass(self) -> None:
        """StirAction for a pass."""
        action = StirAction(player=0, kind="pass", new_suit=None)
        assert action.kind == "pass"
        assert action.new_suit is None


class TestPlayer:
    def test_player_creation(self) -> None:
        """Player data model with index, team, hand."""
        player = Player(index=0, team=0, hand=[], is_declarer=False)
        assert player.index == 0
        assert player.team == 0
        assert player.hand == []
        assert player.is_declarer is False

    def test_player_defaults(self) -> None:
        """is_declarer defaults to False."""
        player = Player(index=1, team=1, hand=[])
        assert player.is_declarer is False


class TestCompletedTrick:
    def test_completed_trick_creation(self) -> None:
        """CompletedTrick holds full trick data."""
        slot0 = CompletedTrickSlot(player=0, cards=[_card(Suit.HEARTS, Rank.ACE)])
        slot1 = CompletedTrickSlot(player=1, cards=[_card(Suit.HEARTS, Rank.KING)])
        trick = CompletedTrick(
            lead_player=0,
            lead_type=PlayType.SINGLE,
            slots=[slot0, slot1],
            winner=0,
            points=10,
        )
        assert trick.lead_player == 0
        assert trick.winner == 0
        assert trick.points == 10
        assert len(trick.slots) == 2

    def test_completed_trick_slot_creation(self) -> None:
        """Individual trick slot with player and cards."""
        slot = CompletedTrickSlot(player=3, cards=[_card(Suit.SPADES, Rank.FIVE)])
        assert slot.player == 3
        assert len(slot.cards) == 1
