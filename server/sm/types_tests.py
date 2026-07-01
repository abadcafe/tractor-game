"""Tests for sm.types module."""

from typing import Literal

import pytest
from pydantic import ValidationError

from server.rules.cards import POINTS_MAP, Card, Rank, Suit
from server.rules.types import SubPlay

from .types import (
    BidEvent,
    BottomExchangeEvent,
    CompletedTrick,
    CompletedTrickSlot,
    Player,
    StirDeclarationEvent,
)


def _card(suit: Suit, rank: Rank, deck: Literal[1, 2] = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit,
        rank=rank,
        points=POINTS_MAP[rank],
    )


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
        cards = [
            _card(Suit.JOKER, Rank.SMALL_JOKER, 1),
            _card(Suit.JOKER, Rank.SMALL_JOKER, 2),
        ]
        event = BidEvent(
            player=1,
            cards=cards,
            kind="joker",
            suit=None,
            joker_type="small",
            count=2,
        )
        assert event.suit is None

    def test_bid_event_frozen(self) -> None:
        """BidEvent is immutable (frozen=True)."""
        cards = [_card(Suit.HEARTS, Rank.TWO)]
        event = BidEvent(
            player=0,
            cards=cards,
            kind="trump_rank",
            suit=Suit.HEARTS,
            joker_type=None,
            count=1,
        )
        with pytest.raises(ValidationError):
            event.player = 1

    def test_bid_event_trump_rank_requires_suit(self) -> None:
        """BidEvent.kind='trump_rank' requires suit to be set."""
        cards = [_card(Suit.HEARTS, Rank.TWO)]
        with pytest.raises(ValidationError):
            BidEvent(
                player=0,
                cards=cards,
                kind="trump_rank",
                suit=None,
                joker_type=None,
                count=1,
            )

    def test_bid_event_joker_rejects_suit(self) -> None:
        """BidEvent.kind='joker' requires suit=None."""
        cards = [
            _card(Suit.JOKER, Rank.BIG_JOKER, 1),
            _card(Suit.JOKER, Rank.BIG_JOKER, 2),
        ]
        with pytest.raises(ValidationError):
            BidEvent(
                player=0,
                cards=cards,
                kind="joker",
                suit=Suit.HEARTS,
                joker_type="big",
                count=2,
            )


class TestStirDeclarationEvent:
    def test_stir_declaration_event_creation(self) -> None:
        """StirDeclarationEvent records a player's stir declaration."""
        cards = [_card(Suit.SPADES, Rank.TWO)]
        event = StirDeclarationEvent(
            player=1,
            kind="stir",
            cards=cards,
            new_suit=Suit.SPADES,
            priority=101,
        )
        assert event.player == 1
        assert event.kind == "stir"
        assert event.cards == cards
        assert event.new_suit == Suit.SPADES
        assert event.priority == 101

    def test_stir_declaration_event_pass(self) -> None:
        """StirDeclarationEvent records a player's pass."""
        event = StirDeclarationEvent(
            player=0,
            kind="pass",
            cards=[],
            new_suit=None,
            priority=None,
        )
        assert event.kind == "pass"
        assert event.cards == []
        assert event.new_suit is None
        assert event.priority is None

    def test_stir_declaration_event_frozen(self) -> None:
        """StirDeclarationEvent is immutable (frozen=True)."""
        event = StirDeclarationEvent(
            player=1,
            kind="stir",
            cards=[_card(Suit.SPADES, Rank.TWO)],
            new_suit=Suit.SPADES,
            priority=101,
        )
        with pytest.raises(ValidationError):
            event.player = 2

    def test_stir_declaration_event_stir_allows_no_trump(self) -> None:
        """StirDeclarationEvent.kind='stir' allows joker no-trump."""
        event = StirDeclarationEvent(
            player=0,
            kind="stir",
            cards=[
                _card(Suit.JOKER, Rank.BIG_JOKER, 1),
                _card(Suit.JOKER, Rank.BIG_JOKER, 2),
            ],
            new_suit=None,
            priority=205,
        )
        assert event.kind == "stir"
        assert event.new_suit is None

    def test_stir_declaration_event_pass_rejects_suit(self) -> None:
        """StirDeclarationEvent.kind='pass' requires new_suit=None."""
        with pytest.raises(ValidationError):
            StirDeclarationEvent(
                player=0,
                kind="pass",
                cards=[],
                new_suit=Suit.HEARTS,
                priority=None,
            )

    def test_stir_declaration_event_pass_rejects_cards(self) -> None:
        """StirDeclarationEvent.kind='pass' cannot carry cards."""
        with pytest.raises(ValidationError):
            StirDeclarationEvent(
                player=0,
                kind="pass",
                cards=[_card(Suit.HEARTS, Rank.TWO)],
                new_suit=None,
                priority=None,
            )


class TestBottomExchangeEvent:
    def test_bottom_exchange_event_initial_creation(self) -> None:
        """BottomExchangeEvent records initial bottom pickup/discard."""
        picked = [_card(Suit.HEARTS, Rank.ACE)]
        discarded = [_card(Suit.CLUBS, Rank.THREE)]
        event = BottomExchangeEvent(
            player=0,
            trigger="initial",
            stir_event_index=None,
            picked_up_bottom_cards=picked,
            discarded_bottom_cards=discarded,
            resulting_bottom_cards=discarded,
        )
        assert event.player == 0
        assert event.trigger == "initial"
        assert event.stir_event_index is None
        assert event.picked_up_bottom_cards == picked
        assert event.discarded_bottom_cards == discarded

    def test_bottom_exchange_event_stir_requires_event_index(
        self,
    ) -> None:
        """Stir-triggered bottom exchanges point to the stir event."""
        event = BottomExchangeEvent(
            player=2,
            trigger="stir",
            stir_event_index=3,
            picked_up_bottom_cards=[_card(Suit.HEARTS, Rank.ACE)],
            discarded_bottom_cards=[_card(Suit.CLUBS, Rank.THREE)],
            resulting_bottom_cards=[_card(Suit.CLUBS, Rank.THREE)],
        )
        assert event.trigger == "stir"
        assert event.stir_event_index == 3

    def test_bottom_exchange_event_rejects_initial_event_index(
        self,
    ) -> None:
        """Initial exchange is not linked to a stir event."""
        with pytest.raises(ValidationError):
            BottomExchangeEvent(
                player=0,
                trigger="initial",
                stir_event_index=0,
                picked_up_bottom_cards=[],
                discarded_bottom_cards=[],
                resulting_bottom_cards=[],
            )

    def test_bottom_exchange_event_rejects_missing_stir_event_index(
        self,
    ) -> None:
        """Stir exchange must reference the triggering stir event."""
        with pytest.raises(ValidationError):
            BottomExchangeEvent(
                player=0,
                trigger="stir",
                stir_event_index=None,
                picked_up_bottom_cards=[],
                discarded_bottom_cards=[],
                resulting_bottom_cards=[],
            )


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

    def test_player_hand_mutable(self) -> None:
        """Player.hand is mutable (Player is NOT frozen)."""
        player = Player(index=0, team=0, hand=[])
        card = _card(Suit.HEARTS, Rank.ACE)
        player.hand.append(card)
        assert len(player.hand) == 1


class TestCompletedTrick:
    def test_completed_trick_creation(self) -> None:
        """CompletedTrick holds full trick data."""
        slot0 = CompletedTrickSlot(
            player=0, cards=[_card(Suit.HEARTS, Rank.ACE)]
        )
        slot1 = CompletedTrickSlot(
            player=1, cards=[_card(Suit.HEARTS, Rank.KING)]
        )
        trick = CompletedTrick(
            lead_player=0,
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
        slot = CompletedTrickSlot(
            player=3, cards=[_card(Suit.SPADES, Rank.FIVE)]
        )
        assert slot.player == 3
        assert len(slot.cards) == 1

    def test_completed_trick_frozen(self) -> None:
        """CompletedTrick is immutable (frozen=True)."""
        slot = CompletedTrickSlot(
            player=0, cards=[_card(Suit.HEARTS, Rank.ACE)]
        )
        trick = CompletedTrick(
            lead_player=0,
            slots=[slot],
            winner=0,
            points=10,
        )
        with pytest.raises(ValidationError):
            trick.winner = 1

    def test_completed_trick_slot_frozen(self) -> None:
        """CompletedTrickSlot is immutable (frozen=True)."""
        slot = CompletedTrickSlot(
            player=0, cards=[_card(Suit.HEARTS, Rank.ACE)]
        )
        with pytest.raises(ValidationError):
            slot.player = 1


class TestSubPlay:
    def test_subplay_single(self) -> None:
        """Single card: pair_count=0."""
        c = _card(Suit.HEARTS, Rank.ACE)
        sp = SubPlay(pair_count=0, cards=[c], suit=Suit.HEARTS)
        assert sp.pair_count == 0
        assert len(sp.cards) == 1
        assert sp.suit == Suit.HEARTS

    def test_subplay_pair(self) -> None:
        """Pair: pair_count=1."""
        c1 = _card(Suit.HEARTS, Rank.ACE, 1)
        c2 = _card(Suit.HEARTS, Rank.ACE, 2)
        sp = SubPlay(pair_count=1, cards=[c1, c2], suit=Suit.HEARTS)
        assert sp.pair_count == 1
        assert len(sp.cards) == 2

    def test_subplay_tractor(self) -> None:
        """Tractor (2 pairs): pair_count=2."""
        cards = [
            _card(Suit.HEARTS, Rank.THREE, 1),
            _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1),
            _card(Suit.HEARTS, Rank.FOUR, 2),
        ]
        sp = SubPlay(pair_count=2, cards=cards, suit=Suit.HEARTS)
        assert sp.pair_count == 2
        assert len(sp.cards) == 4

    def test_subplay_tractor_3_pairs(self) -> None:
        """Tractor (3 pairs): pair_count=3, 6 cards."""
        cards = [
            _card(Suit.HEARTS, Rank.THREE, 1),
            _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1),
            _card(Suit.HEARTS, Rank.FOUR, 2),
            _card(Suit.HEARTS, Rank.FIVE, 1),
            _card(Suit.HEARTS, Rank.FIVE, 2),
        ]
        sp = SubPlay(pair_count=3, cards=cards, suit=Suit.HEARTS)
        assert sp.pair_count == 3
        assert len(sp.cards) == 6

    def test_subplay_trump_suit(self) -> None:
        """SubPlay suit can be 'trump' string."""
        c = _card(Suit.JOKER, Rank.BIG_JOKER)
        sp = SubPlay(pair_count=0, cards=[c], suit="trump")
        assert sp.suit == "trump"

    def test_subplay_level_single(self) -> None:
        """sub_level for single: pair_count=0 -> level=1."""
        sp = SubPlay(pair_count=0, cards=[], suit=Suit.HEARTS)
        assert sp.sub_level == 1

    def test_subplay_level_pair(self) -> None:
        """sub_level for pair: pair_count=1 -> level=2."""
        sp = SubPlay(pair_count=1, cards=[], suit=Suit.HEARTS)
        assert sp.sub_level == 2

    def test_subplay_level_tractor(self) -> None:
        """sub_level for tractor: pair_count=2 -> level=3."""
        sp = SubPlay(pair_count=2, cards=[], suit=Suit.HEARTS)
        assert sp.sub_level == 3

    def test_subplay_level_tractor_3_pairs(self) -> None:
        """sub_level for 3-pair tractor: pair_count=3 -> level=4."""
        sp = SubPlay(pair_count=3, cards=[], suit=Suit.HEARTS)
        assert sp.sub_level == 4

    def test_subplay_frozen(self) -> None:
        """SubPlay is immutable (Pydantic frozen model)."""
        c = _card(Suit.HEARTS, Rank.ACE)
        sp = SubPlay(pair_count=0, cards=[c], suit=Suit.HEARTS)
        with pytest.raises(ValidationError):
            setattr(sp, "pair_count", 1)

    def test_subplay_negative_pair_count_rejected(self) -> None:
        """SubPlay rejects negative pair_count."""
        with pytest.raises(ValidationError):
            SubPlay(pair_count=-1, cards=[], suit=Suit.HEARTS)

    def test_subplay_mismatched_cards_count_rejected(self) -> None:
        """SubPlay rejects cards count that doesn't match pair_count."""
        c1 = _card(Suit.HEARTS, Rank.ACE, 1)
        c2 = _card(Suit.HEARTS, Rank.ACE, 2)
        c3 = _card(
            Suit.HEARTS, Rank.ACE, 1
        )  # 3 cards for pair_count=1 (needs 2)
        with pytest.raises(ValidationError):
            SubPlay(pair_count=1, cards=[c1, c2, c3], suit=Suit.HEARTS)
