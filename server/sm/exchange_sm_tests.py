"""Tests for sm.exchange_sm module."""
from typing import Literal

from server.rules.cards import Card, POINTS_MAP, Suit, Rank
from .exchange_sm import (
    ExchangeInput, ExchangeResult,
    create_exchange, discard,
)
from server.result import Ok, Rejected


def _card(suit: Suit, rank: Rank, deck: Literal[1, 2] = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit, rank=rank,
        points=POINTS_MAP[rank],
    )


def _make_hand(count: int, offset: int = 0) -> list[Card]:
    """Create a hand of `count` cards with unique IDs, starting from `offset`."""
    cards: list[Card] = []
    suits = [Suit.HEARTS, Suit.SPADES, Suit.DIAMONDS, Suit.CLUBS]
    ranks = [Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN,
             Rank.EIGHT, Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]
    for i in range(count):
        idx = i + offset
        suits_per_deck = 4 * len(ranks)  # 48 suited cards per deck
        deck_idx = (idx // suits_per_deck) % 2  # cycle deck 1, 2
        card_in_deck = idx % suits_per_deck
        suit = suits[card_in_deck % 4]
        rank = ranks[card_in_deck // 4 % len(ranks)]
        cards.append(_card(suit, rank, 1 if deck_idx == 0 else 2))
    return cards


class TestCreateExchange:
    def test_create_exchange_initial_state(self) -> None:
        """Initial state: PICKED_UP, hand includes original + bottom."""
        hand = _make_hand(25)
        bottom = _make_hand(8, offset=25)
        state = create_exchange(ExchangeInput(
            declarer_player=0,
            bottom_cards=bottom,
            declarer_hand=hand,
        ))
        assert state.phase == "PICKED_UP"
        assert len(state.hand_after_pickup) == 33  # 25 + 8
        assert state.count == 8

    def test_create_exchange_picks_up_bottom(self) -> None:
        """All bottom cards are in hand_after_pickup."""
        hand = _make_hand(25)
        bottom = _make_hand(8, offset=25)
        state = create_exchange(ExchangeInput(
            declarer_player=0,
            bottom_cards=bottom,
            declarer_hand=hand,
        ))
        # Every bottom card should be in hand_after_pickup
        bottom_ids = {c.id for c in bottom}
        hand_ids = {c.id for c in state.hand_after_pickup}
        assert bottom_ids.issubset(hand_ids)


class TestDiscard:
    def test_discard_correct_count_transitions_to_complete(self) -> None:
        """Discard exactly count cards -> state transitions to COMPLETE phase."""
        hand = _make_hand(25)
        bottom = _make_hand(8, offset=25)
        state = create_exchange(ExchangeInput(
            declarer_player=0, bottom_cards=bottom, declarer_hand=hand,
        ))
        discarded = state.hand_after_pickup[:8]
        result = discard(state, discarded)
        assert isinstance(result, Ok)
        new_state = result.value
        assert new_state.phase == "COMPLETE"
        assert new_state.result is not None

    def test_discard_cards_in_hand(self) -> None:
        """Discarded cards must be in hand_after_pickup."""
        hand = _make_hand(25)
        bottom = _make_hand(8, offset=25)
        state = create_exchange(ExchangeInput(
            declarer_player=0, bottom_cards=bottom, declarer_hand=hand,
        ))
        discarded = state.hand_after_pickup[:8]
        result = discard(state, discarded)
        assert isinstance(result, Ok)
        assert result.value.phase == "COMPLETE"

    def test_discard_wrong_count_rejected(self) -> None:
        """Discarding wrong number of cards returns Rejected."""
        hand = _make_hand(25)
        bottom = _make_hand(8, offset=25)
        state = create_exchange(ExchangeInput(
            declarer_player=0, bottom_cards=bottom, declarer_hand=hand,
        ))
        result = discard(state, state.hand_after_pickup[:7])
        assert isinstance(result, Rejected)
        assert "数量错误" in result.reason

    def test_discard_not_in_hand_rejected(self) -> None:
        """Discarding cards not in hand returns Rejected."""
        hand = _make_hand(25)
        bottom = _make_hand(8, offset=25)
        state = create_exchange(ExchangeInput(
            declarer_player=0, bottom_cards=bottom, declarer_hand=hand,
        ))
        # Create a fake card not in hand
        fake = _card(Suit.JOKER, Rank.BIG_JOKER, 2)
        result = discard(state, [fake] * 8)
        assert isinstance(result, Rejected)
        assert "不在手牌中" in result.reason

    def test_discard_duplicate_cards_rejected(self) -> None:
        """Discarding the same card multiple times returns Rejected."""
        hand = _make_hand(25)
        bottom = _make_hand(8, offset=25)
        state = create_exchange(ExchangeInput(
            declarer_player=0, bottom_cards=bottom, declarer_hand=hand,
        ))
        # Pick a card from hand and repeat it 8 times
        duplicate_card = state.hand_after_pickup[0]
        result = discard(state, [duplicate_card] * 8)
        assert isinstance(result, Rejected)
        assert "重复" in result.reason


class TestDiscardResult:
    def test_discard_result_new_hand(self) -> None:
        """New hand = hand_after_pickup minus discarded."""
        hand = _make_hand(25)
        bottom = _make_hand(8, offset=25)
        state = create_exchange(ExchangeInput(
            declarer_player=0, bottom_cards=bottom, declarer_hand=hand,
        ))
        discarded = state.hand_after_pickup[:8]
        result = discard(state, discarded)
        assert isinstance(result, Ok)
        new_state = result.value
        assert new_state.result is not None
        assert len(new_state.result.new_hand) == 25  # 33 - 8
        # Discarded cards should NOT be in new hand
        discarded_ids = {c.id for c in discarded}
        new_hand_ids = {c.id for c in new_state.result.new_hand}
        assert discarded_ids.isdisjoint(new_hand_ids)

    def test_discard_result_new_bottom(self) -> None:
        """New bottom = the discarded cards."""
        hand = _make_hand(25)
        bottom = _make_hand(8, offset=25)
        state = create_exchange(ExchangeInput(
            declarer_player=0, bottom_cards=bottom, declarer_hand=hand,
        ))
        discarded = state.hand_after_pickup[:8]
        result = discard(state, discarded)
        assert isinstance(result, Ok)
        new_state = result.value
        assert new_state.result is not None
        assert len(new_state.result.new_bottom_cards) == 8
        discarded_ids = {c.id for c in discarded}
        bottom_ids = {c.id for c in new_state.result.new_bottom_cards}
        assert discarded_ids == bottom_ids

    def test_discard_includes_original_bottom(self) -> None:
        """Declarer can discard cards that were originally bottom."""
        hand = _make_hand(25)
        bottom = _make_hand(8, offset=25)
        state = create_exchange(ExchangeInput(
            declarer_player=0, bottom_cards=bottom, declarer_hand=hand,
        ))
        # Discard the original bottom cards
        discarded = bottom[:]
        result = discard(state, discarded)
        assert isinstance(result, Ok)
        assert result.value.phase == "COMPLETE"

    def test_exchange_result_on_state(self) -> None:
        """After discard, state.result has new_hand and new_bottom_cards."""
        hand = _make_hand(25)
        bottom = _make_hand(8, offset=25)
        state = create_exchange(ExchangeInput(
            declarer_player=0, bottom_cards=bottom, declarer_hand=hand,
        ))
        discarded = state.hand_after_pickup[:8]
        result = discard(state, discarded)
        assert isinstance(result, Ok)
        new_state = result.value
        assert new_state.result is not None
        assert isinstance(new_state.result, ExchangeResult)
        assert len(new_state.result.new_hand) + len(new_state.result.new_bottom_cards) == 33
