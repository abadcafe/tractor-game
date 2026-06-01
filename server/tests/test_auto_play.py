"""Tests for ai.auto_play module."""
import pytest
from server.engine.card import Card, Suit, Rank
from server.engine.types import PlayType, PlayAction
from server.engine.constants import BOTTOM_CARD_COUNT
from server.ai.auto_play import choose_play, choose_bid, choose_stir, choose_discard


def _card(suit: Suit, rank: Rank, deck: int = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit, rank=rank,
        is_joker=(suit == Suit.JOKER),
        is_big_joker=(rank == Rank.BIG_JOKER),
        points=0, deck=deck,
    )


class TestChoosePlay:
    def test_choose_play_from_legal(self):
        """AI must choose one of the legal plays."""
        legal = [
            PlayAction(type=PlayType.SINGLE, cards=[_card(Suit.SPADES, Rank.ACE)]),
            PlayAction(type=PlayType.SINGLE, cards=[_card(Suit.HEARTS, Rank.KING)]),
            PlayAction(type=PlayType.PAIR, cards=[
                _card(Suit.SPADES, Rank.QUEEN, 1),
                _card(Suit.SPADES, Rank.QUEEN, 2),
            ]),
        ]
        chosen = choose_play(legal)
        assert chosen in legal

    def test_choose_play_single_option(self):
        """AI picks the only legal play."""
        only = PlayAction(type=PlayType.SINGLE, cards=[_card(Suit.SPADES, Rank.ACE)])
        chosen = choose_play([only])
        assert chosen == only

    def test_choose_play_empty_raises(self):
        """AI raises ValueError when no legal plays available."""
        with pytest.raises(ValueError, match="No legal plays"):
            choose_play([])

    def test_choose_play_deterministic_with_seed(self):
        """AI with a fixed seed produces deterministic results."""
        legal = [
            PlayAction(type=PlayType.SINGLE, cards=[_card(Suit.SPADES, Rank.ACE)]),
            PlayAction(type=PlayType.SINGLE, cards=[_card(Suit.HEARTS, Rank.KING)]),
        ]
        chosen1 = choose_play(legal, seed=42)
        chosen2 = choose_play(legal, seed=42)
        assert chosen1 == chosen2


class TestChooseBid:
    def test_choose_bid_returns_rank_or_none(self):
        """AI must return a valid bid level or None (pass)."""
        result = choose_bid(valid_levels=[Rank.THREE, Rank.FOUR, Rank.FIVE], current_level=Rank.TWO, seed=42)
        # Either a Rank (bid) or None (pass)
        assert result is None or isinstance(result, Rank)

    def test_choose_bid_valid_levels(self):
        """If AI bids, it must bid a valid level."""
        for _ in range(20):
            result = choose_bid(valid_levels=[Rank.THREE, Rank.FOUR], current_level=Rank.TWO, seed=None)
            if result is not None:
                assert result in (Rank.THREE, Rank.FOUR)

    def test_choose_bid_empty_levels_raises(self):
        """AI raises ValueError when no valid bid levels available."""
        with pytest.raises(ValueError, match="No valid bid levels"):
            choose_bid(valid_levels=[], current_level=Rank.TWO, seed=42)

    def test_choose_bid_pass_with_seed(self):
        """AI can pass (return None) with a deterministic seed."""
        # Seed 1 produces random() = 0.134... < 0.4, so pass
        result = choose_bid(valid_levels=[Rank.THREE, Rank.FOUR], current_level=Rank.TWO, seed=1)
        assert result is None


class TestChooseStir:
    def test_choose_stir_pass_or_stir(self):
        """AI must return a valid stir action (pass or stir)."""
        result = choose_stir(
            current_trump=Suit.HEARTS,
            valid_levels=[Rank.THREE, Rank.FIVE],
            player_index=1,
            stir_history=[],
            seed=42,
        )
        # Result is either None (pass) or a (Suit, Rank) tuple
        if result is not None:
            new_suit, level = result
            assert isinstance(new_suit, Suit)
            assert isinstance(level, Rank)
            assert new_suit != Suit.HEARTS  # Must change trump suit
            assert level in [Rank.THREE, Rank.FIVE]  # Must be a valid level

    def test_choose_stir_empty_levels_raises(self):
        """AI raises ValueError when no valid stir levels available."""
        with pytest.raises(ValueError, match="No valid stir levels"):
            choose_stir(
                current_trump=Suit.HEARTS,
                valid_levels=[],
                player_index=1,
                stir_history=[],
                seed=42,
            )


class TestChooseDiscard:
    def test_choose_discard_correct_count(self):
        """AI must discard exactly BOTTOM_CARD_COUNT cards."""
        hand = [
            _card(Suit.SPADES, Rank.TWO, 1),
            _card(Suit.HEARTS, Rank.THREE, 1),
            _card(Suit.CLUBS, Rank.FOUR, 1),
            _card(Suit.DIAMONDS, Rank.FIVE, 1),
            _card(Suit.SPADES, Rank.SIX, 1),
            _card(Suit.HEARTS, Rank.SEVEN, 1),
            _card(Suit.CLUBS, Rank.EIGHT, 1),
            _card(Suit.DIAMONDS, Rank.NINE, 1),
            _card(Suit.SPADES, Rank.TEN, 1),
        ]
        discard = choose_discard(hand, BOTTOM_CARD_COUNT, seed=42)
        assert len(discard) == BOTTOM_CARD_COUNT
        # All discarded cards must be from the hand
        hand_ids = {c.id for c in hand}
        for c in discard:
            assert c.id in hand_ids

    def test_choose_discard_bottom_card_count(self):
        """Verify BOTTOM_CARD_COUNT == 8."""
        assert BOTTOM_CARD_COUNT == 8

    def test_choose_discard_count_exceeds_hand_raises(self):
        """AI raises ValueError when count exceeds hand size."""
        hand = [_card(Suit.SPADES, Rank.ACE)]
        with pytest.raises(ValueError, match="exceeds hand size"):
            choose_discard(hand, count=5, seed=42)

    def test_choose_discard_negative_count_raises(self):
        """AI raises ValueError when count is negative."""
        hand = [_card(Suit.SPADES, Rank.ACE)]
        with pytest.raises(ValueError, match="non-negative"):
            choose_discard(hand, count=-1, seed=42)

    def test_choose_discard_zero_count(self):
        """AI returns empty list when count is 0."""
        hand = [_card(Suit.SPADES, Rank.ACE), _card(Suit.HEARTS, Rank.KING)]
        discard = choose_discard(hand, count=0, seed=42)
        assert discard == []

    def test_choose_discard_empty_hand_zero_count(self):
        """AI returns empty list for empty hand with count 0."""
        discard = choose_discard([], count=0, seed=42)
        assert discard == []
