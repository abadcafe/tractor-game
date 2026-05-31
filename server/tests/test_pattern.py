"""Tests for rules.pattern module."""
from server.engine.card import Card, Suit, Rank
from server.engine.types import PlayType, PlayAction
from server.rules.pattern import (
    detect_singles, detect_pairs, detect_tractors,
    detect_throw_candidates, describe_play,
)


def _card(suit: Suit, rank: Rank, deck: int = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit, rank=rank,
        is_joker=(suit == Suit.JOKER),
        is_big_joker=(rank == Rank.BIG_JOKER),
        points=0, deck=deck,
    )


class TestDetectSingles:
    def test_detect_singles_count(self):
        hand = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.SPADES, Rank.KING)]
        singles = detect_singles(hand)
        assert len(singles) == 2

    def test_detect_singles_each_card(self):
        cards = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.SPADES, Rank.KING)]
        singles = detect_singles(cards)
        for s in singles:
            assert s.type == PlayType.SINGLE
            assert len(s.cards) == 1


class TestDetectPairs:
    def test_detect_pairs_found(self):
        hand = [
            _card(Suit.HEARTS, Rank.ACE, deck=1),
            _card(Suit.HEARTS, Rank.ACE, deck=2),
            _card(Suit.SPADES, Rank.KING, deck=1),
        ]
        pairs = detect_pairs(hand)
        assert len(pairs) == 1
        assert pairs[0].type == PlayType.PAIR
        assert len(pairs[0].cards) == 2

    def test_detect_pairs_not_found(self):
        hand = [
            _card(Suit.HEARTS, Rank.ACE, deck=1),
            _card(Suit.SPADES, Rank.ACE, deck=1),
        ]
        pairs = detect_pairs(hand)
        assert len(pairs) == 0


class TestDetectTractors:
    def test_detect_tractors_consecutive_pairs(self):
        """Consecutive pairs form a tractor: AA-KK."""
        hand = [
            _card(Suit.SPADES, Rank.ACE, 1),
            _card(Suit.SPADES, Rank.ACE, 2),
            _card(Suit.SPADES, Rank.KING, 1),
            _card(Suit.SPADES, Rank.KING, 2),
        ]
        tractors = detect_tractors(hand, Suit.HEARTS, Rank.TWO)
        assert len(tractors) == 1
        found_4card = [t for t in tractors if len(t.cards) == 4]
        assert len(found_4card) == 1
        assert found_4card[0].type == PlayType.TRACTOR

    def test_detect_tractors_trump_joker_tractor(self):
        """Big joker pair + Small joker pair is a trump tractor."""
        hand = [
            _card(Suit.JOKER, Rank.BIG_JOKER, 1),
            _card(Suit.JOKER, Rank.BIG_JOKER, 2),
            _card(Suit.JOKER, Rank.SMALL_JOKER, 1),
            _card(Suit.JOKER, Rank.SMALL_JOKER, 2),
        ]
        tractors = detect_tractors(hand, Suit.HEARTS, Rank.TWO)
        assert len(tractors) == 1

    def test_detect_tractors_non_consecutive_not_tractor(self):
        """AA and QQ (with KK missing) should not form a tractor."""
        hand = [
            _card(Suit.SPADES, Rank.ACE, 1),
            _card(Suit.SPADES, Rank.ACE, 2),
            _card(Suit.SPADES, Rank.QUEEN, 1),
            _card(Suit.SPADES, Rank.QUEEN, 2),
        ]
        tractors = detect_tractors(hand, Suit.HEARTS, Rank.TWO)
        four_card = [t for t in tractors if len(t.cards) == 4]
        assert len(four_card) == 0


class TestDetectThrowCandidates:
    def test_detect_throw_candidates_non_trump_suit(self):
        """Throw candidates exist for a non-trump suit with 2+ cards."""
        hand = [
            _card(Suit.SPADES, Rank.ACE, 1),
            _card(Suit.SPADES, Rank.KING, 1),
            _card(Suit.HEARTS, Rank.QUEEN, 1),
        ]
        candidates = detect_throw_candidates(hand, Suit.SPADES, Suit.HEARTS, Rank.TWO)
        assert len(candidates) == 1

    def test_detect_throw_candidates_trump_suit_empty(self):
        """No throw candidates for the trump suit."""
        hand = [
            _card(Suit.HEARTS, Rank.ACE, 1),
            _card(Suit.HEARTS, Rank.KING, 1),
        ]
        candidates = detect_throw_candidates(hand, Suit.HEARTS, Suit.HEARTS, Rank.TWO)
        assert len(candidates) == 0


class TestDescribePlay:
    def test_describe_play_single(self):
        c = _card(Suit.HEARTS, Rank.ACE)
        action = PlayAction(type=PlayType.SINGLE, cards=[c])
        desc = describe_play(action)
        assert "♥A" in desc or "A" in desc

    def test_describe_play_pair(self):
        cards = [_card(Suit.SPADES, Rank.KING, d) for d in (1, 2)]
        action = PlayAction(type=PlayType.PAIR, cards=cards)
        desc = describe_play(action)
        assert "♠K" in desc or "K" in desc

    def test_describe_play_tractor(self):
        cards = [
            _card(Suit.SPADES, Rank.ACE, 1), _card(Suit.SPADES, Rank.ACE, 2),
            _card(Suit.SPADES, Rank.KING, 1), _card(Suit.SPADES, Rank.KING, 2),
        ]
        action = PlayAction(type=PlayType.TRACTOR, cards=cards)
        desc = describe_play(action)
        assert len(desc) > 0
