"""Tests for rules.validator module."""
import pytest
from server.engine.card import Card, Suit, Rank
from server.engine.types import PlayType, PlayAction
from server.rules.validator import (
    get_legal_plays, get_leading_plays, is_legal_play,
    filter_by_type, describe_legal_plays,
)


def _card(suit: Suit, rank: Rank, deck: int = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit, rank=rank,
        is_joker=(suit == Suit.JOKER),
        is_big_joker=(rank == Rank.BIG_JOKER),
        points=0, deck=deck,
    )


class TestGetLeadingPlays:
    def test_get_leading_plays_includes_singles(self):
        hand = [_card(Suit.SPADES, Rank.ACE), _card(Suit.HEARTS, Rank.KING)]
        plays = get_leading_plays(hand, Suit.HEARTS, Rank.TWO, [])
        singles = [p for p in plays if p.type == PlayType.SINGLE]
        assert len(singles) == 2

    def test_get_leading_plays_includes_pairs(self):
        hand = [
            _card(Suit.SPADES, Rank.ACE, d) for d in (1, 2)
        ] + [_card(Suit.HEARTS, Rank.KING)]
        plays = get_leading_plays(hand, Suit.HEARTS, Rank.TWO, [])
        pairs = [p for p in plays if p.type == PlayType.PAIR]
        assert len(pairs) >= 1

    def test_get_leading_plays_includes_tractors(self):
        hand = [
            _card(Suit.SPADES, Rank.ACE, d) for d in (1, 2)
        ] + [
            _card(Suit.SPADES, Rank.KING, d) for d in (1, 2)
        ]
        plays = get_leading_plays(hand, Suit.HEARTS, Rank.TWO, [])
        tractors = [p for p in plays if p.type == PlayType.TRACTOR]
        assert len(tractors) >= 1


class TestGetLegalPlays:
    def test_get_legal_plays_leading(self):
        hand = [_card(Suit.SPADES, Rank.ACE), _card(Suit.HEARTS, Rank.KING)]
        plays = get_legal_plays(
            hand, [], Suit.HEARTS, Rank.TWO,
            is_leading=True, lead_action=None,
        )
        assert len(plays) >= 2

    def test_get_legal_plays_following(self):
        lead = PlayAction(type=PlayType.SINGLE, cards=[_card(Suit.SPADES, Rank.ACE)])
        hand = [_card(Suit.SPADES, Rank.KING), _card(Suit.HEARTS, Rank.QUEEN)]
        plays = get_legal_plays(
            hand, [{"player_index": 0, "cards": [_card(Suit.SPADES, Rank.ACE)]}],
            Suit.HEARTS, Rank.TWO,
            is_leading=False, lead_action=lead,
        )
        assert len(plays) >= 1

    def test_get_legal_plays_empty_hand(self):
        plays = get_legal_plays(
            [], [], Suit.HEARTS, Rank.TWO,
            is_leading=True, lead_action=None,
        )
        assert len(plays) == 0


class TestIsLegalPlay:
    def test_is_legal_play_true(self):
        hand = [_card(Suit.SPADES, Rank.ACE), _card(Suit.HEARTS, Rank.KING)]
        plays = get_leading_plays(hand, Suit.HEARTS, Rank.TWO, [])
        assert len(plays) > 0
        assert is_legal_play(plays[0], plays) is True

    def test_is_legal_play_false(self):
        hand = [_card(Suit.SPADES, Rank.ACE)]
        legal = get_leading_plays(hand, Suit.HEARTS, Rank.TWO, [])
        fake = PlayAction(type=PlayType.SINGLE, cards=[_card(Suit.HEARTS, Rank.KING)])
        assert is_legal_play(fake, legal) is False


class TestFilterByType:
    def test_filter_by_type(self):
        hand = [
            _card(Suit.SPADES, Rank.ACE, d) for d in (1, 2)
        ] + [_card(Suit.HEARTS, Rank.KING)]
        plays = get_leading_plays(hand, Suit.HEARTS, Rank.TWO, [])
        singles = filter_by_type(plays, PlayType.SINGLE)
        for s in singles:
            assert s.type == PlayType.SINGLE


class TestDescribeLegalPlays:
    def test_describe_legal_plays(self):
        hand = [_card(Suit.SPADES, Rank.ACE), _card(Suit.HEARTS, Rank.KING)]
        plays = get_leading_plays(hand, Suit.HEARTS, Rank.TWO, [])
        descriptions = describe_legal_plays(plays)
        assert len(descriptions) == len(plays)
        for desc in descriptions:
            assert isinstance(desc, str)
            assert len(desc) > 0
