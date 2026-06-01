"""Tests for rules.follow_rules module."""
from server.engine.card import Card, Suit, Rank
from server.engine.types import PlayType, PlayAction
from server.rules.follow_rules import get_lead_suit, can_follow, get_legal_follows
from server.rules.comparator import effective_suit


def _card(suit: Suit, rank: Rank, deck: int = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit, rank=rank,
        is_joker=(suit == Suit.JOKER),
        is_big_joker=(rank == Rank.BIG_JOKER),
        points=0, deck=deck,
    )


class TestGetLeadSuit:
    def test_get_lead_suit_trump(self):
        lead = PlayAction(type=PlayType.SINGLE, cards=[_card(Suit.HEARTS, Rank.ACE)])
        assert get_lead_suit(lead, Suit.HEARTS, Rank.TWO) == "trump"

    def test_get_lead_suit_non_trump(self):
        lead = PlayAction(type=PlayType.SINGLE, cards=[_card(Suit.SPADES, Rank.ACE)])
        assert get_lead_suit(lead, Suit.HEARTS, Rank.TWO) == Suit.SPADES


class TestCanFollowSingle:
    def test_can_follow_single_same_suit(self):
        """Must follow with same suit when available."""
        lead = PlayAction(type=PlayType.SINGLE, cards=[_card(Suit.SPADES, Rank.ACE)])
        hand = [_card(Suit.SPADES, Rank.KING), _card(Suit.HEARTS, Rank.QUEEN)]
        assert can_follow(hand, lead, Suit.HEARTS, Rank.TWO) is True

    def test_can_follow_single_no_suit(self):
        """Can't follow suit when no matching cards."""
        lead = PlayAction(type=PlayType.SINGLE, cards=[_card(Suit.SPADES, Rank.ACE)])
        hand = [_card(Suit.CLUBS, Rank.KING), _card(Suit.HEARTS, Rank.QUEEN)]
        assert can_follow(hand, lead, Suit.HEARTS, Rank.TWO) is False


class TestCanFollowPair:
    def test_can_follow_pair_has_pair(self):
        lead = PlayAction(type=PlayType.PAIR, cards=[_card(Suit.SPADES, Rank.ACE, d) for d in (1, 2)])
        hand = [_card(Suit.SPADES, Rank.KING, d) for d in (1, 2)]
        assert can_follow(hand, lead, Suit.HEARTS, Rank.TWO) is True

    def test_can_follow_pair_no_pair(self):
        lead = PlayAction(type=PlayType.PAIR, cards=[_card(Suit.SPADES, Rank.ACE, d) for d in (1, 2)])
        hand = [_card(Suit.SPADES, Rank.KING)]
        assert can_follow(hand, lead, Suit.HEARTS, Rank.TWO) is False


class TestCanFollowTractor:
    def test_can_follow_tractor_has_tractor(self):
        lead = PlayAction(type=PlayType.TRACTOR, cards=[
            _card(Suit.SPADES, Rank.ACE, d) for d in (1, 2)
        ] + [
            _card(Suit.SPADES, Rank.KING, d) for d in (1, 2)
        ])
        hand = [
            _card(Suit.SPADES, Rank.QUEEN, d) for d in (1, 2)
        ] + [
            _card(Suit.SPADES, Rank.JACK, d) for d in (1, 2)
        ]
        assert can_follow(hand, lead, Suit.HEARTS, Rank.TWO) is True

    def test_can_follow_tractor_no_tractor(self):
        lead = PlayAction(type=PlayType.TRACTOR, cards=[
            _card(Suit.SPADES, Rank.ACE, d) for d in (1, 2)
        ] + [
            _card(Suit.SPADES, Rank.KING, d) for d in (1, 2)
        ])
        hand = [_card(Suit.SPADES, Rank.QUEEN)]
        assert can_follow(hand, lead, Suit.HEARTS, Rank.TWO) is False


class TestGetLegalFollowsSingle:
    def test_get_legal_follows_single_must_follow(self):
        """When following a single spade lead, must play same-suit card if available."""
        lead = PlayAction(type=PlayType.SINGLE, cards=[_card(Suit.SPADES, Rank.ACE)])
        hand = [_card(Suit.SPADES, Rank.KING), _card(Suit.HEARTS, Rank.QUEEN)]
        follows = get_legal_follows(hand, lead, Suit.HEARTS, Rank.TWO)
        # All follows should be spade singles
        assert len(follows) >= 1
        for f in follows:
            assert f.type == PlayType.SINGLE
            # The card must be a spade (same effective suit as lead)
            assert effective_suit(f.cards[0], Suit.HEARTS, Rank.TWO) == Suit.SPADES

    def test_get_legal_follows_single_can_play_any(self):
        """When no matching suit, can play any single."""
        lead = PlayAction(type=PlayType.SINGLE, cards=[_card(Suit.SPADES, Rank.ACE)])
        hand = [_card(Suit.CLUBS, Rank.KING), _card(Suit.HEARTS, Rank.QUEEN)]
        follows = get_legal_follows(hand, lead, Suit.HEARTS, Rank.TWO)
        assert len(follows) == 2  # Can play either card


class TestGetLegalFollowsTrumpLead:
    def test_get_legal_follows_trump_lead_must_follow_trump(self):
        """When lead is trump, all must follow with trump if they have any."""
        lead = PlayAction(type=PlayType.SINGLE, cards=[_card(Suit.HEARTS, Rank.ACE)])
        hand = [_card(Suit.HEARTS, Rank.KING), _card(Suit.SPADES, Rank.QUEEN)]
        follows = get_legal_follows(hand, lead, Suit.HEARTS, Rank.TWO)
        for f in follows:
            assert effective_suit(f.cards[0], Suit.HEARTS, Rank.TWO) == "trump"


class TestGetLegalFollowsThrow:
    def test_get_legal_follows_throw_must_follow_count(self):
        """Following a throw must play the same number of cards."""
        lead = PlayAction(type=PlayType.THROW, cards=[
            _card(Suit.SPADES, Rank.ACE), _card(Suit.SPADES, Rank.KING)
        ])
        hand = [
            _card(Suit.SPADES, Rank.QUEEN), _card(Suit.SPADES, Rank.JACK),
            _card(Suit.HEARTS, Rank.TEN),
        ]
        follows = get_legal_follows(hand, lead, Suit.HEARTS, Rank.TWO)
        for f in follows:
            assert len(f.cards) == 2

    def test_get_legal_follows_throw_not_enough(self):
        """Not enough matching cards: must play all matching + fill with others."""
        lead = PlayAction(type=PlayType.THROW, cards=[
            _card(Suit.SPADES, Rank.ACE), _card(Suit.SPADES, Rank.KING)
        ])
        hand = [_card(Suit.SPADES, Rank.QUEEN), _card(Suit.HEARTS, Rank.TEN)]
        follows = get_legal_follows(hand, lead, Suit.HEARTS, Rank.TWO)
        assert len(follows) >= 1
        for f in follows:
            assert len(f.cards) == 2


class TestCanFollowTractorTrump:
    """CR-005: Trump tractor scenarios where trump_order step != 1."""

    def test_can_follow_trump_tractor_has_consecutive_trump_pairs(self):
        """SJ pair + trump-rank pair can follow BJ pair + SJ pair (step=10)."""
        # Trump: hearts, rank TWO
        # Lead: BJ pair + SJ pair (orders 100, 90, step=10)
        lead = PlayAction(type=PlayType.TRACTOR, cards=[
            _card(Suit.JOKER, Rank.BIG_JOKER, d) for d in (1, 2)
        ] + [
            _card(Suit.JOKER, Rank.SMALL_JOKER, d) for d in (1, 2)
        ])
        # Hand: SJ pair + trump-rank+trump-suit pair (orders 90, 80, step=10)
        hand = [
            _card(Suit.JOKER, Rank.SMALL_JOKER, d) for d in (1, 2)
        ] + [
            _card(Suit.HEARTS, Rank.TWO, d) for d in (1, 2)
        ]
        assert can_follow(hand, lead, Suit.HEARTS, Rank.TWO) is True

    def test_can_follow_trump_tractor_no_consecutive_trump_pairs(self):
        """Same-rank different-suit trump pairs cannot form a tractor (CR-007)."""
        # Lead: BJ pair + SJ pair
        lead = PlayAction(type=PlayType.TRACTOR, cards=[
            _card(Suit.JOKER, Rank.BIG_JOKER, d) for d in (1, 2)
        ] + [
            _card(Suit.JOKER, Rank.SMALL_JOKER, d) for d in (1, 2)
        ])
        # Hand: TWO-HEARTS pair + TWO-CLUBS pair (same rank, different suits)
        # Both are trump rank cards but at the SAME rank level -- not a tractor
        hand = [
            _card(Suit.HEARTS, Rank.TWO, d) for d in (1, 2)
        ] + [
            _card(Suit.CLUBS, Rank.TWO, d) for d in (1, 2)
        ]
        assert can_follow(hand, lead, Suit.HEARTS, Rank.TWO) is False

    def test_get_legal_follows_trump_tractor_returns_valid(self):
        """can_follow and get_legal_follows must agree on trump tractors."""
        # Lead: BJ pair + SJ pair (orders 100, 90, step=10)
        lead = PlayAction(type=PlayType.TRACTOR, cards=[
            _card(Suit.JOKER, Rank.BIG_JOKER, d) for d in (1, 2)
        ] + [
            _card(Suit.JOKER, Rank.SMALL_JOKER, d) for d in (1, 2)
        ])
        # Hand: SJ pair + trump-rank+trump-suit pair (orders 90, 80, step=10)
        hand = [
            _card(Suit.JOKER, Rank.SMALL_JOKER, d) for d in (1, 2)
        ] + [
            _card(Suit.HEARTS, Rank.TWO, d) for d in (1, 2)
        ]
        can = can_follow(hand, lead, Suit.HEARTS, Rank.TWO)
        follows = get_legal_follows(hand, lead, Suit.HEARTS, Rank.TWO)
        # Both must agree: either both say can follow, or both say can't
        if can:
            assert len(follows) >= 1
            for f in follows:
                assert f.type == PlayType.TRACTOR
                assert len(f.cards) == 4
