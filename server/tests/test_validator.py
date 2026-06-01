"""Tests for rules.validator module."""
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
        assert len(pairs) == 1

    def test_get_leading_plays_includes_tractors(self):
        hand = [
            _card(Suit.SPADES, Rank.ACE, d) for d in (1, 2)
        ] + [
            _card(Suit.SPADES, Rank.KING, d) for d in (1, 2)
        ]
        plays = get_leading_plays(hand, Suit.HEARTS, Rank.TWO, [])
        tractors = [p for p in plays if p.type == PlayType.TRACTOR]
        assert len(tractors) == 1


class TestGetLegalPlays:
    def test_get_legal_plays_leading(self):
        hand = [_card(Suit.SPADES, Rank.ACE), _card(Suit.HEARTS, Rank.KING)]
        plays = get_legal_plays(
            hand, [], Suit.HEARTS, Rank.TWO,
            is_leading=True, lead_action=None,
        )
        # 2 singles only (no pairs, no tractors, no valid throws)
        assert len(plays) == 2

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
        # Verify non-SINGLE types exist in the full list
        non_singles = [p for p in plays if p.type != PlayType.SINGLE]
        assert len(non_singles) > 0, "plays should contain pairs"
        # Filter to singles only
        singles = filter_by_type(plays, PlayType.SINGLE)
        assert len(singles) < len(plays)
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


class TestRoutingEdgeCases:
    """Edge cases for get_legal_plays routing logic (CR-005)."""

    def test_following_with_all_empty_slots_treats_as_leading(self):
        """When is_leading=False but all trick slots are empty, treat as leading."""
        hand = [_card(Suit.SPADES, Rank.ACE), _card(Suit.HEARTS, Rank.KING)]
        plays = get_legal_plays(
            hand, [{"player_index": 0, "cards": None}],
            Suit.HEARTS, Rank.TWO,
            is_leading=False, lead_action=None,
        )
        # Should return leading plays (2 singles) even though is_leading=False
        assert len(plays) == 2

    def test_following_with_no_lead_action_returns_empty(self):
        """When is_leading=False, trick has cards, but lead_action=None, return []."""
        hand = [_card(Suit.SPADES, Rank.ACE), _card(Suit.HEARTS, Rank.KING)]
        plays = get_legal_plays(
            hand, [{"player_index": 0, "cards": [_card(Suit.SPADES, Rank.ACE)]}],
            Suit.HEARTS, Rank.TWO,
            is_leading=False, lead_action=None,
        )
        assert len(plays) == 0


class TestThrowValidation:
    """Test _is_throw_valid with non-empty remaining_cards (CR-003)."""

    def test_throw_accepted_when_hand_cards_all_equal_highest_remaining(self):
        """Throw accepted when all thrown cards equal the highest remaining card."""
        # Player hand: A-spades (single card, but throw needs 2+ so use two Aces)
        hand = [_card(Suit.SPADES, Rank.ACE, 1), _card(Suit.SPADES, Rank.ACE, 2)]
        # Opponent also holds A-spades -- but there's no A-spades left (both used)
        # remaining has K-spades which is lower
        remaining = [_card(Suit.SPADES, Rank.KING, 1)]
        plays = get_leading_plays(hand, Suit.HEARTS, Rank.TWO, remaining)
        throws = [p for p in plays if p.type == PlayType.THROW]
        assert len(throws) >= 1, "throw should be accepted when all thrown cards >= highest remaining"

    def test_throw_rejected_when_opponent_has_higher_card(self):
        """Throw rejected when an opponent holds a higher card of the same suit."""
        # Player hand: K-spades, Q-spades (want to throw both)
        hand = [_card(Suit.SPADES, Rank.KING, 1), _card(Suit.SPADES, Rank.QUEEN, 1)]
        # Opponent holds A-spades -- K < A so throw is invalid
        remaining = [_card(Suit.SPADES, Rank.ACE, 2)]
        plays = get_leading_plays(hand, Suit.HEARTS, Rank.TWO, remaining)
        throws = [p for p in plays if p.type == PlayType.THROW]
        assert len(throws) == 0, "throw should be rejected when opponent has higher card"

    def test_throw_accepted_when_all_remaining_are_lower(self):
        """Throw accepted when all remaining same-suit cards are lower."""
        # Player hand: A-spades, K-spades
        hand = [_card(Suit.SPADES, Rank.ACE, 1), _card(Suit.SPADES, Rank.KING, 1)]
        # Opponent only has Q-spades
        remaining = [_card(Suit.SPADES, Rank.QUEEN, 2)]
        plays = get_leading_plays(hand, Suit.HEARTS, Rank.TWO, remaining)
        throws = [p for p in plays if p.type == PlayType.THROW]
        assert len(throws) >= 1, "throw should be accepted when all remaining are lower"

    def test_throw_with_empty_remaining_always_accepted(self):
        """Throw with empty remaining_cards is always accepted."""
        hand = [_card(Suit.SPADES, Rank.KING, 1), _card(Suit.SPADES, Rank.QUEEN, 1)]
        plays = get_leading_plays(hand, Suit.HEARTS, Rank.TWO, [])
        throws = [p for p in plays if p.type == PlayType.THROW]
        assert len(throws) >= 1
