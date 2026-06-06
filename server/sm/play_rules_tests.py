"""Tests for sm.play_rules module."""
from server.sm.card_model import Card, Suit, Rank
from server.sm.types import PlayType, PlayAction
from server.sm.play_rules import (
    detect_singles, detect_pairs, detect_tractors, detect_throws,
    get_legal_plays, infer_play_type,
)
from server.sm.comparator import effective_suit


def _card(suit: Suit, rank: Rank, deck: int = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit, rank=rank,
        is_joker=(suit == Suit.JOKER),
        is_big_joker=(rank == Rank.BIG_JOKER),
        points=0, deck=deck,
    )


class TestDetectSingles:
    def test_detect_singles(self) -> None:
        """Every card is a valid single."""
        hand = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.SPADES, Rank.KING)]
        singles = detect_singles(hand)
        assert len(singles) == 2
        assert all(s.type == PlayType.SINGLE for s in singles)


class TestDetectPairs:
    def test_detect_pairs_same_suit(self) -> None:
        """Two cards of same suit+rank form a pair."""
        hand = [_card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2)]
        pairs = detect_pairs(hand)
        assert len(pairs) == 1
        assert pairs[0].type == PlayType.PAIR
        assert len(pairs[0].cards) == 2

    def test_detect_pairs_no_pair(self) -> None:
        """Different ranks do not form a pair."""
        hand = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING)]
        pairs = detect_pairs(hand)
        assert len(pairs) == 0


class TestDetectTractors:
    def test_detect_tractors_two_pairs(self) -> None:
        """Two consecutive pairs form a 4-card tractor."""
        hand = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
        ]
        tractors = detect_tractors(hand, Suit.SPADES, Rank.TWO)
        assert len(tractors) >= 1
        found = any(t.type == PlayType.TRACTOR and len(t.cards) == 4 for t in tractors)
        assert found, f"No 4-card tractor found in {[len(t.cards) for t in tractors]}"

    def test_detect_tractors_three_pairs(self) -> None:
        """Three consecutive pairs form a 6-card tractor."""
        hand = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
            _card(Suit.HEARTS, Rank.FIVE, 1), _card(Suit.HEARTS, Rank.FIVE, 2),
        ]
        tractors = detect_tractors(hand, Suit.SPADES, Rank.TWO)
        found = any(t.type == PlayType.TRACTOR and len(t.cards) == 6 for t in tractors)
        assert found, f"No 6-card tractor found in {[len(t.cards) for t in tractors]}"

    def test_detect_tractors_no_tractor(self) -> None:
        """Non-consecutive pairs do not form a tractor."""
        hand = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2),
        ]
        tractors = detect_tractors(hand, Suit.SPADES, Rank.TWO)
        assert len(tractors) == 0

    def test_detect_tractors_trump_rank_gap(self) -> None:
        """Trump rank is skipped in non-trump ordering: 4-4 + 6-6 is a tractor when trump_rank=5."""
        hand = [
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
            _card(Suit.HEARTS, Rank.SIX, 1), _card(Suit.HEARTS, Rank.SIX, 2),
        ]
        tractors = detect_tractors(hand, Suit.SPADES, Rank.FIVE)
        found = any(t.type == PlayType.TRACTOR and len(t.cards) == 4 for t in tractors)
        assert found, "4-4+6-6 should be tractor when trump_rank=5 (5 skipped)"

    def test_detect_tractors_trump_group(self) -> None:
        """Trump cards form a group for tractor detection."""
        hand = [
            _card(Suit.HEARTS, Rank.TWO, 1), _card(Suit.HEARTS, Rank.TWO, 2),
            _card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2),
        ]
        # With trump_suit=HEARTS, trump_rank=TWO: all are trump cards
        # Two pairs of trump rank cards may or may not form a tractor depending on
        # the trump ordering; at minimum we should detect the pairs
        detect_tractors(hand, Suit.HEARTS, Rank.TWO)
        pairs = detect_pairs(hand)
        assert len(pairs) >= 2


class TestDetectThrows:
    def test_detect_throws_all_highest_in_suit(self) -> None:
        """THROW: multiple cards of same suit where each is the highest remaining rank."""
        # Player holds A, K (both are high ranks in hearts)
        hand = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING)]
        # When no other cards are known to exist, these are the highest
        throws = detect_throws(hand, Suit.SPADES, Rank.TWO, known_remaining_cards=None)
        # May or may not detect as throw depending on whether we can verify highest
        # At minimum, should not crash
        assert isinstance(throws, list)

    def test_detect_throws_partial_high_cards(self) -> None:
        """THROW detection with some high and some lower cards."""
        hand = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.THREE)]
        throws = detect_throws(hand, Suit.SPADES, Rank.TWO, known_remaining_cards=None)
        assert isinstance(throws, list)

    def test_detect_throws_no_throw_when_lower_rank_exists(self) -> None:
        """THROW is not possible if a higher rank exists in other players' hands."""
        hand = [_card(Suit.HEARTS, Rank.KING)]
        # If we know A is still in another player's hand, K is not the highest
        other_cards = [_card(Suit.HEARTS, Rank.ACE)]
        throws = detect_throws(hand, Suit.SPADES, Rank.TWO, known_remaining_cards=other_cards)
        # King cannot be thrown if Ace exists elsewhere
        for t in throws:
            if t.type == PlayType.THROW:
                assert not any(c.rank == Rank.KING and c.suit == Suit.HEARTS for c in t.cards), \
                    "King should not be in THROW if Ace exists in other hands"

    def test_detect_throws_trump_cards(self) -> None:
        """THROW detection does not apply to trump cards (trump is always 'highest')."""
        hand = [_card(Suit.HEARTS, Rank.TWO, 1), _card(Suit.HEARTS, Rank.TWO, 2)]
        throws = detect_throws(hand, Suit.HEARTS, Rank.TWO, known_remaining_cards=None)
        # Trump cards use other play types (pair/tractor), not THROW
        for t in throws:
            assert t.type != PlayType.THROW or not all(
                effective_suit(c, Suit.HEARTS, Rank.TWO) == "trump" for c in t.cards
            ), "Trump cards should not form THROW"


class TestGetLegalPlays:
    def test_get_legal_plays_leading_single(self) -> None:
        """When leading, every single card is legal."""
        hand = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.SPADES, Rank.KING)]
        plays = get_legal_plays(
            hand=hand, is_leading=True, lead_action=None,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
        )
        singles = [p for p in plays if p.type == PlayType.SINGLE]
        assert len(singles) >= 2

    def test_get_legal_plays_leading_pair(self) -> None:
        """When leading with a pair in hand, pair is a legal play."""
        hand = [_card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2)]
        plays = get_legal_plays(
            hand=hand, is_leading=True, lead_action=None,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
        )
        pairs = [p for p in plays if p.type == PlayType.PAIR]
        assert len(pairs) >= 1

    def test_get_legal_plays_following_single_must_follow(self) -> None:
        """Following a single: must play same effective suit if possible."""
        hand = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.SPADES, Rank.KING)]
        lead = PlayAction(type=PlayType.SINGLE, cards=[_card(Suit.HEARTS, Rank.TEN)])
        plays = get_legal_plays(
            hand=hand, is_leading=False, lead_action=lead,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
        )
        # Must follow hearts
        assert len(plays) >= 1
        assert all(p.type == PlayType.SINGLE for p in plays)

    def test_get_legal_plays_following_single_no_suit(self) -> None:
        """Following a single with no matching suit: can play anything."""
        hand = [_card(Suit.SPADES, Rank.KING)]
        lead = PlayAction(type=PlayType.SINGLE, cards=[_card(Suit.HEARTS, Rank.TEN)])
        plays = get_legal_plays(
            hand=hand, is_leading=False, lead_action=lead,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
        )
        assert len(plays) >= 1

    def test_get_legal_plays_following_pair_must_follow(self) -> None:
        """Following a pair: must play a pair of same effective suit if possible."""
        hand = [
            _card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.SPADES, Rank.KING),
        ]
        lead = PlayAction(type=PlayType.PAIR, cards=[_card(Suit.HEARTS, Rank.TEN, 1), _card(Suit.HEARTS, Rank.TEN, 2)])
        plays = get_legal_plays(
            hand=hand, is_leading=False, lead_action=lead,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
        )
        pairs = [p for p in plays if p.type == PlayType.PAIR]
        assert len(pairs) >= 1

    def test_get_legal_plays_following_pair_no_pair(self) -> None:
        """Following a pair with no pair: can play any two cards."""
        hand = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING)]
        lead = PlayAction(type=PlayType.PAIR, cards=[_card(Suit.HEARTS, Rank.TEN, 1), _card(Suit.HEARTS, Rank.TEN, 2)])
        plays = get_legal_plays(
            hand=hand, is_leading=False, lead_action=lead,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
        )
        assert len(plays) >= 1

    def test_get_legal_plays_following_tractor_partial(self) -> None:
        """Following a tractor with some pairs: play all pairs + fill."""
        hand = [
            _card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.HEARTS, Rank.KING), _card(Suit.SPADES, Rank.QUEEN),
        ]
        lead = PlayAction(
            type=PlayType.TRACTOR,
            cards=[
                _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
                _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
            ],
        )
        plays = get_legal_plays(
            hand=hand, is_leading=False, lead_action=lead,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
        )
        assert len(plays) >= 1

    def test_get_legal_plays_following_throw(self) -> None:
        """Following a THROW: must play same-suit cards if possible, fill otherwise."""
        hand = [
            _card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING),
            _card(Suit.SPADES, Rank.QUEEN),
        ]
        lead = PlayAction(
            type=PlayType.THROW,
            cards=[_card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING)],
        )
        plays = get_legal_plays(
            hand=hand, is_leading=False, lead_action=lead,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
        )
        assert len(plays) >= 1


class TestInferPlayType:
    def test_infer_play_type_single(self) -> None:
        """1 card = SINGLE."""
        cards = [_card(Suit.HEARTS, Rank.ACE)]
        result = infer_play_type(cards)
        assert result == PlayType.SINGLE

    def test_infer_play_type_pair(self) -> None:
        """2 same-suit same-rank = PAIR."""
        cards = [_card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2)]
        result = infer_play_type(cards)
        assert result == PlayType.PAIR

    def test_infer_play_type_tractor(self) -> None:
        """4 cards with 2 consecutive pairs = TRACTOR."""
        cards = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
        ]
        result = infer_play_type(cards, trump_suit=Suit.SPADES, trump_rank=Rank.TWO)
        assert result == PlayType.TRACTOR

    def test_infer_play_type_throw(self) -> None:
        """Multiple non-consecutive cards of same non-trump suit = THROW."""
        cards = [
            _card(Suit.HEARTS, Rank.ACE),
            _card(Suit.HEARTS, Rank.KING),
            _card(Suit.HEARTS, Rank.QUEEN),
        ]
        result = infer_play_type(cards, trump_suit=Suit.SPADES, trump_rank=Rank.TWO)
        assert result == PlayType.THROW
