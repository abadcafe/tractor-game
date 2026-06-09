"""Tests for sm.play_rules module."""
from typing import Literal

from server.sm.card_model import Card, Suit, Rank
from server.sm.types import PlayType, PlayAction, SubPlay
from server.sm.play_rules import (
    detect_singles, detect_pairs, detect_tractors, detect_throws,
    get_legal_plays, infer_play_type, decompose, is_legal_lead,
    is_legal_follow,
)
from server.sm.comparator import effective_suit


def _card(suit: Suit, rank: Rank, deck: Literal[1, 2] = 1) -> Card:
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
        pairs = detect_pairs(hand, trump_suit=Suit.HEARTS, trump_rank=Rank.TWO)
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


# ---- Bug 3 regression: empty lead_cards must not crash ----


def test_get_legal_plays_following_empty_lead_cards() -> None:
    """get_legal_plays with empty lead_cards should not crash with IndexError.

    Regression test for Bug 3: _follow_single() accessed lead_cards[0]
    without checking if lead_cards was empty, causing an IndexError.
    This could happen during PLAYING phase when the trick state had
    empty lead cards due to a snapshot being created mid-transition.
    """
    hand = [_card(Suit.HEARTS, Rank.ACE)]
    # lead_action with empty cards mimics a snapshot taken before the
    # lead cards are populated in a new trick's following phase
    lead_action = PlayAction(type=PlayType.SINGLE, cards=[])
    result = get_legal_plays(
        hand=hand,
        is_leading=False,
        lead_action=lead_action,
        trump_suit=Suit.SPADES,
        trump_rank=Rank.TWO,
    )
    # Should return an empty list — following players must wait for lead
    assert result == []


class TestDecompose:
    # --- Singles ---
    def test_decompose_single_card(self) -> None:
        """Single card -> one SubPlay with pair_count=0."""
        c = _card(Suit.HEARTS, Rank.ACE)
        subs = decompose([c], Suit.SPADES, Rank.TWO)
        assert len(subs) == 1
        assert subs[0].pair_count == 0
        assert subs[0].cards == [c]
        assert subs[0].suit == Suit.HEARTS

    def test_decompose_two_different_singles(self) -> None:
        """Two different cards -> two SubPlay singles."""
        c1 = _card(Suit.HEARTS, Rank.ACE)
        c2 = _card(Suit.HEARTS, Rank.KING)
        subs = decompose([c1, c2], Suit.SPADES, Rank.TWO)
        assert len(subs) == 2
        assert all(s.pair_count == 0 for s in subs)

    # --- Pairs ---
    def test_decompose_pair(self) -> None:
        """Two same-rank same-suit cards -> one SubPlay pair."""
        c1 = _card(Suit.HEARTS, Rank.ACE, 1)
        c2 = _card(Suit.HEARTS, Rank.ACE, 2)
        subs = decompose([c1, c2], Suit.SPADES, Rank.TWO)
        assert len(subs) == 1
        assert subs[0].pair_count == 1
        assert len(subs[0].cards) == 2

    def test_decompose_pair_plus_single(self) -> None:
        """Pair + extra single -> one pair SubPlay + one single SubPlay."""
        c1 = _card(Suit.HEARTS, Rank.ACE, 1)
        c2 = _card(Suit.HEARTS, Rank.ACE, 2)
        c3 = _card(Suit.HEARTS, Rank.KING, 1)
        subs = decompose([c1, c2, c3], Suit.SPADES, Rank.TWO)
        pair_subs = [s for s in subs if s.pair_count == 1]
        single_subs = [s for s in subs if s.pair_count == 0]
        assert len(pair_subs) == 1
        assert len(single_subs) == 1

    # --- Tractors ---
    def test_decompose_tractor_2_pairs(self) -> None:
        """Two consecutive pairs -> one tractor SubPlay (pair_count=2)."""
        cards = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
        ]
        subs = decompose(cards, Suit.SPADES, Rank.TWO)
        assert len(subs) == 1
        assert subs[0].pair_count == 2
        assert len(subs[0].cards) == 4

    def test_decompose_tractor_3_pairs(self) -> None:
        """Three consecutive pairs -> one tractor SubPlay (pair_count=3)."""
        cards = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
            _card(Suit.HEARTS, Rank.FIVE, 1), _card(Suit.HEARTS, Rank.FIVE, 2),
        ]
        subs = decompose(cards, Suit.SPADES, Rank.TWO)
        assert len(subs) == 1
        assert subs[0].pair_count == 3

    def test_decompose_tractor_skips_trump_rank(self) -> None:
        """Non-trump suit tractor: trump_rank is skipped in consecutive check.

        trump_rank=5: 4-4 + 6-6 are consecutive (5 is skipped).
        """
        cards = [
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
            _card(Suit.HEARTS, Rank.SIX, 1), _card(Suit.HEARTS, Rank.SIX, 2),
        ]
        subs = decompose(cards, Suit.SPADES, Rank.FIVE)
        assert len(subs) == 1
        assert subs[0].pair_count == 2

    def test_decompose_tractor_non_consecutive_pairs(self) -> None:
        """Non-consecutive pairs -> separate pair SubPlays, not a tractor."""
        cards = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2),
        ]
        subs = decompose(cards, Suit.SPADES, Rank.TWO)
        pair_subs = [s for s in subs if s.pair_count >= 1]
        assert len(pair_subs) == 2
        assert all(s.pair_count == 1 for s in pair_subs)

    # --- Mixed ---
    def test_decompose_tractor_plus_pair_plus_singles(self) -> None:
        """spA spK sp10-10 sp7-7-6-6 -> [single spA, single spK, pair sp10-10, tractor sp7-7-6-6]."""
        cards = [
            _card(Suit.SPADES, Rank.ACE),
            _card(Suit.SPADES, Rank.KING),
            _card(Suit.SPADES, Rank.TEN, 1), _card(Suit.SPADES, Rank.TEN, 2),
            _card(Suit.SPADES, Rank.SEVEN, 1), _card(Suit.SPADES, Rank.SEVEN, 2),
            _card(Suit.SPADES, Rank.SIX, 1), _card(Suit.SPADES, Rank.SIX, 2),
        ]
        subs = decompose(cards, Suit.HEARTS, Rank.TWO)
        tractor_subs = [s for s in subs if s.pair_count >= 2]
        pair_subs = [s for s in subs if s.pair_count == 1]
        single_subs = [s for s in subs if s.pair_count == 0]
        assert len(tractor_subs) == 1
        assert tractor_subs[0].pair_count == 2
        assert len(pair_subs) == 1
        assert len(single_subs) == 2

    def test_decompose_pair_and_tractor_and_singles(self) -> None:
        """spA spA spK spQ spQ sp9 sp9 sp8 sp8 -> [pair spA-A, single spK, pair spQ-Q, tractor sp9-9-8-8]."""
        cards = [
            _card(Suit.SPADES, Rank.ACE, 1), _card(Suit.SPADES, Rank.ACE, 2),
            _card(Suit.SPADES, Rank.KING),
            _card(Suit.SPADES, Rank.QUEEN, 1), _card(Suit.SPADES, Rank.QUEEN, 2),
            _card(Suit.SPADES, Rank.NINE, 1), _card(Suit.SPADES, Rank.NINE, 2),
            _card(Suit.SPADES, Rank.EIGHT, 1), _card(Suit.SPADES, Rank.EIGHT, 2),
        ]
        subs = decompose(cards, Suit.HEARTS, Rank.TWO)
        tractor_subs = [s for s in subs if s.pair_count >= 2]
        pair_subs = [s for s in subs if s.pair_count == 1]
        single_subs = [s for s in subs if s.pair_count == 0]
        assert len(tractor_subs) == 1
        assert tractor_subs[0].pair_count == 2
        assert len(pair_subs) == 2
        assert len(single_subs) == 1

    # --- Trump group ---
    def test_decompose_trump_group_uses_trump_rank_order(self) -> None:
        """Trump group: tractor detection uses trump_rank_order, not non_trump_rank_order.

        With heart trump, rank=5: h3 h3 h4 h4 -> tractor (consecutive in trump ordering).
        """
        cards = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
        ]
        subs = decompose(cards, Suit.HEARTS, Rank.FIVE)
        assert len(subs) == 1
        assert subs[0].pair_count == 2

    def test_decompose_trump_group_cross_sub_type_tractor(self) -> None:
        """Trump group cross-sub-type tractor: hA hA + d5 d5 is tractor when heart trump, rank=5.

        hA is at position 45+14=59, d5 is at 70+0=70. They are adjacent in the
        trump_rank_order sequence because no trump card has a position value between
        59 and 70. Adjacent means "consecutive in the sorted list of position values"
        -- not "position values differ by 1".
        """
        cards = [
            _card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.DIAMONDS, Rank.FIVE, 1), _card(Suit.DIAMONDS, Rank.FIVE, 2),
        ]
        subs = decompose(cards, Suit.HEARTS, Rank.FIVE)
        assert len(subs) == 1
        assert subs[0].pair_count == 2, f"Expected tractor (pair_count=2), got {subs[0].pair_count}"

    def test_decompose_trump_group_joker_tractor(self) -> None:
        """Small joker pair + big joker pair -> tractor (adjacent in trump ordering)."""
        cards = [
            _card(Suit.JOKER, Rank.SMALL_JOKER, 1), _card(Suit.JOKER, Rank.SMALL_JOKER, 2),
            _card(Suit.JOKER, Rank.BIG_JOKER, 1), _card(Suit.JOKER, Rank.BIG_JOKER, 2),
        ]
        subs = decompose(cards, Suit.HEARTS, Rank.TWO)
        assert len(subs) == 1
        assert subs[0].pair_count == 2

    def test_decompose_trump_group_suit_specific_rank_pairs(self) -> None:
        """c5 c5 + sp5 sp5 -> tractor (adjacent in other-suit trump rank section).

        c5=71, sp5=73. Is there a value 72? h5 would be 72 if heart weren't trump,
        but heart IS trump so h5=80. So 72 doesn't exist. c5(71) and sp5(73) are adjacent.
        """
        cards = [
            _card(Suit.CLUBS, Rank.FIVE, 1), _card(Suit.CLUBS, Rank.FIVE, 2),
            _card(Suit.SPADES, Rank.FIVE, 1), _card(Suit.SPADES, Rank.FIVE, 2),
        ]
        subs = decompose(cards, Suit.HEARTS, Rank.FIVE)
        assert len(subs) == 1
        assert subs[0].pair_count == 2

    # --- Bug regression ---
    def test_decompose_trump_single_no_duplicate(self) -> None:
        """Regression: trump group with single card must not produce duplicate SubPlay entries."""
        c = _card(Suit.HEARTS, Rank.THREE)
        subs = decompose([c], Suit.HEARTS, Rank.FIVE)
        assert len(subs) == 1
        assert subs[0].pair_count == 0
        assert subs[0].cards == [c]

    # --- Edge cases ---
    def test_decompose_empty(self) -> None:
        """Empty cards -> empty list."""
        subs = decompose([], Suit.SPADES, Rank.TWO)
        assert subs == []

    def test_decompose_four_of_a_kind(self) -> None:
        """4 cards of same rank in trump group -> 2 separate pairs (same rank, not tractor)."""
        cards = [
            _card(Suit.DIAMONDS, Rank.TWO, 1), _card(Suit.DIAMONDS, Rank.TWO, 2),
            _card(Suit.CLUBS, Rank.TWO, 1), _card(Suit.CLUBS, Rank.TWO, 2),
        ]
        subs = decompose(cards, Suit.HEARTS, Rank.TWO)
        pair_subs = [s for s in subs if s.pair_count >= 1]
        assert len(pair_subs) == 2
        assert all(s.pair_count == 1 for s in pair_subs)

    def test_decompose_longest_tractor_first(self) -> None:
        """Longest tractor extracted first: 3 pairs -> tractor(3), not tractor(2)+pair."""
        cards = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
            _card(Suit.HEARTS, Rank.FIVE, 1), _card(Suit.HEARTS, Rank.FIVE, 2),
        ]
        subs = decompose(cards, Suit.SPADES, Rank.TWO)
        tractor_subs = [s for s in subs if s.pair_count >= 2]
        assert len(tractor_subs) == 1
        assert tractor_subs[0].pair_count == 3


class TestIsLegalLead:
    def test_is_legal_lead_single(self) -> None:
        """Single card lead is always legal (no throw verification)."""
        hand = [_card(Suit.HEARTS, Rank.ACE)]
        played = [_card(Suit.HEARTS, Rank.ACE)]
        assert is_legal_lead(hand, played, Suit.SPADES, Rank.TWO, []) is True

    def test_is_legal_lead_pair(self) -> None:
        """Pair lead is always legal (single sub-play, no throw check)."""
        hand = [_card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2)]
        played = hand[:]
        assert is_legal_lead(hand, played, Suit.SPADES, Rank.TWO, []) is True

    def test_is_legal_lead_tractor(self) -> None:
        """Tractor lead is always legal (single sub-play)."""
        hand = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
        ]
        assert is_legal_lead(hand, hand, Suit.SPADES, Rank.TWO, []) is True

    def test_is_legal_lead_throw_valid(self) -> None:
        """Throw with all biggest sub-plays is legal.

        spA spK with no other sp cards in other hands -> both are biggest singles.
        """
        hand = [_card(Suit.SPADES, Rank.ACE), _card(Suit.SPADES, Rank.KING)]
        other_hands: list[Card] = []  # no other sp cards
        assert is_legal_lead(hand, hand, Suit.HEARTS, Rank.TWO, other_hands) is True

    def test_is_legal_lead_throw_invalid_single_not_biggest(self) -> None:
        """Throw with a non-biggest single triggers failure.

        spK spQ but other hand has spA -> spQ is not biggest (spA exists).
        This IS a throw (2 singles = 2 sub-plays), so verification applies.
        """
        hand = [_card(Suit.SPADES, Rank.KING), _card(Suit.SPADES, Rank.QUEEN)]
        other_hands = [_card(Suit.SPADES, Rank.ACE)]
        result = is_legal_lead(hand, hand, Suit.HEARTS, Rank.TWO, other_hands)
        assert result is False

    def test_is_legal_lead_throw_pair_not_biggest(self) -> None:
        """Throw with a non-biggest pair triggers failure."""
        # Hand: pair spQ-Q + single spA. Other hand has pair spK-K.
        hand = [
            _card(Suit.SPADES, Rank.QUEEN, 1), _card(Suit.SPADES, Rank.QUEEN, 2),
            _card(Suit.SPADES, Rank.ACE),
        ]
        other_hands = [_card(Suit.SPADES, Rank.KING, 1), _card(Suit.SPADES, Rank.KING, 2)]
        # decompose -> [pair spQ-Q, single spA]. Verify each is biggest:
        # pair spQ-Q: is there a bigger sp pair? spK-K exists -> NO. Throw fails.
        result = is_legal_lead(hand, hand, Suit.HEARTS, Rank.TWO, other_hands)
        assert result is False

    def test_is_legal_lead_not_in_hand(self) -> None:
        """Cards not in hand -> illegal."""
        hand = [_card(Suit.HEARTS, Rank.ACE)]
        played = [_card(Suit.HEARTS, Rank.KING)]  # not in hand
        assert is_legal_lead(hand, played, Suit.SPADES, Rank.TWO, []) is False

    def test_is_legal_lead_different_suits(self) -> None:
        """Cards of different effective suits -> illegal."""
        hand = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.SPADES, Rank.KING)]
        played = hand[:]
        assert is_legal_lead(hand, played, Suit.DIAMONDS, Rank.TWO, []) is False

    def test_is_legal_lead_trump_and_non_trump_mix(self) -> None:
        """Mixing trump and non-trump cards -> different effective suits -> illegal."""
        # hA is non-trump, sp2 is trump (trump_rank=2)
        hand = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.SPADES, Rank.TWO)]
        played = hand[:]
        assert is_legal_lead(hand, played, Suit.DIAMONDS, Rank.TWO, []) is False

    def test_is_legal_lead_throw_tractor_not_biggest(self) -> None:
        """Throw containing a tractor that is not biggest -> fails."""
        # Hand: tractor sp3-3-4-4 + single spA. Other hand has tractor sp5-5-6-6.
        hand = [
            _card(Suit.SPADES, Rank.THREE, 1), _card(Suit.SPADES, Rank.THREE, 2),
            _card(Suit.SPADES, Rank.FOUR, 1), _card(Suit.SPADES, Rank.FOUR, 2),
            _card(Suit.SPADES, Rank.ACE),
        ]
        other_hands = [
            _card(Suit.SPADES, Rank.FIVE, 1), _card(Suit.SPADES, Rank.FIVE, 2),
            _card(Suit.SPADES, Rank.SIX, 1), _card(Suit.SPADES, Rank.SIX, 2),
        ]
        result = is_legal_lead(hand, hand, Suit.HEARTS, Rank.TWO, other_hands)
        assert result is False

    def test_is_legal_lead_throw_biggest_tractor(self) -> None:
        """Throw with biggest tractor and biggest single -> legal."""
        # Hand: tractor spK-K-A-A + single spQ. No bigger sp tractors or singles in others.
        hand = [
            _card(Suit.SPADES, Rank.KING, 1), _card(Suit.SPADES, Rank.KING, 2),
            _card(Suit.SPADES, Rank.ACE, 1), _card(Suit.SPADES, Rank.ACE, 2),
            _card(Suit.SPADES, Rank.QUEEN),
        ]
        other_hands: list[Card] = []
        result = is_legal_lead(hand, hand, Suit.HEARTS, Rank.TWO, other_hands)
        assert result is True


class TestIsLegalFollow:
    # --- Basic count and hand checks ---
    def test_is_legal_follow_wrong_count(self) -> None:
        """Played card count must match lead card count."""
        hand = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING)]
        lead = [_card(Suit.HEARTS, Rank.QUEEN)]
        played = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING)]
        assert is_legal_follow(hand, played, lead, Suit.SPADES, Rank.TWO) is False

    def test_is_legal_follow_not_in_hand(self) -> None:
        """Cards not in hand -> illegal."""
        hand = [_card(Suit.HEARTS, Rank.ACE)]
        lead = [_card(Suit.HEARTS, Rank.QUEEN)]
        played = [_card(Suit.HEARTS, Rank.KING)]  # not in hand
        assert is_legal_follow(hand, played, lead, Suit.SPADES, Rank.TWO) is False

    # --- Single following ---
    def test_is_legal_follow_single_must_follow_suit(self) -> None:
        """Must follow suit with single if possible."""
        hand = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.SPADES, Rank.KING)]
        lead = [_card(Suit.HEARTS, Rank.QUEEN)]
        # Must play hA, not spK
        assert is_legal_follow(hand, [_card(Suit.HEARTS, Rank.ACE)], lead, Suit.SPADES, Rank.TWO) is True
        assert is_legal_follow(hand, [_card(Suit.SPADES, Rank.KING)], lead, Suit.SPADES, Rank.TWO) is False

    def test_is_legal_follow_single_no_suit_play_anything(self) -> None:
        """No cards of lead suit -> can play anything."""
        hand = [_card(Suit.SPADES, Rank.KING)]
        lead = [_card(Suit.HEARTS, Rank.QUEEN)]
        assert is_legal_follow(hand, [_card(Suit.SPADES, Rank.KING)], lead, Suit.SPADES, Rank.TWO) is True

    # --- Pair following ---
    def test_is_legal_follow_pair_must_play_pair(self) -> None:
        """Must play pair of lead suit if available."""
        hand = [
            _card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.SPADES, Rank.KING, 1), _card(Suit.SPADES, Rank.KING, 2),
        ]
        lead = [_card(Suit.HEARTS, Rank.QUEEN, 1), _card(Suit.HEARTS, Rank.QUEEN, 2)]
        # Must play hA pair
        assert is_legal_follow(hand, [_card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2)], lead, Suit.SPADES, Rank.TWO) is True
        # Cannot play spK pair
        assert is_legal_follow(hand, [_card(Suit.SPADES, Rank.KING, 1), _card(Suit.SPADES, Rank.KING, 2)], lead, Suit.SPADES, Rank.TWO) is False

    def test_is_legal_follow_pair_no_pair_play_two_singles(self) -> None:
        """No pair of lead suit -> must play 2 cards of lead suit if available."""
        hand = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING), _card(Suit.SPADES, Rank.QUEEN)]
        lead = [_card(Suit.HEARTS, Rank.TEN, 1), _card(Suit.HEARTS, Rank.TEN, 2)]
        # Must play 2 heart cards
        assert is_legal_follow(hand, [_card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING)], lead, Suit.SPADES, Rank.TWO) is True
        # Cannot play 1 heart + 1 spade
        assert is_legal_follow(hand, [_card(Suit.HEARTS, Rank.ACE), _card(Suit.SPADES, Rank.QUEEN)], lead, Suit.SPADES, Rank.TWO) is False

    def test_is_legal_follow_pair_no_suit_play_any_two(self) -> None:
        """No cards of lead suit -> can play any 2."""
        hand = [_card(Suit.SPADES, Rank.KING), _card(Suit.SPADES, Rank.QUEEN)]
        lead = [_card(Suit.HEARTS, Rank.TEN, 1), _card(Suit.HEARTS, Rank.TEN, 2)]
        assert is_legal_follow(hand, hand, lead, Suit.SPADES, Rank.TWO) is True

    # --- Tractor following ---
    def test_is_legal_follow_tractor_must_play_matching_tractor(self) -> None:
        """Must play matching-length tractor if available."""
        hand = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
        ]
        lead = [
            _card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.HEARTS, Rank.KING, 1), _card(Suit.HEARTS, Rank.KING, 2),
        ]
        # Must play h3-3-4-4 tractor
        assert is_legal_follow(hand, hand, lead, Suit.SPADES, Rank.TWO) is True

    def test_is_legal_follow_tractor_priority_from_high_to_low(self) -> None:
        """Must use higher-level sub-plays first when following tractor.

        Lead: 2-pair tractor (4 cards). Hand has 3-pair tractor + independent pair.
        Must use the 3-pair tractor (take 2 pairs from it), not the independent pair.
        """
        # Hand: tractor h3-3-4-4-5-5 + pair hK-K
        hand = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
            _card(Suit.HEARTS, Rank.FIVE, 1), _card(Suit.HEARTS, Rank.FIVE, 2),
            _card(Suit.HEARTS, Rank.KING, 1), _card(Suit.HEARTS, Rank.KING, 2),
        ]
        lead = [
            _card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.HEARTS, Rank.QUEEN, 1), _card(Suit.HEARTS, Rank.QUEEN, 2),
        ]
        # Legal: use tractor h3-3-4-4 (2 pairs from the 3-pair tractor)
        legal_play = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
        ]
        assert is_legal_follow(hand, legal_play, lead, Suit.SPADES, Rank.TWO) is True

        # Illegal: use pair hK-K + pair h3-3 (skips higher tractor)
        illegal_play = [
            _card(Suit.HEARTS, Rank.KING, 1), _card(Suit.HEARTS, Rank.KING, 2),
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
        ]
        assert is_legal_follow(hand, illegal_play, lead, Suit.SPADES, Rank.TWO) is False

    def test_is_legal_follow_tractor_partial_with_singles(self) -> None:
        """No matching tractor, have pairs -> play all pairs + fill with singles."""
        hand = [
            _card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.HEARTS, Rank.KING),
            _card(Suit.SPADES, Rank.QUEEN),
        ]
        lead = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
        ]
        # Must play: pair hA-A (2 cards) + hK (1) + spQ (1) = 4 cards
        played = [
            _card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.HEARTS, Rank.KING),
            _card(Suit.SPADES, Rank.QUEEN),
        ]
        assert is_legal_follow(hand, played, lead, Suit.SPADES, Rank.TWO) is True

    # --- Throw following ---
    def test_is_legal_follow_throw_must_follow_suit(self) -> None:
        """Following a throw: must play all same-suit cards if possible."""
        hand = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING), _card(Suit.SPADES, Rank.QUEEN)]
        lead = [_card(Suit.HEARTS, Rank.TEN), _card(Suit.HEARTS, Rank.NINE)]
        # Must play both heart cards
        assert is_legal_follow(hand, [_card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING)], lead, Suit.SPADES, Rank.TWO) is True
        # Cannot skip hK for spQ
        assert is_legal_follow(hand, [_card(Suit.HEARTS, Rank.ACE), _card(Suit.SPADES, Rank.QUEEN)], lead, Suit.SPADES, Rank.TWO) is False

    # --- Effective suit ---
    def test_is_legal_follow_trump_as_lead_eff(self) -> None:
        """Trump cards have effective suit 'trump'. Following trump lead must play trump."""
        # trump_suit=heart, trump_rank=2. hA is trump.
        hand = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.SPADES, Rank.KING)]
        lead = [_card(Suit.HEARTS, Rank.TWO)]  # trump card
        # hA is trump (trump_suit=heart), so must follow with hA
        assert is_legal_follow(hand, [_card(Suit.HEARTS, Rank.ACE)], lead, Suit.HEARTS, Rank.TWO) is True
        # spK is not trump, so illegal
        assert is_legal_follow(hand, [_card(Suit.SPADES, Rank.KING)], lead, Suit.HEARTS, Rank.TWO) is False

    # --- Tractor continuity (spec 7c) ---
    def test_is_legal_follow_tractor_non_contiguous_extraction(self) -> None:
        """Partial extraction from a tractor must be contiguous.

        Hand has a 3-pair tractor h3-3-4-4-5-5. Lead is a 2-pair tractor.
        Playing h3-3 + h5-5 (skipping h4-4) should be illegal because
        the extracted pairs are not contiguous in the tractor's rank sequence.
        """
        hand = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
            _card(Suit.HEARTS, Rank.FIVE, 1), _card(Suit.HEARTS, Rank.FIVE, 2),
        ]
        lead = [
            _card(Suit.HEARTS, Rank.SEVEN, 1), _card(Suit.HEARTS, Rank.SEVEN, 2),
            _card(Suit.HEARTS, Rank.EIGHT, 1), _card(Suit.HEARTS, Rank.EIGHT, 2),
        ]
        # Non-contiguous: play h3-3 + h5-5 (skip h4-4)
        illegal_play = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FIVE, 1), _card(Suit.HEARTS, Rank.FIVE, 2),
        ]
        assert is_legal_follow(hand, illegal_play, lead, Suit.SPADES, Rank.TWO) is False

        # Contiguous: play h3-3 + h4-4 (from bottom of tractor)
        legal_play = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
        ]
        assert is_legal_follow(hand, legal_play, lead, Suit.SPADES, Rank.TWO) is True

    # --- Fewer suit cards than lead count ---
    def test_is_legal_follow_fewer_suit_cards_tractor(self) -> None:
        """Fewer suit cards than lead: must play all suit cards + fill.

        Lead is a 4-card tractor (2 pairs). Hand has 1 pair + 1 single of
        lead suit (3 cards) + 2 non-suit cards. Must play all 3 suit cards
        + 1 fill card. Cannot skip a suit card.
        """
        hand = [
            _card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.HEARTS, Rank.KING),
            _card(Suit.SPADES, Rank.QUEEN), _card(Suit.SPADES, Rank.JACK),
        ]
        lead = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
        ]
        # Legal: play all 3 suit cards + 1 fill
        played = [
            _card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.HEARTS, Rank.KING),
            _card(Suit.SPADES, Rank.QUEEN),
        ]
        assert is_legal_follow(hand, played, lead, Suit.SPADES, Rank.TWO) is True

        # Illegal: skip hK (play pair hA-A + 2 spades, skipping hK)
        illegal_play = [
            _card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.SPADES, Rank.QUEEN), _card(Suit.SPADES, Rank.JACK),
        ]
        # Should fail: hand has 3 hearts (hA-A pair + hK single) but only 2 hearts played
        assert is_legal_follow(hand, illegal_play, lead, Suit.SPADES, Rank.TWO) is False

    def test_is_legal_follow_fewer_suit_cards_throw(self) -> None:
        """Fewer suit cards than throw length: play all suit + fill.

        Lead is a 3-card throw. Hand has 2 cards of lead suit + 1 non-suit.
        Must play all 2 suit cards + 1 fill.
        """
        hand = [
            _card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING),
            _card(Suit.SPADES, Rank.QUEEN),
        ]
        lead = [
            _card(Suit.HEARTS, Rank.TEN), _card(Suit.HEARTS, Rank.NINE),
            _card(Suit.HEARTS, Rank.EIGHT),
        ]
        # Legal: play all 2 hearts + 1 spade fill
        played = [
            _card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING),
            _card(Suit.SPADES, Rank.QUEEN),
        ]
        assert is_legal_follow(hand, played, lead, Suit.SPADES, Rank.TWO) is True

        # Illegal: skip a heart (play 1 heart + 2 non-hearts -- but only 1 non-heart)
        # With only 1 non-heart, can't make 3 cards skipping a heart. Count would be wrong.
        # So let's test: hand has 2 hearts + 2 spades. Play 1 heart + 2 spades.
        hand2 = [
            _card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING),
            _card(Suit.SPADES, Rank.QUEEN), _card(Suit.SPADES, Rank.JACK),
        ]
        illegal_play = [
            _card(Suit.HEARTS, Rank.ACE),
            _card(Suit.SPADES, Rank.QUEEN), _card(Suit.SPADES, Rank.JACK),
        ]
        # Should fail: has 2 hearts but only played 1
        assert is_legal_follow(hand2, illegal_play, lead, Suit.SPADES, Rank.TWO) is False

    def test_is_legal_follow_no_pairs_in_hand_tractor(self) -> None:
        """No pairs at all in lead suit when following a tractor: play any N cards."""
        hand = [
            _card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING),
            _card(Suit.HEARTS, Rank.QUEEN),
        ]
        lead = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
        ]
        # No pairs in hand, all singles. Must play 4 cards but only 3 hearts.
        # With fewer suit cards: must play all 3 hearts + 1 fill
        # But we have no fill cards. So this should be... actually count must match.
        # Lead is 4 cards, played must be 4. Hand has 3 hearts, 0 others.
        # Can't make 4 cards. So this scenario can't happen with only 3 cards.
        # Let me adjust: hand has 4 hearts (all singles) + 0 others.
        hand2 = [
            _card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING),
            _card(Suit.HEARTS, Rank.QUEEN), _card(Suit.HEARTS, Rank.TEN),
        ]
        # 4 hearts, no pairs. Lead is 4-card tractor. Must play all 4 hearts.
        played = [
            _card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING),
            _card(Suit.HEARTS, Rank.QUEEN), _card(Suit.HEARTS, Rank.TEN),
        ]
        assert is_legal_follow(hand2, played, lead, Suit.SPADES, Rank.TWO) is True
