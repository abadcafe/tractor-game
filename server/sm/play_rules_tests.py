"""Tests for sm.play_rules module."""
from typing import Literal

from .card_model import Card, Suit, Rank
from .play_rules import (
    can_win,
    compare_plays,
    decompose,
    detect_throws,
    get_legal_plays,
    is_legal_follow,
    is_legal_lead,
    resolve_lead_throw,
    sort_play_action_hints,
)
from .result import Ok


def _card(suit: Suit, rank: Rank, deck: Literal[1, 2] = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit, rank=rank,
        is_joker=(suit == Suit.JOKER),
        is_big_joker=(rank == Rank.BIG_JOKER),
        points=0, deck=deck,
    )


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

        hA is at position 45+14=59, d5 is at 70. They are adjacent in the
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
        """c5 c5 + sp5 sp5 -> tractor by same-rank suit structure.

        This structural tractor rule still uses SUIT_OFFSET to decide same-rank
        suit adjacency. It does not mean c5 and sp5 have different trick-winning
        strength; they are equal during comparison.
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
        assert is_legal_lead(hand, hand, Suit.HEARTS, Rank.TWO, [other_hands]) is True

    def test_is_legal_lead_throw_failed_single_still_submittable(self) -> None:
        """A failed throw attempt is still a submittable lead action."""
        hand = [_card(Suit.SPADES, Rank.KING), _card(Suit.SPADES, Rank.QUEEN)]
        other_hands = [_card(Suit.SPADES, Rank.ACE)]
        result = is_legal_lead(hand, hand, Suit.HEARTS, Rank.TWO, [other_hands])
        assert result is True

    def test_resolve_lead_throw_forces_smallest_failed_subplay(self) -> None:
        """Leading resolution accepts failed throws and returns forced cards."""
        hand = [_card(Suit.SPADES, Rank.KING), _card(Suit.SPADES, Rank.QUEEN)]
        other_hands = [_card(Suit.SPADES, Rank.ACE)]

        result = resolve_lead_throw(
            hand,
            hand,
            Suit.HEARTS,
            Rank.TWO,
            [other_hands],
        )

        assert isinstance(result, Ok)
        assert result.value.attempted_cards == hand
        assert result.value.played_cards == [hand[1]]

    def test_detect_throw_does_not_combine_pairs_across_players(self) -> None:
        """A pair only exists when one opponent holds both cards."""
        hand = [
            _card(Suit.SPADES, Rank.KING, 1),
            _card(Suit.SPADES, Rank.KING, 2),
        ]
        other_players_hands = [
            [_card(Suit.SPADES, Rank.ACE, 1)],
            [_card(Suit.SPADES, Rank.ACE, 2)],
            [],
        ]

        throws = detect_throws(
            hand,
            Suit.HEARTS,
            Rank.TWO,
            other_players_hands,
        )

        assert len(throws) == 1
        assert throws[0] == hand

    def test_resolve_throw_akkqq_not_beaten_by_phantom_low_tractor(self) -> None:
        """Scattered lower pairs across opponents cannot force AKKQQ to pick small."""
        hand = [
            _card(Suit.HEARTS, Rank.ACE, 1),
            _card(Suit.HEARTS, Rank.KING, 1),
            _card(Suit.HEARTS, Rank.KING, 2),
            _card(Suit.HEARTS, Rank.QUEEN, 1),
            _card(Suit.HEARTS, Rank.QUEEN, 2),
        ]
        other_players_hands = [
            [
                _card(Suit.HEARTS, Rank.NINE, 1),
                _card(Suit.HEARTS, Rank.NINE, 2),
                _card(Suit.HEARTS, Rank.EIGHT, 1),
            ],
            [
                _card(Suit.HEARTS, Rank.ACE, 2),
                _card(Suit.HEARTS, Rank.TEN, 1),
                _card(Suit.HEARTS, Rank.TEN, 2),
                _card(Suit.HEARTS, Rank.EIGHT, 2),
            ],
            [
                _card(Suit.HEARTS, Rank.JACK, 1),
                _card(Suit.HEARTS, Rank.JACK, 2),
                _card(Suit.HEARTS, Rank.SEVEN, 1),
                _card(Suit.HEARTS, Rank.SEVEN, 2),
            ],
        ]

        result = resolve_lead_throw(
            hand,
            hand,
            Suit.SPADES,
            Rank.FIVE,
            other_players_hands,
        )

        assert isinstance(result, Ok)
        assert result.value.played_cards == hand

    def test_is_legal_lead_throw_pair_not_biggest_still_submittable(self) -> None:
        """A throw containing a non-biggest pair is still submittable."""
        # Hand: pair spQ-Q + single spA. Other hand has pair spK-K.
        hand = [
            _card(Suit.SPADES, Rank.QUEEN, 1), _card(Suit.SPADES, Rank.QUEEN, 2),
            _card(Suit.SPADES, Rank.ACE),
        ]
        other_hands = [_card(Suit.SPADES, Rank.KING, 1), _card(Suit.SPADES, Rank.KING, 2)]
        result = is_legal_lead(hand, hand, Suit.HEARTS, Rank.TWO, [other_hands])
        assert result is True

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

    def test_is_legal_lead_throw_tractor_not_biggest_still_submittable(self) -> None:
        """A throw containing a non-biggest tractor is still submittable."""
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
        result = is_legal_lead(hand, hand, Suit.HEARTS, Rank.TWO, [other_hands])
        assert result is True

    def test_is_legal_lead_throw_biggest_tractor(self) -> None:
        """Throw with biggest tractor and biggest single -> legal."""
        # Hand: tractor spK-K-A-A + single spQ. No bigger sp tractors or singles in others.
        hand = [
            _card(Suit.SPADES, Rank.KING, 1), _card(Suit.SPADES, Rank.KING, 2),
            _card(Suit.SPADES, Rank.ACE, 1), _card(Suit.SPADES, Rank.ACE, 2),
            _card(Suit.SPADES, Rank.QUEEN),
        ]
        other_hands: list[Card] = []
        result = is_legal_lead(hand, hand, Suit.HEARTS, Rank.TWO, [other_hands])
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

    def test_is_legal_follow_no_trump_rank_card_is_not_lead_suit(self) -> None:
        """In no-trump, rank cards are trump and do not satisfy their printed suit."""
        hand = [
            _card(Suit.JOKER, Rank.SMALL_JOKER),
            _card(Suit.HEARTS, Rank.FOUR),
            _card(Suit.CLUBS, Rank.FOUR),
            _card(Suit.DIAMONDS, Rank.FOUR),
            _card(Suit.SPADES, Rank.TWO),
            _card(Suit.HEARTS, Rank.JACK),
            _card(Suit.HEARTS, Rank.FIVE),
            _card(Suit.HEARTS, Rank.TWO),
            _card(Suit.CLUBS, Rank.SIX),
            _card(Suit.CLUBS, Rank.THREE),
        ]
        lead = [_card(Suit.DIAMONDS, Rank.ACE)]
        played = [_card(Suit.HEARTS, Rank.FIVE)]

        assert is_legal_follow(hand, played, lead, None, Rank.FOUR) is True

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


class TestCanWin:
    def test_can_win_all_lead_suit(self) -> None:
        """All cards are lead suit -> can win."""
        cards = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING)]
        assert can_win(cards, Suit.HEARTS, Suit.SPADES, Rank.TWO) is True

    def test_can_win_all_trump(self) -> None:
        """All cards are trump -> can win (trump = lead_eff when lead is trump)."""
        cards = [_card(Suit.JOKER, Rank.BIG_JOKER)]
        assert can_win(cards, "trump", Suit.SPADES, Rank.TWO) is True

    def test_can_win_lead_is_trump_play_trump(self) -> None:
        """Lead is trump, play is trump -> can win."""
        cards = [_card(Suit.SPADES, Rank.ACE)]  # trump when trump_suit=spade
        assert can_win(cards, "trump", Suit.SPADES, Rank.TWO) is True

    def test_can_win_off_suit_non_trump(self) -> None:
        """Card is neither lead suit nor trump -> cannot win."""
        cards = [_card(Suit.DIAMONDS, Rank.ACE)]
        assert can_win(cards, Suit.HEARTS, Suit.SPADES, Rank.TWO) is False

    def test_can_win_off_suit_but_trump(self) -> None:
        """Card is not lead suit but is trump -> can win."""
        # sp2 is trump when trump_suit=spade, trump_rank=2
        cards = [_card(Suit.SPADES, Rank.TWO)]
        assert can_win(cards, Suit.HEARTS, Suit.SPADES, Rank.TWO) is True

    def test_can_win_mixed_lead_and_off_suit(self) -> None:
        """One card is lead suit, one is off-suit non-trump -> cannot win."""
        cards = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.DIAMONDS, Rank.KING)]
        assert can_win(cards, Suit.HEARTS, Suit.SPADES, Rank.TWO) is False

    def test_can_win_mixed_lead_and_trump(self) -> None:
        """One card is lead suit, one is trump -> CAN win.

        Per spec 8.2: any card that is (not lead_suit AND not trump) -> cannot win.
        Trump cards are always OK. So hA + sp2(trump) -> both valid -> can win.
        """
        # hA (lead suit) + sp2 (trump, trump_suit=spade, trump_rank=2)
        cards = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.SPADES, Rank.TWO)]
        assert can_win(cards, Suit.HEARTS, Suit.SPADES, Rank.TWO) is True


class TestComparePlays:
    # --- can_win gating ---
    def test_compare_plays_a_wins_by_eligibility(self) -> None:
        """A can win, B cannot -> A wins."""
        a = [_card(Suit.HEARTS, Rank.THREE)]
        b = [_card(Suit.DIAMONDS, Rank.ACE)]  # off-suit, not trump
        result = compare_plays(a, b, Suit.HEARTS, Suit.SPADES, Rank.TWO)
        assert result > 0

    def test_compare_plays_b_wins_by_eligibility(self) -> None:
        """B can win, A cannot -> B wins."""
        a = [_card(Suit.DIAMONDS, Rank.ACE)]
        b = [_card(Suit.HEARTS, Rank.THREE)]
        result = compare_plays(a, b, Suit.HEARTS, Suit.SPADES, Rank.TWO)
        assert result < 0

    def test_compare_plays_neither_can_win(self) -> None:
        """Neither can win -> tie (0)."""
        a = [_card(Suit.DIAMONDS, Rank.ACE)]
        b = [_card(Suit.CLUBS, Rank.KING)]
        result = compare_plays(a, b, Suit.HEARTS, Suit.SPADES, Rank.TWO)
        assert result == 0

    # --- Trump vs non-trump ---
    def test_compare_plays_trump_beats_non_trump(self) -> None:
        """All-trump play beats all-lead-suit play."""
        # spA is trump (trump_suit=spade)
        a = [_card(Suit.SPADES, Rank.ACE)]
        b = [_card(Suit.HEARTS, Rank.ACE)]
        result = compare_plays(a, b, Suit.HEARTS, Suit.SPADES, Rank.TWO)
        assert result > 0

    # --- Sub-level comparison ---
    def test_compare_plays_pair_beats_single(self) -> None:
        """Pair (level 2) beats single (level 1), even if single has higher rank."""
        # hA pair vs hK single -- pair wins by level
        a = [_card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2)]
        b = [_card(Suit.HEARTS, Rank.KING)]
        result = compare_plays(a, b, Suit.HEARTS, Suit.SPADES, Rank.TWO)
        assert result > 0

    def test_compare_plays_tractor_beats_pair(self) -> None:
        """Tractor (level 3) beats pair (level 2)."""
        a = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
        ]
        b = [_card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2)]
        result = compare_plays(a, b, Suit.HEARTS, Suit.SPADES, Rank.TWO)
        assert result > 0

    def test_compare_plays_same_level_higher_rank_wins(self) -> None:
        """Same sub-level: higher max rank wins."""
        a = [_card(Suit.HEARTS, Rank.ACE)]
        b = [_card(Suit.HEARTS, Rank.KING)]
        result = compare_plays(a, b, Suit.HEARTS, Suit.SPADES, Rank.TWO)
        assert result > 0

    def test_compare_plays_same_level_same_rank_tie(self) -> None:
        """Same sub-level, same max rank -> tie."""
        a = [_card(Suit.HEARTS, Rank.ACE, 1)]
        b = [_card(Suit.HEARTS, Rank.ACE, 2)]
        result = compare_plays(a, b, Suit.HEARTS, Suit.SPADES, Rank.TWO)
        assert result == 0

    def test_compare_plays_both_trump_higher_wins(self) -> None:
        """Both trump: higher trump_rank_order wins."""
        a = [_card(Suit.JOKER, Rank.BIG_JOKER)]
        b = [_card(Suit.JOKER, Rank.SMALL_JOKER)]
        result = compare_plays(a, b, "trump", Suit.SPADES, Rank.TWO)
        assert result > 0

    # --- Trump sub-type comparison (spec 2.3 / 8.4) ---
    def test_compare_plays_trump_suit_level_beats_other_suit_level(self) -> None:
        """Trump-suit level card beats other-suit level card at same rank.

        trump_suit=♥, trump_rank=5:
          ♥5 = 主花色级牌 (spec value=80)
          ♠5 = 其他花色级牌 (spec value=70)
        ♥5 should win.
        """
        a = [_card(Suit.HEARTS, Rank.FIVE)]
        b = [_card(Suit.SPADES, Rank.FIVE)]
        # lead_eff=♠ (spades is trump, so both are trump)
        result = compare_plays(a, b, Suit.SPADES, Suit.HEARTS, Rank.FIVE)
        assert result > 0

    def test_compare_plays_trump_suit_level_beats_diamond_level(self) -> None:
        """Trump-suit level card beats diamond-level card (lowest other-suit level).

        trump_suit=♥, trump_rank=5:
          ♥5 = 80
          ♦5 = 70
        ♥5 should win.
        """
        a = [_card(Suit.HEARTS, Rank.FIVE)]
        b = [_card(Suit.DIAMONDS, Rank.FIVE)]
        result = compare_plays(a, b, Suit.DIAMONDS, Suit.HEARTS, Rank.FIVE)
        assert result > 0

    def test_compare_plays_other_suit_level_cards_tie(self) -> None:
        """Other-suit level cards are equal; earlier play order wins the trick.

        trump_rank=5, trump_suit=♥:
          ♣5 = 70
          ♠5 = 70
        """
        a = [_card(Suit.SPADES, Rank.FIVE)]
        b = [_card(Suit.CLUBS, Rank.FIVE)]
        result = compare_plays(a, b, Suit.HEARTS, Suit.HEARTS, Rank.FIVE)
        assert result == 0

    def test_compare_plays_no_trump_level_cards_tie(self) -> None:
        """In no-trump rounds, all trump-rank suits are equal."""
        a = [_card(Suit.SPADES, Rank.TWO)]
        b = [_card(Suit.HEARTS, Rank.TWO)]
        result = compare_plays(a, b, "trump", None, Rank.TWO)
        assert result == 0

    def test_compare_plays_trump_pair_sub_type_diff(self) -> None:
        """Both trump pairs at same rank, different sub-types.

        trump_suit=♥, trump_rank=5:
          ♥5♥5 = 主花色级牌对子 (max rank = 80)
          ♠5♠5 = 其他花色级牌对子 (max rank = 70)
        ♥5♥5 should win.
        """
        a = [_card(Suit.HEARTS, Rank.FIVE, 1), _card(Suit.HEARTS, Rank.FIVE, 2)]
        b = [_card(Suit.SPADES, Rank.FIVE, 1), _card(Suit.SPADES, Rank.FIVE, 2)]
        result = compare_plays(a, b, Suit.SPADES, Suit.HEARTS, Rank.FIVE)
        assert result > 0

    def test_compare_plays_trump_suit_non_level_vs_other_suit_level(self) -> None:
        """Trump-suit non-level card vs other-suit level card at same rank.

        trump_suit=♥, trump_rank=K:
          ♥A = 主花色非级牌 (spec value=45+14=59)
          ♠K = 其他花色级牌 (spec value=70)
        ♠K should win (70 > 59).
        """
        a = [_card(Suit.HEARTS, Rank.ACE)]
        b = [_card(Suit.SPADES, Rank.KING)]
        result = compare_plays(a, b, Suit.SPADES, Suit.HEARTS, Rank.KING)
        assert result < 0


class TestDetectThrows:
    def test_detect_throws_single_suit_all_biggest(self) -> None:
        """All spade cards are biggest -> one throw option with all spade cards."""
        hand = [_card(Suit.SPADES, Rank.ACE), _card(Suit.SPADES, Rank.KING)]
        other_hands: list[Card] = []
        throws = detect_throws(hand, Suit.HEARTS, Rank.TWO, [other_hands])
        assert len(throws) == 1
        assert set(c.id for c in throws[0]) == {hand[0].id, hand[1].id}

    def test_detect_throws_sub_play_not_biggest(self) -> None:
        """Sub-play not biggest -> no throw for that suit.

        Hand: spK, spQ. Other hand has spA. spQ is not biggest single.
        """
        hand = [_card(Suit.SPADES, Rank.KING), _card(Suit.SPADES, Rank.QUEEN)]
        other_hands = [_card(Suit.SPADES, Rank.ACE)]
        throws = detect_throws(hand, Suit.HEARTS, Rank.TWO, [other_hands])
        assert len(throws) == 0

    def test_detect_throws_pair_biggest(self) -> None:
        """Pair of spA-A is biggest pair -> throw option."""
        hand = [_card(Suit.SPADES, Rank.ACE, 1), _card(Suit.SPADES, Rank.ACE, 2)]
        other_hands: list[Card] = []
        throws = detect_throws(hand, Suit.HEARTS, Rank.TWO, [other_hands])
        assert len(throws) == 1

    def test_detect_throws_pair_not_biggest(self) -> None:
        """Pair of spK-K but spA-A exists in other hands -> not biggest pair -> no throw."""
        hand = [_card(Suit.SPADES, Rank.KING, 1), _card(Suit.SPADES, Rank.KING, 2)]
        other_hands = [_card(Suit.SPADES, Rank.ACE, 1), _card(Suit.SPADES, Rank.ACE, 2)]
        throws = detect_throws(hand, Suit.HEARTS, Rank.TWO, [other_hands])
        assert len(throws) == 0

    def test_detect_throws_tractor_biggest(self) -> None:
        """Biggest tractor in spade -> throw option."""
        hand = [
            _card(Suit.SPADES, Rank.ACE, 1), _card(Suit.SPADES, Rank.ACE, 2),
            _card(Suit.SPADES, Rank.KING, 1), _card(Suit.SPADES, Rank.KING, 2),
        ]
        other_hands: list[Card] = []
        throws = detect_throws(hand, Suit.HEARTS, Rank.TWO, [other_hands])
        assert len(throws) == 1
        assert len(throws[0]) == 4

    def test_detect_throws_mixed_biggest(self) -> None:
        """Throw with tractor + pair + singles, all biggest -> valid."""
        hand = [
            _card(Suit.SPADES, Rank.ACE, 1), _card(Suit.SPADES, Rank.ACE, 2),
            _card(Suit.SPADES, Rank.KING, 1), _card(Suit.SPADES, Rank.KING, 2),
            _card(Suit.SPADES, Rank.QUEEN, 1), _card(Suit.SPADES, Rank.QUEEN, 2),
            _card(Suit.SPADES, Rank.JACK),
        ]
        other_hands: list[Card] = []
        throws = detect_throws(hand, Suit.HEARTS, Rank.TWO, [other_hands])
        assert len(throws) == 1
        assert len(throws[0]) == 7

    def test_detect_throws_trump_allowed(self) -> None:
        """Trump cards CAN throw (spec section 7.4).

        heart is trump. hA hK are trump cards, both biggest trump singles -> valid throw.
        """
        hand = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING)]
        throws = detect_throws(hand, Suit.HEARTS, Rank.TWO, [])
        assert len(throws) == 1
        assert len(throws[0]) == 2

    def test_detect_throws_trump_not_biggest(self) -> None:
        """Trump throw where a sub-play is not biggest -> no throw."""
        # heart is trump. hK hQ but other hand has hA (biggest trump single).
        hand = [_card(Suit.HEARTS, Rank.KING), _card(Suit.HEARTS, Rank.QUEEN)]
        other_hands = [_card(Suit.HEARTS, Rank.ACE)]
        throws = detect_throws(hand, Suit.HEARTS, Rank.TWO, [other_hands])
        assert len(throws) == 0

    def test_detect_throws_multiple_suits(self) -> None:
        """Multiple suits with valid throws -> one throw per suit."""
        hand = [
            _card(Suit.SPADES, Rank.ACE), _card(Suit.SPADES, Rank.KING),
            _card(Suit.DIAMONDS, Rank.ACE), _card(Suit.DIAMONDS, Rank.KING),
        ]
        throws = detect_throws(hand, Suit.HEARTS, Rank.TWO, [])
        assert len(throws) == 2

    def test_detect_throws_single_card_not_throw(self) -> None:
        """Single card per suit -> not a throw (need 2+ sub-plays)."""
        hand = [_card(Suit.SPADES, Rank.ACE)]
        throws = detect_throws(hand, Suit.HEARTS, Rank.TWO, [])
        # Single card -> decompose returns 1 sub-play -> not a throw
        assert len(throws) == 0

    def test_detect_throws_two_singles_valid(self) -> None:
        """Two singles of same suit, both biggest -> valid throw (2 sub-plays)."""
        hand = [_card(Suit.SPADES, Rank.ACE), _card(Suit.SPADES, Rank.KING)]
        throws = detect_throws(hand, Suit.HEARTS, Rank.TWO, [])
        assert len(throws) == 1

    def test_detect_throws_pair_plus_single_biggest(self) -> None:
        """Pair + single of same suit, all biggest -> valid throw."""
        hand = [
            _card(Suit.SPADES, Rank.ACE, 1), _card(Suit.SPADES, Rank.ACE, 2),
            _card(Suit.SPADES, Rank.KING),
        ]
        throws = detect_throws(hand, Suit.HEARTS, Rank.TWO, [])
        assert len(throws) == 1
        assert len(throws[0]) == 3

    def test_detect_throws_verify_from_low_to_high(self) -> None:
        """Throw verification checks from low to high level.

        If a single is not biggest, throw fails immediately without checking pairs/tractors.
        """
        # Hand: pair spA-A + single spQ. Other has spK.
        # single spQ is not biggest (spK exists) -> fail at level 1, don't check pair.
        hand = [
            _card(Suit.SPADES, Rank.ACE, 1), _card(Suit.SPADES, Rank.ACE, 2),
            _card(Suit.SPADES, Rank.QUEEN),
        ]
        other_hands = [_card(Suit.SPADES, Rank.KING)]
        throws = detect_throws(hand, Suit.HEARTS, Rank.TWO, [other_hands])
        assert len(throws) == 0


class TestGetLegalPlays:
    # --- Leading ---
    def test_leading_returns_singles(self) -> None:
        """Leading: each card is a single option."""
        hand = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING)]
        plays = get_legal_plays(hand, True, None, Suit.SPADES, Rank.TWO, [])
        singles = [p for p in plays if len(p) == 1]
        assert len(singles) >= 2

    def test_leading_returns_pairs(self) -> None:
        """Leading: pairs are options."""
        hand = [_card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2)]
        plays = get_legal_plays(hand, True, None, Suit.SPADES, Rank.TWO, [])
        pairs = [p for p in plays if len(p) == 2]
        assert len(pairs) >= 1

    def test_leading_returns_tractors(self) -> None:
        """Leading: tractors are options."""
        hand = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
        ]
        plays = get_legal_plays(hand, True, None, Suit.SPADES, Rank.TWO, [])
        tractors = [p for p in plays if len(p) == 4]
        assert len(tractors) >= 1

    def test_leading_returns_valid_throws(self) -> None:
        """Leading: valid throws (all sub-plays biggest) are options."""
        hand = [_card(Suit.SPADES, Rank.ACE), _card(Suit.SPADES, Rank.KING)]
        plays = get_legal_plays(hand, True, None, Suit.HEARTS, Rank.TWO, [])
        throws = [p for p in plays if len(p) == 2]
        assert len(throws) >= 1

    # --- Following ---
    def test_following_single_must_follow(self) -> None:
        """Following single: must play same suit if possible."""
        hand = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.SPADES, Rank.KING)]
        lead = [_card(Suit.HEARTS, Rank.QUEEN)]
        plays = get_legal_plays(hand, False, lead, Suit.SPADES, Rank.TWO, [])
        # All plays must be heart
        for p in plays:
            assert all(c.suit == Suit.HEARTS for c in p)

    def test_following_single_no_suit(self) -> None:
        """Following single with no matching suit: can play anything."""
        hand = [_card(Suit.SPADES, Rank.KING)]
        lead = [_card(Suit.HEARTS, Rank.QUEEN)]
        plays = get_legal_plays(hand, False, lead, Suit.SPADES, Rank.TWO, [])
        assert len(plays) >= 1

    def test_following_single_no_trump_rank_card_hints_all_cards(self) -> None:
        """No effective lead-suit card means every single card is a legal hint."""
        hand = [
            _card(Suit.JOKER, Rank.SMALL_JOKER),
            _card(Suit.HEARTS, Rank.FOUR),
            _card(Suit.CLUBS, Rank.FOUR),
            _card(Suit.DIAMONDS, Rank.FOUR),
            _card(Suit.SPADES, Rank.TWO),
            _card(Suit.HEARTS, Rank.JACK),
            _card(Suit.HEARTS, Rank.FIVE),
            _card(Suit.HEARTS, Rank.TWO),
            _card(Suit.CLUBS, Rank.SIX),
            _card(Suit.CLUBS, Rank.THREE),
        ]
        lead = [_card(Suit.DIAMONDS, Rank.ACE)]

        plays = get_legal_plays(hand, False, lead, None, Rank.FOUR, [])
        play_ids = {play[0].id for play in plays}

        assert play_ids == {card.id for card in hand}

    def test_following_pair_must_follow(self) -> None:
        """Following pair: must play pair of same suit if available."""
        hand = [
            _card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.SPADES, Rank.KING, 1), _card(Suit.SPADES, Rank.KING, 2),
        ]
        lead = [_card(Suit.HEARTS, Rank.QUEEN, 1), _card(Suit.HEARTS, Rank.QUEEN, 2)]
        plays = get_legal_plays(hand, False, lead, Suit.SPADES, Rank.TWO, [])
        # Must include hA pair option
        has_heart_pair = any(
            len(p) == 2 and all(c.suit == Suit.HEARTS for c in p)
            for p in plays
        )
        assert has_heart_pair

    def test_following_tractor_priority(self) -> None:
        """Following tractor: must use higher-level sub-plays first."""
        # Hand: tractor h3-3-4-4-5-5 + pair hK-K. Lead: 2-pair tractor.
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
        plays = get_legal_plays(hand, False, lead, Suit.SPADES, Rank.TWO, [])
        # All plays must be 4 cards
        for p in plays:
            assert len(p) == 4

    def test_following_empty_lead_cards(self) -> None:
        """Following with empty lead_cards -> returns empty (wait for lead)."""
        hand = [_card(Suit.HEARTS, Rank.ACE)]
        plays = get_legal_plays(hand, False, [], Suit.SPADES, Rank.TWO, [])
        assert plays == []

    def test_following_lead_cards_none(self) -> None:
        """Following with lead_cards=None -> returns empty."""
        hand = [_card(Suit.HEARTS, Rank.ACE)]
        plays = get_legal_plays(hand, False, None, Suit.SPADES, Rank.TWO, [])
        assert plays == []

    def test_sort_play_action_hints_orders_from_small_to_large(self) -> None:
        """Player-facing play hints are sorted weakest first."""
        hints = [
            [_card(Suit.JOKER, Rank.SMALL_JOKER)],
            [_card(Suit.HEARTS, Rank.THREE)],
            [_card(Suit.DIAMONDS, Rank.FIVE)],
            [_card(Suit.SPADES, Rank.TWO)],
        ]

        result = sort_play_action_hints(hints, Suit.HEARTS, Rank.TWO)

        assert [[card.id for card in hint] for hint in result] == [
            ["D1-diamonds-5"],
            ["D1-hearts-3"],
            ["D1-spades-2"],
            ["D1-joker-SJ"],
        ]


# ---- Sub-level comparison edge case ----


class TestSubLevelComparison:
    def test_compare_plays_lower_rank_pair_beats_higher_rank_single(self) -> None:
        """Pair (level 2) beats single (level 1) even when pair has lower rank.

        h3 pair (rank 3) vs hA single (rank A): pair wins because level 2 > level 1.
        The existing test_compare_plays_pair_beats_single uses hA pair vs hK single,
        which doesn't actually test the edge case (pair has higher rank too).
        """
        a = [_card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2)]
        b = [_card(Suit.HEARTS, Rank.ACE)]
        result = compare_plays(a, b, Suit.HEARTS, Suit.SPADES, Rank.TWO)
        assert result > 0  # pair wins despite lower rank
