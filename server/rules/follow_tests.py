"""Tests for rules.follow public interface."""

from typing import Literal

from server.rules.cards import Card, POINTS_MAP, Suit, Rank
from server.rules.follow import is_legal_follow, is_legal_lead


def _card(suit: Suit, rank: Rank, deck: Literal[1, 2] = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit, rank=rank,
        points=POINTS_MAP[rank],
    )


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
