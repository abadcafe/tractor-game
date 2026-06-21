"""Tests for rules.throw public interface."""

from typing import Literal

from server.result import Ok
from server.rules.cards import POINTS_MAP, Card, Rank, Suit
from server.rules.throw import detect_throws, resolve_lead_throw


def _card(suit: Suit, rank: Rank, deck: Literal[1, 2] = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit,
        rank=rank,
        points=POINTS_MAP[rank],
    )


class TestResolveLeadThrow:
    def test_resolve_lead_throw_forces_smallest_failed_subplay(
        self,
    ) -> None:
        """
        Leading resolution accepts failed throws and returns forced
        cards.
        """
        hand = [
            _card(Suit.SPADES, Rank.KING),
            _card(Suit.SPADES, Rank.QUEEN),
        ]
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

    def test_resolve_throw_akkqq_not_beaten_by_phantom_low_tractor(
        self,
    ) -> None:
        """
        Scattered lower pairs across opponents cannot force AKKQQ to
        pick small.
        """
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


class TestDetectThrows:
    def test_detect_throw_does_not_combine_pairs_across_players(
        self,
    ) -> None:
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

    def test_detect_throws_single_suit_all_biggest(self) -> None:
        """
        All spade cards are biggest -> one throw option with all spade
        cards.
        """
        hand = [
            _card(Suit.SPADES, Rank.ACE),
            _card(Suit.SPADES, Rank.KING),
        ]
        other_hands: list[Card] = []
        throws = detect_throws(
            hand, Suit.HEARTS, Rank.TWO, [other_hands]
        )
        assert len(throws) == 1
        assert set(c.id for c in throws[0]) == {hand[0].id, hand[1].id}

    def test_detect_throws_sub_play_not_biggest(self) -> None:
        """Sub-play not biggest -> no throw for that suit.

        Hand: spK, spQ. Other hand has spA. spQ is not biggest single.
        """
        hand = [
            _card(Suit.SPADES, Rank.KING),
            _card(Suit.SPADES, Rank.QUEEN),
        ]
        other_hands = [_card(Suit.SPADES, Rank.ACE)]
        throws = detect_throws(
            hand, Suit.HEARTS, Rank.TWO, [other_hands]
        )
        assert len(throws) == 0

    def test_detect_throws_pair_biggest(self) -> None:
        """Pair of spA-A is biggest pair -> throw option."""
        hand = [
            _card(Suit.SPADES, Rank.ACE, 1),
            _card(Suit.SPADES, Rank.ACE, 2),
        ]
        other_hands: list[Card] = []
        throws = detect_throws(
            hand, Suit.HEARTS, Rank.TWO, [other_hands]
        )
        assert len(throws) == 1

    def test_detect_throws_pair_not_biggest(self) -> None:
        """
        Pair of spK-K but spA-A exists in other hands -> not biggest
        pair -> no throw.
        """
        hand = [
            _card(Suit.SPADES, Rank.KING, 1),
            _card(Suit.SPADES, Rank.KING, 2),
        ]
        other_hands = [
            _card(Suit.SPADES, Rank.ACE, 1),
            _card(Suit.SPADES, Rank.ACE, 2),
        ]
        throws = detect_throws(
            hand, Suit.HEARTS, Rank.TWO, [other_hands]
        )
        assert len(throws) == 0

    def test_detect_throws_tractor_biggest(self) -> None:
        """Biggest tractor in spade -> throw option."""
        hand = [
            _card(Suit.SPADES, Rank.ACE, 1),
            _card(Suit.SPADES, Rank.ACE, 2),
            _card(Suit.SPADES, Rank.KING, 1),
            _card(Suit.SPADES, Rank.KING, 2),
        ]
        other_hands: list[Card] = []
        throws = detect_throws(
            hand, Suit.HEARTS, Rank.TWO, [other_hands]
        )
        assert len(throws) == 1
        assert len(throws[0]) == 4

    def test_detect_throws_mixed_biggest(self) -> None:
        """Throw with tractor + pair + singles, all biggest -> valid."""
        hand = [
            _card(Suit.SPADES, Rank.ACE, 1),
            _card(Suit.SPADES, Rank.ACE, 2),
            _card(Suit.SPADES, Rank.KING, 1),
            _card(Suit.SPADES, Rank.KING, 2),
            _card(Suit.SPADES, Rank.QUEEN, 1),
            _card(Suit.SPADES, Rank.QUEEN, 2),
            _card(Suit.SPADES, Rank.JACK),
        ]
        other_hands: list[Card] = []
        throws = detect_throws(
            hand, Suit.HEARTS, Rank.TWO, [other_hands]
        )
        assert len(throws) == 1
        assert len(throws[0]) == 7

    def test_detect_throws_trump_allowed(self) -> None:
        """Trump cards CAN throw (spec section 7.4).

        heart is trump. hA hK are trump cards, both biggest trump
        singles -> valid throw.
        """
        hand = [
            _card(Suit.HEARTS, Rank.ACE),
            _card(Suit.HEARTS, Rank.KING),
        ]
        throws = detect_throws(hand, Suit.HEARTS, Rank.TWO, [])
        assert len(throws) == 1
        assert len(throws[0]) == 2

    def test_detect_throws_trump_not_biggest(self) -> None:
        """Trump throw where a sub-play is not biggest -> no throw."""
        # heart is trump. hK hQ but other hand has hA (biggest trump
        # single).
        hand = [
            _card(Suit.HEARTS, Rank.KING),
            _card(Suit.HEARTS, Rank.QUEEN),
        ]
        other_hands = [_card(Suit.HEARTS, Rank.ACE)]
        throws = detect_throws(
            hand, Suit.HEARTS, Rank.TWO, [other_hands]
        )
        assert len(throws) == 0

    def test_detect_throws_multiple_suits(self) -> None:
        """Multiple suits with valid throws -> one throw per suit."""
        hand = [
            _card(Suit.SPADES, Rank.ACE),
            _card(Suit.SPADES, Rank.KING),
            _card(Suit.DIAMONDS, Rank.ACE),
            _card(Suit.DIAMONDS, Rank.KING),
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
        """
        Two singles of same suit, both biggest -> valid throw (2
        sub-plays).
        """
        hand = [
            _card(Suit.SPADES, Rank.ACE),
            _card(Suit.SPADES, Rank.KING),
        ]
        throws = detect_throws(hand, Suit.HEARTS, Rank.TWO, [])
        assert len(throws) == 1

    def test_detect_throws_pair_plus_single_biggest(self) -> None:
        """Pair + single of same suit, all biggest -> valid throw."""
        hand = [
            _card(Suit.SPADES, Rank.ACE, 1),
            _card(Suit.SPADES, Rank.ACE, 2),
            _card(Suit.SPADES, Rank.KING),
        ]
        throws = detect_throws(hand, Suit.HEARTS, Rank.TWO, [])
        assert len(throws) == 1
        assert len(throws[0]) == 3

    def test_detect_throws_verify_from_low_to_high(self) -> None:
        """Throw verification checks from low to high level.

        If a single is not biggest, throw fails immediately without
        checking pairs/tractors.
        """
        # Hand: pair spA-A + single spQ. Other has spK.
        # single spQ is not biggest (spK exists) -> fail at level 1,
        # don't check pair.
        hand = [
            _card(Suit.SPADES, Rank.ACE, 1),
            _card(Suit.SPADES, Rank.ACE, 2),
            _card(Suit.SPADES, Rank.QUEEN),
        ]
        other_hands = [_card(Suit.SPADES, Rank.KING)]
        throws = detect_throws(
            hand, Suit.HEARTS, Rank.TWO, [other_hands]
        )
        assert len(throws) == 0
