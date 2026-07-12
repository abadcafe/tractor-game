"""Tests for rules.compare public interface."""

from typing import Literal

from server.game.rules.cards import POINTS_MAP, Card, Rank, Suit
from server.game.rules.compare import compare_plays


def _card(suit: Suit, rank: Rank, deck: Literal[1, 2] = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit,
        rank=rank,
        points=POINTS_MAP[rank],
    )


class TestComparePlays:
    def test_lead_suit_beats_off_suit_padding(self) -> None:
        lead = [_card(Suit.HEARTS, Rank.FIVE)]
        lead_suit_play = [_card(Suit.HEARTS, Rank.THREE)]
        off_suit_play = [_card(Suit.DIAMONDS, Rank.ACE)]

        result = compare_plays(
            lead_suit_play,
            off_suit_play,
            lead,
            Suit.SPADES,
            Rank.TWO,
        )

        assert result > 0

    def test_matching_single_trump_kills_non_trump_single(self) -> None:
        lead = [_card(Suit.HEARTS, Rank.ACE)]
        trump_play = [_card(Suit.SPADES, Rank.THREE)]
        lead_suit_play = [_card(Suit.HEARTS, Rank.KING)]

        result = compare_plays(
            trump_play,
            lead_suit_play,
            lead,
            Suit.SPADES,
            Rank.TWO,
        )

        assert result > 0

    def test_structurally_invalid_trump_kill_cannot_win(self) -> None:
        lead = [
            _card(Suit.HEARTS, Rank.ACE, 1),
            _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.HEARTS, Rank.KING, 1),
            _card(Suit.HEARTS, Rank.QUEEN, 1),
        ]
        valid_kill = [
            _card(Suit.SPADES, Rank.FOUR, 1),
            _card(Suit.SPADES, Rank.FOUR, 2),
            _card(Suit.SPADES, Rank.FIVE, 1),
            _card(Suit.SPADES, Rank.SIX, 1),
        ]
        invalid_big_cards = [
            _card(Suit.JOKER, Rank.BIG_JOKER, 1),
            _card(Suit.JOKER, Rank.SMALL_JOKER, 1),
            _card(Suit.SPADES, Rank.ACE, 1),
            _card(Suit.SPADES, Rank.KING, 1),
        ]

        result = compare_plays(
            invalid_big_cards,
            valid_kill,
            lead,
            Suit.SPADES,
            Rank.TWO,
        )

        assert result < 0

    def test_matching_trump_kills_compare_by_main_pattern(
        self,
    ) -> None:
        lead = [
            _card(Suit.HEARTS, Rank.ACE, 1),
            _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.HEARTS, Rank.KING, 1),
            _card(Suit.HEARTS, Rank.QUEEN, 1),
        ]
        high_pair_kill = [
            _card(Suit.SPADES, Rank.KING, 1),
            _card(Suit.SPADES, Rank.KING, 2),
            _card(Suit.SPADES, Rank.THREE, 1),
            _card(Suit.SPADES, Rank.FOUR, 1),
        ]
        low_pair_with_big_jokers = [
            _card(Suit.SPADES, Rank.QUEEN, 1),
            _card(Suit.SPADES, Rank.QUEEN, 2),
            _card(Suit.JOKER, Rank.BIG_JOKER, 1),
            _card(Suit.JOKER, Rank.SMALL_JOKER, 1),
        ]

        result = compare_plays(
            high_pair_kill,
            low_pair_with_big_jokers,
            lead,
            Suit.SPADES,
            Rank.TWO,
        )

        assert result > 0

    def test_all_single_throw_kills_compare_by_highest_trump(
        self,
    ) -> None:
        lead = [
            _card(Suit.HEARTS, Rank.ACE),
            _card(Suit.HEARTS, Rank.KING),
            _card(Suit.HEARTS, Rank.QUEEN),
        ]
        big_joker_kill = [
            _card(Suit.JOKER, Rank.BIG_JOKER),
            _card(Suit.SPADES, Rank.THREE),
            _card(Suit.SPADES, Rank.FOUR),
        ]
        small_joker_kill = [
            _card(Suit.JOKER, Rank.SMALL_JOKER),
            _card(Suit.SPADES, Rank.ACE),
            _card(Suit.SPADES, Rank.KING),
        ]

        result = compare_plays(
            big_joker_kill,
            small_joker_kill,
            lead,
            Suit.SPADES,
            Rank.TWO,
        )

        assert result > 0

    def test_same_highest_trump_returns_tie_for_play_order(
        self,
    ) -> None:
        lead = [
            _card(Suit.HEARTS, Rank.ACE),
            _card(Suit.HEARTS, Rank.KING),
        ]
        first_kill = [
            _card(Suit.JOKER, Rank.BIG_JOKER, 1),
            _card(Suit.SPADES, Rank.THREE, 1),
        ]
        later_kill = [
            _card(Suit.JOKER, Rank.BIG_JOKER, 2),
            _card(Suit.SPADES, Rank.FOUR, 1),
        ]

        result = compare_plays(
            first_kill,
            later_kill,
            lead,
            Suit.SPADES,
            Rank.TWO,
        )

        assert result == 0
