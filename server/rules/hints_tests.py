"""Tests for rules.hints public interface."""

from typing import Literal

from server.result import Ok, Rejected
from server.rules.cards import POINTS_MAP, Card, Rank, Suit
from server.rules.hints import (
    TOO_MANY_PLAY_HINTS,
    get_legal_play_hints,
    sort_play_action_hints,
)
from server.rules.rejections import TooManyPlayHintsRejected


def _card(suit: Suit, rank: Rank, deck: Literal[1, 2] = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit,
        rank=rank,
        points=POINTS_MAP[rank],
    )


def _hint_id_sets(hints: list[list[Card]]) -> set[frozenset[str]]:
    return {frozenset(card.id for card in hint) for hint in hints}


class TestGetLegalPlayHints:
    def test_no_lead_cards_returns_no_hints(self) -> None:
        """
        No lead cards means leading/free-form play, so no closed hint
        set.
        """
        hand = [
            _card(Suit.HEARTS, Rank.ACE),
            _card(Suit.HEARTS, Rank.KING),
        ]
        result = get_legal_play_hints(
            hand,
            None,
            Suit.SPADES,
            Rank.TWO,
            max_hints=5,
        )

        assert isinstance(result, Ok)
        assert result.value == []

    def test_following_single_enumerates_all_same_suit_cards(
        self,
    ) -> None:
        """
        Following a single enumerates every same-suit card, not just the
        first.
        """
        hand = [
            _card(Suit.HEARTS, Rank.ACE),
            _card(Suit.HEARTS, Rank.KING),
            _card(Suit.SPADES, Rank.KING),
        ]
        lead = [_card(Suit.HEARTS, Rank.QUEEN)]
        result = get_legal_play_hints(
            hand,
            lead,
            Suit.SPADES,
            Rank.TWO,
            max_hints=5,
        )

        assert isinstance(result, Ok)
        assert _hint_id_sets(result.value) == {
            frozenset({"D1-hearts-A"}),
            frozenset({"D1-hearts-K"}),
        }

    def test_following_single_no_suit_enumerates_all_cards(
        self,
    ) -> None:
        """
        No effective lead-suit card means every single card is a legal
        hint.
        """
        hand = [
            _card(Suit.SPADES, Rank.KING),
            _card(Suit.CLUBS, Rank.QUEEN),
            _card(Suit.DIAMONDS, Rank.JACK),
        ]
        lead = [_card(Suit.HEARTS, Rank.QUEEN)]
        result = get_legal_play_hints(
            hand,
            lead,
            Suit.SPADES,
            Rank.TWO,
            max_hints=5,
        )

        assert isinstance(result, Ok)
        assert _hint_id_sets(result.value) == {
            frozenset({"D1-spades-K"}),
            frozenset({"D1-clubs-Q"}),
            frozenset({"D1-diamonds-J"}),
        }

    def test_pair_plus_single_enumerates_all_suit_fill(
        self,
    ) -> None:
        """
        After a required pair, all same-suit filler choices are
        enumerated.
        """
        hand = [
            _card(Suit.HEARTS, Rank.ACE, 1),
            _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.HEARTS, Rank.KING),
            _card(Suit.HEARTS, Rank.QUEEN),
            _card(Suit.SPADES, Rank.JACK),
        ]
        lead = [
            _card(Suit.HEARTS, Rank.THREE, 1),
            _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR),
        ]
        result = get_legal_play_hints(
            hand,
            lead,
            Suit.SPADES,
            Rank.TWO,
            max_hints=5,
        )

        assert isinstance(result, Ok)
        assert _hint_id_sets(result.value) == {
            frozenset({"D1-hearts-A", "D2-hearts-A", "D1-hearts-K"}),
            frozenset({"D1-hearts-A", "D2-hearts-A", "D1-hearts-Q"}),
        }

    def test_not_enough_suit_enumerates_all_off_suit_fill(
        self,
    ) -> None:
        """
        When suit cards are exhausted, all off-suit filler choices are
        enumerated.
        """
        hand = [
            _card(Suit.HEARTS, Rank.ACE, 1),
            _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.HEARTS, Rank.KING),
            _card(Suit.SPADES, Rank.QUEEN),
            _card(Suit.CLUBS, Rank.JACK),
        ]
        lead = [
            _card(Suit.HEARTS, Rank.THREE, 1),
            _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1),
            _card(Suit.HEARTS, Rank.FOUR, 2),
        ]
        result = get_legal_play_hints(
            hand,
            lead,
            Suit.SPADES,
            Rank.TWO,
            max_hints=5,
        )

        assert isinstance(result, Ok)
        assert _hint_id_sets(result.value) == {
            frozenset(
                {
                    "D1-hearts-A",
                    "D2-hearts-A",
                    "D1-hearts-K",
                    "D1-spades-Q",
                }
            ),
            frozenset(
                {
                    "D1-hearts-A",
                    "D2-hearts-A",
                    "D1-hearts-K",
                    "D1-clubs-J",
                }
            ),
        }

    def test_following_hints_reject_when_too_many(self) -> None:
        """
        The max_hints + 1-th unique hint rejects before full
        enumeration.
        """
        hand = [
            _card(Suit.CLUBS, Rank.ACE),
            _card(Suit.CLUBS, Rank.KING),
            _card(Suit.CLUBS, Rank.QUEEN),
            _card(Suit.CLUBS, Rank.JACK),
            _card(Suit.CLUBS, Rank.TEN),
            _card(Suit.CLUBS, Rank.NINE),
        ]
        lead = [_card(Suit.HEARTS, Rank.QUEEN)]
        result = get_legal_play_hints(
            hand,
            lead,
            Suit.SPADES,
            Rank.TWO,
            max_hints=5,
        )

        assert isinstance(result, Rejected)
        assert isinstance(result, TooManyPlayHintsRejected)
        assert result.reason == TOO_MANY_PLAY_HINTS

    def test_following_empty_lead_cards(self) -> None:
        """
        Following with empty lead_cards -> returns empty (wait for
        lead).
        """
        hand = [_card(Suit.HEARTS, Rank.ACE)]
        result = get_legal_play_hints(
            hand,
            [],
            Suit.SPADES,
            Rank.TWO,
            max_hints=5,
        )

        assert isinstance(result, Ok)
        assert result.value == []

    def test_following_lead_cards_none(self) -> None:
        """Following with lead_cards=None -> returns empty."""
        hand = [_card(Suit.HEARTS, Rank.ACE)]
        result = get_legal_play_hints(
            hand,
            None,
            Suit.SPADES,
            Rank.TWO,
            max_hints=5,
        )

        assert isinstance(result, Ok)
        assert result.value == []

    def test_sort_play_action_hints_orders_from_small_to_large(
        self,
    ) -> None:
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
