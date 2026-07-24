"""Black-box tests for the complete public play facade."""

from __future__ import annotations

from server.foundation.result import Ok
from server.game.rules import play
from server.game.rules.cards import Rank, Suit
from tests.support import card


def test_failed_throw_is_accepted_as_forced_smallest_subplay() -> None:
    king = card("spades", "K")
    queen = card("spades", "Q")

    result = play.resolve_lead(
        hand=(king, queen),
        attempted=(king, queen),
        trump_suit=Suit.HEARTS,
        trump_rank=Rank.TWO,
        other_hands=((card("spades", "A"),),),
    )

    assert isinstance(result, Ok)
    assert result.value.cards == (queen,)


def test_unbeatable_throw_keeps_the_complete_attempt() -> None:
    attempted = (
        card("spades", "A"),
        card("spades", "K"),
    )

    result = play.resolve_lead(
        hand=attempted,
        attempted=attempted,
        trump_suit=Suit.HEARTS,
        trump_rank=Rank.TWO,
        other_hands=((), (), ()),
    )

    assert isinstance(result, Ok)
    assert result.value.cards == attempted


def test_closed_hints_return_complete_small_follow_set() -> None:
    heart_ace = card("hearts", "A")
    heart_king = card("hearts", "K")

    hints = play.closed_hints(
        hand=(
            heart_ace,
            heart_king,
            card("clubs", "Q"),
        ),
        lead=(card("hearts", "Q"),),
        trump_suit=Suit.SPADES,
        trump_rank=Rank.TWO,
    )

    assert hints == ((heart_king,), (heart_ace,))


def test_closed_hints_return_empty_when_set_is_not_bounded() -> None:
    hand = (
        card("spades", "A"),
        card("spades", "K"),
        card("spades", "Q"),
        card("clubs", "A"),
        card("clubs", "K"),
        card("clubs", "Q"),
        card("diamonds", "A"),
        card("diamonds", "K"),
    )
    lead = (
        card("hearts", "3", 1),
        card("hearts", "3", 2),
    )

    hints = play.closed_hints(
        hand,
        lead,
        trump_suit=None,
        trump_rank=Rank.TWO,
    )

    assert hints == ()


def test_hand_sort_and_effective_suit_share_trump_semantics() -> None:
    big_joker = card("joker", "BJ")
    main_two = card("hearts", "2")
    side_ace = card("spades", "A")
    cards = (side_ace, main_two, big_joker)

    assert play.sort_hand(
        cards,
        Suit.HEARTS,
        Rank.TWO,
    ) == (big_joker, main_two, side_ace)
    assert (
        play.effective_suit(
            main_two,
            Suit.HEARTS,
            Rank.TWO,
        )
        == "trump"
    )
    assert play.trump_order(
        big_joker,
        Suit.HEARTS,
        Rank.TWO,
    ) > play.trump_order(
        main_two,
        Suit.HEARTS,
        Rank.TWO,
    )
