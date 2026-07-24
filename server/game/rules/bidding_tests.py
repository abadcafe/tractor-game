"""Black-box tests for the public bidding rules."""

from __future__ import annotations

from server.foundation.result import Ok, Rejected
from server.game.rules import bidding
from server.game.rules.cards import Rank
from tests.support import card


def test_legal_reveals_are_complete_and_ordered_by_strength() -> None:
    spade_two = card("spades", "2")
    heart_twos = (
        card("hearts", "2", 1),
        card("hearts", "2", 2),
    )
    small_jokers = (
        card("joker", "SJ", 1),
        card("joker", "SJ", 2),
    )
    hand = (spade_two, *heart_twos, *small_jokers)

    reveals = bidding.legal_reveals(
        hand,
        Rank.TWO,
        current=None,
    )

    assert tuple(item.cards for item in reveals) == (
        (heart_twos[0],),
        (spade_two,),
        heart_twos,
        small_jokers,
    )


def test_current_declaration_filters_equal_and_weaker_reveals() -> None:
    heart_twos = (
        card("hearts", "2", 1),
        card("hearts", "2", 2),
    )
    small_jokers = (
        card("joker", "SJ", 1),
        card("joker", "SJ", 2),
    )
    current = bidding.Declaration(cards=heart_twos)

    reveals = bidding.legal_reveals(
        (*heart_twos, *small_jokers),
        Rank.TWO,
        current=current,
    )

    assert tuple(item.cards for item in reveals) == (small_jokers,)
    assert bidding.priority(
        reveals[0],
        Rank.TWO,
    ) > bidding.priority(current, Rank.TWO)


def test_validate_reveal_rejects_non_level_and_duplicate_cards() -> (
    None
):
    non_level = bidding.validate_reveal(
        (card("spades", "3"),),
        Rank.TWO,
        current=None,
    )
    repeated = card("hearts", "2")
    duplicate = bidding.validate_reveal(
        (repeated, repeated),
        Rank.TWO,
        current=None,
    )

    assert isinstance(non_level, Rejected)
    assert isinstance(duplicate, Rejected)


def test_stir_requires_a_valid_pair() -> None:
    single = bidding.validate_stir(
        (card("spades", "2"),),
        Rank.TWO,
        current=None,
    )
    pair = bidding.validate_stir(
        (
            card("spades", "2", 1),
            card("spades", "2", 2),
        ),
        Rank.TWO,
        current=None,
    )

    assert isinstance(single, Rejected)
    assert isinstance(pair, Ok)


def test_big_joker_pair_is_the_maximum_declaration() -> None:
    maximum = bidding.Declaration(
        cards=(
            card("joker", "BJ", 1),
            card("joker", "BJ", 2),
        )
    )
    smaller = bidding.Declaration(
        cards=(
            card("joker", "SJ", 1),
            card("joker", "SJ", 2),
        )
    )

    assert bidding.is_maximum(maximum)
    assert not bidding.is_maximum(smaller)
