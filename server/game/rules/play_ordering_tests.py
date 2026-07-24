"""Black-box tests for the public trump-ordering contract."""

from server.game.rules import play
from server.game.rules.cards import Rank, Suit
from tests.support import card


def test_trump_order_has_the_complete_strength_hierarchy() -> None:
    cards = (
        card("joker", "BJ"),
        card("joker", "SJ"),
        card("hearts", "2"),
        card("spades", "2"),
        card("hearts", "A"),
        card("spades", "A"),
    )

    strengths = tuple(
        play.trump_order(item, Suit.HEARTS, Rank.TWO) for item in cards
    )

    assert strengths == (100, 90, 80, 70, 59, 12)


def test_side_level_cards_are_equal_under_no_trump() -> None:
    values = {
        play.trump_order(
            card(suit, "2"),
            None,
            Rank.TWO,
        )
        for suit in ("spades", "hearts", "clubs", "diamonds")
    }

    assert values == {70}


def test_effective_suit_groups_every_kind_of_trump() -> None:
    values = tuple(
        play.effective_suit(item, Suit.HEARTS, Rank.TWO)
        for item in (
            card("joker", "SJ"),
            card("spades", "2"),
            card("hearts", "A"),
            card("spades", "A"),
        )
    )

    assert values == ("trump", "trump", "trump", Suit.SPADES)


def test_no_trump_keeps_all_four_side_suits_distinct() -> None:
    assert tuple(
        play.effective_suit(card(suit, "A"), None, Rank.TWO)
        for suit in ("spades", "hearts", "clubs", "diamonds")
    ) == (
        Suit.SPADES,
        Suit.HEARTS,
        Suit.CLUBS,
        Suit.DIAMONDS,
    )


def test_hand_sort_matches_the_complete_player_display_order() -> None:
    unsorted = (
        card("diamonds", "A"),
        card("hearts", "K"),
        card("clubs", "2"),
        card("joker", "SJ"),
        card("spades", "2"),
        card("joker", "BJ"),
        card("hearts", "2"),
        card("spades", "A"),
        card("hearts", "A"),
    )

    sorted_cards = play.sort_hand(
        unsorted,
        Suit.HEARTS,
        Rank.TWO,
    )

    assert tuple(item.id for item in sorted_cards) == tuple(
        item.id
        for item in (
            card("joker", "BJ"),
            card("joker", "SJ"),
            card("hearts", "2"),
            card("spades", "2"),
            card("clubs", "2"),
            card("hearts", "A"),
            card("hearts", "K"),
            card("spades", "A"),
            card("diamonds", "A"),
        )
    )


def test_no_trump_hand_sort_has_four_complete_side_suit_groups() -> (
    None
):
    unsorted = (
        card("diamonds", "K"),
        card("clubs", "A"),
        card("hearts", "K"),
        card("spades", "A"),
        card("diamonds", "A"),
        card("clubs", "K"),
        card("hearts", "A"),
        card("spades", "K"),
    )

    sorted_cards = play.sort_hand(unsorted, None, Rank.TWO)

    assert tuple((item.suit, item.rank) for item in sorted_cards) == (
        (Suit.SPADES, Rank.ACE),
        (Suit.SPADES, Rank.KING),
        (Suit.HEARTS, Rank.ACE),
        (Suit.HEARTS, Rank.KING),
        (Suit.CLUBS, Rank.ACE),
        (Suit.CLUBS, Rank.KING),
        (Suit.DIAMONDS, Rank.ACE),
        (Suit.DIAMONDS, Rank.KING),
    )
