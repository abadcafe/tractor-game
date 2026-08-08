"""Black-box tests for the public round scoring rules."""

from __future__ import annotations

from server.game.rules import scoring
from server.game.rules.cards import Card, Rank, Suit
from tests.support import TestRank, TestSuit, card


def test_score_round_covers_every_level_threshold() -> None:
    cases = (
        (0, False, 3, 0),
        (1, False, 2, 0),
        (39, False, 2, 0),
        (40, False, 1, 0),
        (79, False, 1, 0),
        (80, True, 0, 0),
        (119, True, 0, 0),
        (120, True, 0, 1),
        (159, True, 0, 1),
        (160, True, 0, 2),
        (200, True, 0, 3),
        (240, True, 0, 4),
    )

    for points, defenders_win, declarer_gain, defender_gain in cases:
        result = scoring.score_round(
            defender_points=points,
            bottom_cards=(),
            bottom_winning_cards=None,
            trump_suit=Suit.HEARTS,
            trump_rank=Rank.TWO,
        )
        assert result.defenders_win is defenders_win
        assert result.declarer_level_gain == declarer_gain
        assert result.defender_level_gain == defender_gain


def test_no_bottom_capture_preserves_bottom_facts() -> None:
    result = scoring.score_round(
        defender_points=70,
        bottom_cards=(card("hearts", "5"), card("spades", "K")),
        bottom_winning_cards=None,
        trump_suit=Suit.HEARTS,
        trump_rank=Rank.TWO,
    )

    assert result == scoring.RoundScore(
        total_defender_points=70,
        declarer_level_gain=1,
        defender_level_gain=0,
        defenders_win=False,
        bottom_base_points=15,
        bottom_multiplier=None,
        bottom_points=0,
    )


def test_score_round_applies_side_discount_to_every_pattern_size() -> (
    None
):
    bottom = (card("hearts", "5"),)
    cases: tuple[tuple[tuple[Card, ...], int, int], ...] = (
        ((card("spades", "A"),), 1, 5),
        (
            (card("spades", "A", 1), card("spades", "A", 2)),
            2,
            10,
        ),
        (_pairs("spades", ("3", "4")), 8, 40),
        (_pairs("spades", ("3", "4", "5")), 32, 160),
    )

    for winning_cards, multiplier, bottom_points in cases:
        result = scoring.score_round(
            defender_points=0,
            bottom_cards=bottom,
            bottom_winning_cards=winning_cards,
            trump_suit=Suit.CLUBS,
            trump_rank=Rank.TWO,
        )
        assert result.bottom_multiplier == multiplier
        assert result.bottom_points == bottom_points


def test_score_round_uses_full_power_for_every_trump_pattern_size() -> (
    None
):
    bottom = (card("hearts", "5"),)
    cases: tuple[tuple[tuple[Card, ...], int, int], ...] = (
        ((card("clubs", "A"),), 2, 10),
        (
            (card("clubs", "A", 1), card("clubs", "A", 2)),
            4,
            20,
        ),
        (_pairs("clubs", ("3", "4")), 16, 80),
        (_pairs("clubs", ("3", "4", "5")), 64, 320),
    )

    for winning_cards, multiplier, bottom_points in cases:
        result = scoring.score_round(
            defender_points=0,
            bottom_cards=bottom,
            bottom_winning_cards=winning_cards,
            trump_suit=Suit.CLUBS,
            trump_rank=Rank.TWO,
        )
        assert result.bottom_multiplier == multiplier
        assert result.bottom_points == bottom_points


def test_score_round_does_not_cap_a_four_pair_tractor() -> None:
    result = scoring.score_round(
        defender_points=0,
        bottom_cards=(card("hearts", "5"),),
        bottom_winning_cards=_pairs("clubs", ("3", "4", "5", "6")),
        trump_suit=Suit.CLUBS,
        trump_rank=Rank.TWO,
    )

    assert result.bottom_multiplier == 256
    assert result.bottom_points == 1280


def test_score_round_uses_fixed_suited_trump_topology() -> None:
    bottom = (card("hearts", "5"),)
    cases: tuple[tuple[tuple[Card, ...], Rank, int], ...] = (
        (_pairs("clubs", ("3", "A")), Rank.TWO, 4),
        (_pairs("clubs", ("4", "6")), Rank.FIVE, 16),
        (_pairs("clubs", ("A",)) + _pair("spades", "2"), Rank.TWO, 16),
        (_pair("spades", "2") + _pair("clubs", "2"), Rank.TWO, 16),
        (_pair("clubs", "2") + _pair("joker", "SJ"), Rank.TWO, 16),
        (_pair("joker", "SJ") + _pair("joker", "BJ"), Rank.TWO, 16),
        (_pair("spades", "2") + _pair("hearts", "2"), Rank.TWO, 4),
    )

    for winning_cards, trump_rank, multiplier in cases:
        result = scoring.score_round(
            defender_points=0,
            bottom_cards=bottom,
            bottom_winning_cards=winning_cards,
            trump_suit=Suit.CLUBS,
            trump_rank=trump_rank,
        )
        assert result.bottom_multiplier == multiplier


def test_score_round_uses_fixed_no_trump_topology() -> None:
    bottom = (card("hearts", "5"),)
    cases: tuple[tuple[tuple[Card, ...], int], ...] = (
        (_pair("spades", "2") + _pair("hearts", "2"), 4),
        (_pair("spades", "2") + _pair("joker", "SJ"), 16),
        (_pair("joker", "SJ") + _pair("joker", "BJ"), 16),
    )

    for winning_cards, multiplier in cases:
        result = scoring.score_round(
            defender_points=0,
            bottom_cards=bottom,
            bottom_winning_cards=winning_cards,
            trump_suit=None,
            trump_rank=Rank.TWO,
        )
        assert result.bottom_multiplier == multiplier


def test_screenshot_regression_uses_jack_queen_as_maximum() -> None:
    winning_cards = (
        *_pair("clubs", "J"),
        *_pair("clubs", "Q"),
        *_pair("clubs", "2"),
        card("joker", "BJ"),
    )
    bottom = (
        card("spades", "3"),
        card("spades", "4"),
        card("spades", "5"),
        card("spades", "6"),
        card("spades", "10"),
        card("spades", "Q"),
        card("hearts", "K"),
        card("hearts", "4"),
    )

    result = scoring.score_round(
        defender_points=125,
        bottom_cards=bottom,
        bottom_winning_cards=winning_cards,
        trump_suit=Suit.CLUBS,
        trump_rank=Rank.TWO,
    )

    assert result.bottom_base_points == 25
    assert result.bottom_multiplier == 16
    assert result.bottom_points == 400
    assert result.total_defender_points == 525
    assert result.defender_level_gain == 11


def test_score_round_uses_largest_pattern_inside_a_throw() -> None:
    winning_cards = (
        *_pairs("spades", ("3", "4")),
        *_pair("spades", "A"),
        card("spades", "Q"),
    )

    result = scoring.score_round(
        defender_points=0,
        bottom_cards=(card("hearts", "5"),),
        bottom_winning_cards=winning_cards,
        trump_suit=Suit.CLUBS,
        trump_rank=Rank.TWO,
    )

    assert result.bottom_multiplier == 8
    assert result.bottom_points == 40


def _pair(suit: TestSuit, rank: TestRank) -> tuple[Card, Card]:
    return (
        card(suit, rank, 1),
        card(suit, rank, 2),
    )


def _pairs(
    suit: TestSuit, ranks: tuple[TestRank, ...]
) -> tuple[Card, ...]:
    return tuple(item for rank in ranks for item in _pair(suit, rank))
