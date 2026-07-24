"""Black-box tests for the public round scoring rules."""

from __future__ import annotations

from server.game.rules import scoring
from server.game.rules.cards import Rank, Suit
from tests.support import card


def test_all_score_threshold_boundaries() -> None:
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
            last_trick_won_by_defenders=False,
            last_lead_cards=(card("spades", "A"),),
            trump_suit=Suit.HEARTS,
            trump_rank=Rank.TWO,
        )
        assert result.defenders_win is defenders_win
        assert result.declarer_level_gain == declarer_gain
        assert result.defender_level_gain == defender_gain


def test_declarer_shutout_advances_three_levels() -> None:
    result = scoring.score_round(
        defender_points=0,
        bottom_cards=(),
        last_trick_won_by_defenders=False,
        last_lead_cards=(card("spades", "A"),),
        trump_suit=Suit.HEARTS,
        trump_rank=Rank.TWO,
    )

    assert result == scoring.RoundScore(
        total_defender_points=0,
        declarer_level_gain=3,
        defender_level_gain=0,
        defenders_win=False,
        bottom_card_bonus=0,
    )


def test_defenders_win_at_eighty_and_gain_above_threshold() -> None:
    threshold = scoring.score_round(
        defender_points=80,
        bottom_cards=(),
        last_trick_won_by_defenders=False,
        last_lead_cards=(card("spades", "A"),),
        trump_suit=None,
        trump_rank=Rank.TWO,
    )
    advanced = scoring.score_round(
        defender_points=120,
        bottom_cards=(),
        last_trick_won_by_defenders=False,
        last_lead_cards=(card("spades", "A"),),
        trump_suit=None,
        trump_rank=Rank.TWO,
    )

    assert threshold.defenders_win
    assert threshold.defender_level_gain == 0
    assert advanced.defenders_win
    assert advanced.defender_level_gain == 1


def test_defender_last_trick_multiplies_bottom_by_lead_shape() -> None:
    result = scoring.score_round(
        defender_points=70,
        bottom_cards=(card("hearts", "5"),),
        last_trick_won_by_defenders=True,
        last_lead_cards=(
            card("spades", "A", 1),
            card("spades", "A", 2),
        ),
        trump_suit=Suit.HEARTS,
        trump_rank=Rank.TWO,
    )

    assert result.bottom_card_bonus == 20
    assert result.total_defender_points == 90
    assert result.defenders_win


def test_declarer_last_trick_never_scores_the_bottom() -> None:
    result = scoring.score_round(
        defender_points=70,
        bottom_cards=(card("hearts", "5"),),
        last_trick_won_by_defenders=False,
        last_lead_cards=(
            card("spades", "A", 1),
            card("spades", "A", 2),
        ),
        trump_suit=Suit.HEARTS,
        trump_rank=Rank.TWO,
    )

    assert result.bottom_card_bonus == 0
    assert result.total_defender_points == 70
    assert not result.defenders_win


def test_single_pair_and_tractor_use_distinct_bottom_multipliers() -> (
    None
):
    bottom = (card("hearts", "5"),)
    single = scoring.score_round(
        defender_points=0,
        bottom_cards=bottom,
        last_trick_won_by_defenders=True,
        last_lead_cards=(card("spades", "A"),),
        trump_suit=Suit.HEARTS,
        trump_rank=Rank.TWO,
    )
    pair = scoring.score_round(
        defender_points=0,
        bottom_cards=bottom,
        last_trick_won_by_defenders=True,
        last_lead_cards=(
            card("spades", "A", 1),
            card("spades", "A", 2),
        ),
        trump_suit=Suit.HEARTS,
        trump_rank=Rank.TWO,
    )
    tractor = scoring.score_round(
        defender_points=0,
        bottom_cards=bottom,
        last_trick_won_by_defenders=True,
        last_lead_cards=(
            card("spades", "3", 1),
            card("spades", "3", 2),
            card("spades", "4", 1),
            card("spades", "4", 2),
        ),
        trump_suit=Suit.HEARTS,
        trump_rank=Rank.TWO,
    )

    assert single.bottom_card_bonus == 10
    assert pair.bottom_card_bonus == 20
    assert tractor.bottom_card_bonus == 80
