"""Pure round scoring rules."""

from __future__ import annotations

from dataclasses import dataclass

from ._patterns import analyze_patterns
from .cards import Card, Rank, Suit

__all__ = ["RoundScore", "score_round"]


@dataclass(frozen=True, slots=True)
class RoundScore:
    """Score and raw level gains produced by one round."""

    total_defender_points: int
    declarer_level_gain: int
    defender_level_gain: int
    defenders_win: bool
    bottom_base_points: int
    bottom_multiplier: int | None
    bottom_points: int


def score_round(
    *,
    defender_points: int,
    bottom_cards: tuple[Card, ...],
    bottom_winning_cards: tuple[Card, ...] | None,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> RoundScore:
    """Calculate bottom multiplication and raw level gains."""
    bottom_base = sum(card.points for card in bottom_cards)
    bottom_multiplier = None
    bottom_points = 0
    if bottom_winning_cards is not None:
        bottom_multiplier = _bottom_multiplier(
            bottom_winning_cards,
            trump_suit,
            trump_rank,
        )
        bottom_points = bottom_base * bottom_multiplier
    total = defender_points + bottom_points
    if total == 0:
        declarer_gain = 3
        defender_gain = 0
        defenders_win = False
    elif total < 40:
        declarer_gain = 2
        defender_gain = 0
        defenders_win = False
    elif total < 80:
        declarer_gain = 1
        defender_gain = 0
        defenders_win = False
    else:
        declarer_gain = 0
        defender_gain = max(0, (total - 80) // 40)
        defenders_win = True
    return RoundScore(
        total_defender_points=total,
        declarer_level_gain=declarer_gain,
        defender_level_gain=defender_gain,
        defenders_win=defenders_win,
        bottom_base_points=bottom_base,
        bottom_multiplier=bottom_multiplier,
        bottom_points=bottom_points,
    )


def _bottom_multiplier(
    winning_cards: tuple[Card, ...],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> int:
    analysis = analyze_patterns(winning_cards, trump_suit, trump_rank)
    largest_pattern_width = max(
        len(pattern.cards) for pattern in analysis.patterns
    )
    full_multiplier = 1 << largest_pattern_width
    if analysis.effective_suit == "trump":
        return full_multiplier
    return full_multiplier // 2
