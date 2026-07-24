"""Pure round scoring rules."""

from __future__ import annotations

from dataclasses import dataclass

from ._decompose import decompose
from .cards import Card, Rank, Suit

__all__ = ["RoundScore", "score_round"]


@dataclass(frozen=True, slots=True)
class RoundScore:
    """Score and raw level gains produced by one round."""

    total_defender_points: int
    declarer_level_gain: int
    defender_level_gain: int
    defenders_win: bool
    bottom_card_bonus: int


def score_round(
    *,
    defender_points: int,
    bottom_cards: tuple[Card, ...],
    last_trick_won_by_defenders: bool,
    last_lead_cards: tuple[Card, ...],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> RoundScore:
    """Calculate bottom multiplication and raw level gains."""
    bottom_base = sum(card.points for card in bottom_cards)
    bottom_bonus = 0
    if last_trick_won_by_defenders:
        bottom_bonus = bottom_base * _ambush_multiplier(
            last_lead_cards,
            trump_suit,
            trump_rank,
        )
    total = defender_points + bottom_bonus
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
        bottom_card_bonus=bottom_bonus,
    )


def _ambush_multiplier(
    lead_cards: tuple[Card, ...],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> int:
    subs = decompose(
        list(lead_cards),
        trump_suit,
        trump_rank,
    )
    if not subs:
        return 2
    return max(
        2
        if sub.pair_count == 0
        else 4
        if sub.pair_count == 1
        else 2 ** len(sub.cards)
        for sub in subs
    )
