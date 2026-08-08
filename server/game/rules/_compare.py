"""Private play comparison algorithms."""

from __future__ import annotations

from ._ordering import EffectiveSuit, effective_suit, trump_order
from ._patterns import analyze_patterns
from .cards import Card, Rank, Suit


def _compare_same_suit(
    a_cards: list[Card],
    b_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> int:
    """
    Compare two plays that share the same effective suit (spec 8.3-8.4).

    Uses fixed-topology pattern analysis, then compares:
    1. Maximum pattern level (tractor > pair > single)
    2. Same level: strongest card in the highest-level patterns

    Returns >0 if a wins, <0 if b wins, 0 if tie.
    """
    a_subs = analyze_patterns(a_cards, trump_suit, trump_rank).patterns
    b_subs = analyze_patterns(b_cards, trump_suit, trump_rank).patterns

    a_max_level = max(pattern.level for pattern in a_subs)
    b_max_level = max(pattern.level for pattern in b_subs)

    if a_max_level != b_max_level:
        return a_max_level - b_max_level

    a_high_subs = [
        pattern for pattern in a_subs if pattern.level == a_max_level
    ]
    b_high_subs = [
        pattern for pattern in b_subs if pattern.level == b_max_level
    ]

    a_max_rank = max(
        trump_order(card, trump_suit, trump_rank)
        for pattern in a_high_subs
        for card in pattern.cards
    )
    b_max_rank = max(
        trump_order(card, trump_suit, trump_rank)
        for pattern in b_high_subs
        for card in pattern.cards
    )

    return a_max_rank - b_max_rank


def _all_effective_suit(
    cards: list[Card],
    suit: EffectiveSuit,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> bool:
    return all(
        effective_suit(card, trump_suit, trump_rank) == suit
        for card in cards
    )


def _shape_signature(
    cards: list[Card], trump_suit: Suit | None, trump_rank: Rank
) -> tuple[int, ...]:
    return tuple(
        pattern.pair_count
        for pattern in analyze_patterns(
            cards, trump_suit, trump_rank
        ).patterns
    )


def _matches_lead_structure(
    played_cards: list[Card],
    lead_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> bool:
    return _shape_signature(
        played_cards, trump_suit, trump_rank
    ) == _shape_signature(lead_cards, trump_suit, trump_rank)


def _can_win_against_lead(
    played_cards: list[Card],
    lead_cards: list[Card],
    lead_eff: EffectiveSuit,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> bool:
    if _all_effective_suit(
        played_cards, lead_eff, trump_suit, trump_rank
    ):
        return True
    if lead_eff != "trump" and _all_effective_suit(
        played_cards, "trump", trump_suit, trump_rank
    ):
        return _matches_lead_structure(
            played_cards, lead_cards, trump_suit, trump_rank
        )
    return False


def compare_plays(
    a_cards: list[Card],
    b_cards: list[Card],
    lead_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> int:
    """
    Compare two trick plays using the actual leading cards.

    Trump cards can kill a non-trump throw only when their analyzed
    structure exactly matches the leading structure. A structurally
    invalid kill is treated as padding and cannot win.
    """
    if not lead_cards:
        return 0
    lead_eff = effective_suit(lead_cards[0], trump_suit, trump_rank)
    a_eligible = _can_win_against_lead(
        a_cards, lead_cards, lead_eff, trump_suit, trump_rank
    )
    b_eligible = _can_win_against_lead(
        b_cards, lead_cards, lead_eff, trump_suit, trump_rank
    )

    if a_eligible and not b_eligible:
        return 1
    if b_eligible and not a_eligible:
        return -1
    if not a_eligible and not b_eligible:
        return 0

    a_all_trump = _all_effective_suit(
        a_cards, "trump", trump_suit, trump_rank
    )
    b_all_trump = _all_effective_suit(
        b_cards, "trump", trump_suit, trump_rank
    )

    if a_all_trump and not b_all_trump:
        return 1
    if b_all_trump and not a_all_trump:
        return -1
    if a_all_trump and b_all_trump:
        return _compare_same_suit(
            a_cards, b_cards, trump_suit, trump_rank
        )
    return _compare_same_suit(a_cards, b_cards, trump_suit, trump_rank)
