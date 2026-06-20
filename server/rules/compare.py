"""Play comparison rules."""

from __future__ import annotations

from .cards import Card, Rank, Suit
from .decompose import non_trump_rank_order, decompose
from .ordering import effective_suit, trump_rank_order
from .types import EffectiveSuit


def can_win(
    played_cards: list[Card],
    lead_eff: EffectiveSuit,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> bool:
    """Check whether a player's cards are eligible to win the trick (spec 8.2).

    For each card: if effective_suit is neither lead_eff nor "trump" -> False.
    Otherwise True.  When lead_eff is "trump", only trump cards are eligible.
    """
    for card in played_cards:
        eff = effective_suit(card, trump_suit, trump_rank)
        if eff != lead_eff and eff != "trump":
            return False
    return True

def _compare_same_suit(
    a_cards: list[Card],
    b_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
    is_trump: bool,
) -> int:
    """Compare two plays that share the same effective suit (spec 8.3-8.4).

    Uses decompose to extract sub-plays, then compares:
    1. Max sub_level (tractor > pair > single)
    2. Same level: max rank of the highest-level sub-plays

    Returns >0 if a wins, <0 if b wins, 0 if tie.
    """
    a_subs = decompose(a_cards, trump_suit, trump_rank)
    b_subs = decompose(b_cards, trump_suit, trump_rank)

    if not a_subs and not b_subs:
        return 0
    if not a_subs:
        return -1
    if not b_subs:
        return 1

    # Find max sub_level for each
    a_max_level = max(s.sub_level for s in a_subs)
    b_max_level = max(s.sub_level for s in b_subs)

    if a_max_level != b_max_level:
        return a_max_level - b_max_level

    # Same max level: compare max rank of the highest-level sub-plays
    a_high_subs = [s for s in a_subs if s.sub_level == a_max_level]
    b_high_subs = [s for s in b_subs if s.sub_level == b_max_level]

    # Get the max rank order across all highest-level sub-plays
    if is_trump:
        # Use trump_rank_order (comparator) which distinguishes sub-types:
        #   trump-suit level=80, other-suit level=70, etc.
        # rank_order_for_effective_suit only gives组内 position (1-15), which
        # collapses different sub-types at the same rank.
        a_max_rank = max(
            trump_rank_order(c, trump_suit, trump_rank)
            for s in a_high_subs
            for c in s.cards
        )
        b_max_rank = max(
            trump_rank_order(c, trump_suit, trump_rank)
            for s in b_high_subs
            for c in s.cards
        )
    else:
        a_max_rank = max(
            non_trump_rank_order(c.rank, trump_rank)
            for s in a_high_subs
            for c in s.cards
        )
        b_max_rank = max(
            non_trump_rank_order(c.rank, trump_rank)
            for s in b_high_subs
            for c in s.cards
        )

    return a_max_rank - b_max_rank

def compare_plays(
    a_cards: list[Card],
    b_cards: list[Card],
    lead_eff: EffectiveSuit,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> int:
    """Compare two plays using sub-level decomposition (spec 8.3-8.4).

    Returns >0 if a wins, <0 if b wins, 0 if tie.

    1. can_win eligibility gating (spec 8.2)
    2. Trump vs non-trump
    3. Same suit: decompose-based comparison
    """
    a_eligible = can_win(a_cards, lead_eff, trump_suit, trump_rank)
    b_eligible = can_win(b_cards, lead_eff, trump_suit, trump_rank)

    if a_eligible and not b_eligible:
        return 1
    if b_eligible and not a_eligible:
        return -1
    if not a_eligible and not b_eligible:
        return 0

    # Both eligible: determine effective suit groups
    a_all_trump = all(
        effective_suit(c, trump_suit, trump_rank) == "trump" for c in a_cards
    )
    b_all_trump = all(
        effective_suit(c, trump_suit, trump_rank) == "trump" for c in b_cards
    )

    if a_all_trump and not b_all_trump:
        return 1  # trump beats non-trump
    if b_all_trump and not a_all_trump:
        return -1

    if a_all_trump and b_all_trump:
        return _compare_same_suit(a_cards, b_cards, trump_suit, trump_rank, is_trump=True)

    # Both lead-suit (non-trump)
    return _compare_same_suit(a_cards, b_cards, trump_suit, trump_rank, is_trump=False)
