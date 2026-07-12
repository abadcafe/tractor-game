"""Play comparison rules."""

from __future__ import annotations

from .cards import Card, Rank, Suit
from .decompose import decompose, non_trump_rank_order
from .ordering import effective_suit, trump_rank_order
from .types import EffectiveSuit


def _compare_same_suit(
    a_cards: list[Card],
    b_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
    is_trump: bool,
) -> int:
    """
    Compare two plays that share the same effective suit (spec 8.3-8.4).

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
        # Use trump_rank_order (comparator) which distinguishes
        # sub-types:
        #   trump-suit level=80, other-suit level=70, etc.
        # rank_order_for_effective_suit only gives组内 position (1-15),
        # which
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
        sub.pair_count
        for sub in decompose(cards, trump_suit, trump_rank)
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

    Trump cards can kill a non-trump throw only when their decomposed
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
            a_cards, b_cards, trump_suit, trump_rank, is_trump=True
        )
    return _compare_same_suit(
        a_cards, b_cards, trump_suit, trump_rank, is_trump=False
    )
