"""Play validation: enumerate all legal plays from a hand given the current trick state.

Pure functions -- the engine calls these to get legal actions for the current player.

Ported from src/rules/validator.ts.
"""

from __future__ import annotations

from server.engine.card import Card, Suit, Rank
from server.engine.card_utils import RANK_ORDER
from server.engine.types import PlayType, PlayAction
from server.rules.pattern import (
    detect_singles, detect_pairs, detect_tractors, detect_throw_candidates,
    describe_play,
)
from server.rules.follow_rules import get_legal_follows


# ---- Public API ----


def get_legal_plays(
    hand: list[Card],
    current_trick: list[dict],
    trump_suit: Suit,
    trump_rank: Rank,
    is_leading: bool,
    lead_action: PlayAction | None,
    remaining_cards: list[Card] | None = None,
) -> list[PlayAction]:
    """Enumerate all legal plays for the current player.

    Args:
        hand: Current player's hand.
        current_trick: The current trick slots (list of dicts with player_index and cards).
        trump_suit: The trump suit.
        trump_rank: The trump rank (= current level).
        is_leading: True if this player is leading the trick.
        lead_action: The lead play action (None if leading).
        remaining_cards: All cards not yet played (for throw validation).

    Returns:
        List of all legal PlayAction options.
    """
    if not hand:
        return []

    all_slots_empty = all(
        slot.get("cards") is None or slot.get("cards") == []
        for slot in current_trick
    )

    if is_leading or all_slots_empty:
        return get_leading_plays(hand, trump_suit, trump_rank, remaining_cards or [])

    # Following
    if lead_action is None:
        return []

    return get_legal_follows(hand, lead_action, trump_suit, trump_rank)


def get_leading_plays(
    hand: list[Card],
    trump_suit: Suit,
    trump_rank: Rank,
    remaining_cards: list[Card],
) -> list[PlayAction]:
    """Get all legal plays when leading a trick.

    Includes singles, pairs, tractors, and valid throws.
    """
    plays: list[PlayAction] = []

    # Single: every card
    plays.extend(detect_singles(hand))

    # Pair: every identical pair
    plays.extend(detect_pairs(hand))

    # Tractor: consecutive pairs
    plays.extend(detect_tractors(hand, trump_suit, trump_rank))

    # Throw: for each non-trump suit
    for suit in (Suit.HEARTS, Suit.SPADES, Suit.DIAMONDS, Suit.CLUBS):
        if suit == trump_suit:
            continue
        candidates = detect_throw_candidates(hand, suit, trump_suit, trump_rank)
        # Filter throws: each card must be unbeatable by remaining cards of that suit
        for candidate in candidates:
            if _is_throw_valid(candidate, suit, trump_rank, remaining_cards):
                plays.append(candidate)

    return plays


def is_legal_play(action: PlayAction, legal_plays: list[PlayAction]) -> bool:
    """Check if a specific play action is legal.

    Compares by card IDs (set equality).
    """
    action_ids = {c.id for c in action.cards}
    return any(
        action_ids == {c.id for c in legal.cards}
        for legal in legal_plays
    )


def filter_by_type(plays: list[PlayAction], play_type: PlayType) -> list[PlayAction]:
    """Filter legal plays to only those matching a specific play type."""
    return [p for p in plays if p.type == play_type]


def describe_legal_plays(plays: list[PlayAction]) -> list[str]:
    """Get a human-readable list of legal play descriptions.

    Useful for AI prompts.
    """
    return [describe_play(p) for p in plays]


# ---- Private helpers ----


def _is_throw_valid(
    throw_action: PlayAction,
    suit: Suit,
    trump_rank: Rank,
    remaining_cards: list[Card],
) -> bool:
    """Check if a throw is valid: all thrown cards must be the highest remaining
    cards of that suit among all players.
    """
    if not remaining_cards:
        return True

    # Find the highest remaining card of this suit (excluding trump rank)
    remaining_of_suit = [
        c for c in remaining_cards
        if c.suit == suit and c.rank != trump_rank
    ]

    if not remaining_of_suit:
        return True

    # Sort remaining by rank descending
    sorted_remaining = sorted(
        remaining_of_suit,
        key=lambda c: RANK_ORDER[c.rank],
        reverse=True,
    )

    highest_remaining_rank = sorted_remaining[0].rank

    # Each thrown card must be >= the highest remaining card of that suit
    return all(
        RANK_ORDER[c.rank] >= RANK_ORDER[highest_remaining_rank]
        for c in throw_action.cards
    )
