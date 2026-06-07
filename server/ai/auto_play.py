"""AI auto-play module for selecting plays and bids for non-human players.

Provides baseline random card selection from legal plays. The interface
supports future LLM-based AI integration by keeping all functions pure
(deterministic with seed) and accepting full game context.
"""

import random
from typing import Optional

from server.sm.card_model import Card, Rank, Suit
from server.sm.types import PlayAction


def choose_play(legal_plays: list[PlayAction], seed: Optional[int] = None) -> PlayAction:
    """Select a random play from the list of legal plays.

    Args:
        legal_plays: Non-empty list of legal PlayAction options.
        seed: Random seed for deterministic testing. None uses system randomness.

    Returns:
        A randomly selected PlayAction from legal_plays.

    Raises:
        ValueError: If legal_plays is empty.
    """
    if not legal_plays:
        raise ValueError("No legal plays available")
    rng = random.Random(seed)
    return rng.choice(legal_plays)


def choose_bid(
    valid_levels: list[Rank],
    current_level: Rank,
    seed: Optional[int] = None,
) -> Optional[Rank]:
    """Choose whether to bid and at what level.

    Simple strategy: 40% chance to pass, 60% chance to bid the lowest valid level.

    Args:
        valid_levels: Non-empty list of Rank values that are valid bids.
        current_level: The current trump rank for the round.
        seed: Random seed for deterministic testing.

    Returns:
        A Rank if bidding, None if passing.
    """
    if not valid_levels:
        raise ValueError("No valid bid levels available")
    rng = random.Random(seed)
    if rng.random() < 0.4:
        return None  # pass
    return min(valid_levels, key=lambda r: list(Rank).index(r))


def choose_stir(
    current_trump: Suit,
    valid_levels: list[Rank],
    player_index: int,
    stir_history: list,
    seed: Optional[int] = None,
) -> Optional[tuple[Suit, Rank]]:
    """Choose whether to stir (change trump suit) or pass.

    Simple strategy: 50% chance to pass. If stirring, pick a random valid
    suit different from current_trump and a random valid level.

    Args:
        current_trump: The current trump suit.
        valid_levels: Non-empty list of valid levels to stir to.
        player_index: Index of the AI player.
        stir_history: History of stir actions this round.
        seed: Random seed for deterministic testing.

    Returns:
        A (Suit, Rank) tuple if stirring, None if passing.

    Raises:
        ValueError: If valid_levels is empty.
    """
    if not valid_levels:
        raise ValueError("No valid stir levels available")
    rng = random.Random(seed)
    if rng.random() < 0.5:
        return None  # pass
    # Pick a suit different from current trump
    non_joker_suits = [s for s in Suit if s != Suit.JOKER and s != current_trump]
    if not non_joker_suits:
        return None
    new_suit = rng.choice(non_joker_suits)
    level = rng.choice(valid_levels)
    return (new_suit, level)


def choose_discard(
    hand: list[Card],
    count: int,
    seed: Optional[int] = None,
) -> list[Card]:
    """Select cards from hand to discard (for bottom cards).

    Args:
        hand: The player's current hand of cards.
        count: Number of cards to discard.
        seed: Random seed for deterministic testing.

    Returns:
        A list of Card objects selected for discard.
    """
    if count < 0:
        raise ValueError(f"Discard count must be non-negative, got {count}")
    if count > len(hand):
        raise ValueError(f"Discard count {count} exceeds hand size {len(hand)}")
    rng = random.Random(seed)
    return rng.sample(hand, count)
