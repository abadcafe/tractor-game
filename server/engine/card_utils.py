"""Card utility functions and constants for 升级 (Shengji/Tractor) card game.

Provides natural rank ordering, pair detection, suit grouping, hand sorting,
and point mapping. Split from the card module to keep the data model (card.py)
separate from utility logic.
"""

from server.engine.card import Card, Suit, Rank


# ---- Constants ----

RANK_ORDER: dict[Rank, int] = {
    Rank.TWO: 2, Rank.THREE: 3, Rank.FOUR: 4, Rank.FIVE: 5,
    Rank.SIX: 6, Rank.SEVEN: 7, Rank.EIGHT: 8, Rank.NINE: 9,
    Rank.TEN: 10, Rank.JACK: 11, Rank.QUEEN: 12, Rank.KING: 13,
    Rank.ACE: 14,
    Rank.SMALL_JOKER: 15,
    Rank.BIG_JOKER: 16,
}

SUITED_RANKS: list[Rank] = [
    Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE,
    Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE,
    Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE,
]

SUITS: list[Suit] = [Suit.HEARTS, Suit.SPADES, Suit.DIAMONDS, Suit.CLUBS]

POINTS_MAP: dict[Rank, int] = {
    Rank.TWO: 0, Rank.THREE: 0, Rank.FOUR: 0, Rank.FIVE: 5,
    Rank.SIX: 0, Rank.SEVEN: 0, Rank.EIGHT: 0, Rank.NINE: 0,
    Rank.TEN: 10, Rank.JACK: 0, Rank.QUEEN: 0, Rank.KING: 10,
    Rank.ACE: 0, Rank.SMALL_JOKER: 0, Rank.BIG_JOKER: 0,
}

TOTAL_POINTS: int = 200


# ---- Helper Functions ----

_SUIT_SORT_ORDER: dict[Suit, int] = {
    Suit.JOKER: 0, Suit.HEARTS: 1, Suit.SPADES: 2,
    Suit.DIAMONDS: 3, Suit.CLUBS: 4,
}


def natural_rank(rank: Rank) -> int:
    """Return the natural numeric rank value (2-16)."""
    return RANK_ORDER[rank]


def is_pair(a: Card, b: Card) -> bool:
    """Check if two cards are a pair (same suit, same rank, different id)."""
    if a.suit != b.suit:
        return False
    if a.id == b.id:
        return False
    return a.rank == b.rank


def group_by_suit(cards: list[Card]) -> dict[Suit, list[Card]]:
    """Group cards by suit."""
    groups: dict[Suit, list[Card]] = {}
    for c in cards:
        groups.setdefault(c.suit, []).append(c)
    return groups


def sort_hand(cards: list[Card]) -> list[Card]:
    """Sort cards for display: grouped by suit, then by rank descending.

    Suit order: Joker > Hearts > Spades > Diamonds > Clubs
    """
    return sorted(cards, key=lambda c: (_SUIT_SORT_ORDER[c.suit], -natural_rank(c.rank)))
