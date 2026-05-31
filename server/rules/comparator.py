"""Trump-aware card comparison for 升级 (Shengji/Tractor).

Trump ordering (highest to lowest):
  1. Big Joker (大王)
  2. Small Joker (小王)
  3. Trump rank + Trump suit  (主牌)
  4. Trump rank + Other suits (副级牌), tie-broken by suit convention
  5. Trump suit other ranks: A > K > Q > J > 10 > ... (excluding trump rank)
  6. Non-trump suit other ranks: A > K > ... (excluding trump rank)

Non-trump suit ordering:
  A > K > Q > J > 10 > 9 > ... > 2, EXCLUDING the trump rank
  (trump rank cards of non-trump suits are always trump)
"""

from server.engine.card import Card, Suit, Rank
from server.engine.card_utils import RANK_ORDER


# ---- Suit offset for sub-trump-rank cards ----

_SUIT_OFFSET: dict[Suit, int] = {
    Suit.HEARTS: 3, Suit.SPADES: 2,
    Suit.DIAMONDS: 1, Suit.CLUBS: 0,
    Suit.JOKER: -1,
}


# ---- Public API ----


def trump_order(card: Card, trump_suit: Suit, trump_rank: Rank) -> int:
    """Assign an effective ordering number to a card in the trump context.

    Higher number = stronger card.

    Range allocation (non-overlapping bands):
      100:    Big Joker
      90:     Small Joker
      80:     Trump rank + trump suit (主牌)
      60-63:  Trump rank + other suits (副级牌), suit-offset 0-3
      47-59:  Trump suit cards (non-trump-rank), 47 + (rank-2)
      0-12:   Non-trump suit cards, rank-2
    """
    # 1. Big Joker
    if card.is_joker and card.is_big_joker:
        return 100

    # 2. Small Joker
    if card.is_joker and not card.is_big_joker:
        return 90

    # 3. Trump rank + Trump suit (主牌)
    if card.rank == trump_rank and card.suit == trump_suit:
        return 80

    # 4. Trump rank + Other suits (副级牌)
    if card.rank == trump_rank and card.suit != trump_suit:
        return 70 + _SUIT_OFFSET[card.suit]

    # 5. Trump suit cards (non-trump-rank)
    if card.suit == trump_suit and card.rank != trump_rank:
        return 45 + RANK_ORDER[card.rank]

    # 6. Non-trump suit cards
    return RANK_ORDER[card.rank] - 2


def non_trump_order(card: Card, trump_rank: Rank) -> int:
    """Non-trump suit ordering: natural ranks, but trump rank is excluded.

    Trump rank cards are removed from non-trump suit ordering (return -1).
    """
    if card.rank == trump_rank:
        return -1
    return RANK_ORDER[card.rank]


def compare_cards(a: Card, b: Card, trump_suit: Suit, trump_rank: Rank) -> int:
    """Compare two cards in trump context.

    Returns: negative if a < b, 0 if equal, positive if a > b.
    """
    return trump_order(a, trump_suit, trump_rank) - trump_order(b, trump_suit, trump_rank)


def is_equal_in_trump(a: Card, b: Card, trump_suit: Suit, trump_rank: Rank) -> bool:
    """Check if two cards are effectively equal in trump context (used for pair matching)."""
    return trump_order(a, trump_suit, trump_rank) == trump_order(b, trump_suit, trump_rank)


def effective_suit(card: Card, trump_suit: Suit, trump_rank: Rank) -> Suit | str:
    """Get the effective suit of a card in trump context.

    All trump cards belong to a virtual "trump" group.
    Non-trump cards keep their suit.
    """
    if card.is_joker:
        return "trump"
    if card.rank == trump_rank:
        return "trump"
    if card.suit == trump_suit:
        return "trump"
    return card.suit


def compare_plays(
    a: list[Card],
    b: list[Card],
    trump_suit: Suit,
    trump_rank: Rank,
    lead_suit: Suit | None,
) -> int:
    """Compare two play actions to determine which wins the trick.

    a and b must be of the same PlayType.
    Returns: positive if a wins, negative if b wins, 0 if tie.
    """
    if not a or not b:
        return 0
    if len(a) != len(b):
        return len(a) - len(b)

    a_is_trump = _is_trump_play(a, trump_suit, trump_rank)
    b_is_trump = _is_trump_play(b, trump_suit, trump_rank)

    # Trump beats non-trump
    if a_is_trump and not b_is_trump:
        return 1
    if not a_is_trump and b_is_trump:
        return -1

    # Both trump: compare by highest card in each play
    if a_is_trump and b_is_trump:
        best_a = max(trump_order(c, trump_suit, trump_rank) for c in a)
        best_b = max(trump_order(c, trump_suit, trump_rank) for c in b)
        return best_a - best_b

    # Both non-trump, same suit: compare by highest card
    if a[0].suit == b[0].suit:
        best_a = max(RANK_ORDER[c.rank] for c in a)
        best_b = max(RANK_ORDER[c.rank] for c in b)
        return best_a - best_b

    # Both non-trump, different suits: the one matching lead suit wins
    if lead_suit is not None and a[0].suit == lead_suit:
        return 1
    if lead_suit is not None and b[0].suit == lead_suit:
        return -1

    return 0  # both are off-suit discards


def sort_by_trump_order(cards: list[Card], trump_suit: Suit, trump_rank: Rank) -> list[Card]:
    """Sort cards in descending trump order (strongest first)."""
    return sorted(cards, key=lambda c: trump_order(c, trump_suit, trump_rank), reverse=True)


# ---- Private helpers ----


def _is_trump_play(cards: list[Card], trump_suit: Suit, trump_rank: Rank) -> bool:
    """Check if a set of cards is a trump play."""
    return any(
        c.is_joker or c.rank == trump_rank or c.suit == trump_suit
        for c in cards
    )
