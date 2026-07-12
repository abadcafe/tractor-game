"""Trump-aware card ordering rules for Shengji/Tractor.

Provides trump ordering, effective suit determination, and play
comparison.
"""

from .cards import Card, Rank, Suit
from .types import EffectiveSuit

# ---- Constants ----

SUIT_OFFSET: dict[Suit, int] = {
    Suit.DIAMONDS: 0,
    Suit.CLUBS: 1,
    Suit.HEARTS: 2,
    Suit.SPADES: 3,
    Suit.JOKER: -1,
}

RANK_ORDER: dict[Rank, int] = {
    Rank.TWO: 2,
    Rank.THREE: 3,
    Rank.FOUR: 4,
    Rank.FIVE: 5,
    Rank.SIX: 6,
    Rank.SEVEN: 7,
    Rank.EIGHT: 8,
    Rank.NINE: 9,
    Rank.TEN: 10,
    Rank.JACK: 11,
    Rank.QUEEN: 12,
    Rank.KING: 13,
    Rank.ACE: 14,
    Rank.SMALL_JOKER: 15,
    Rank.BIG_JOKER: 16,
}

_BID_SUIT_ORDER: dict[Suit, int] = {
    Suit.DIAMONDS: 0,
    Suit.CLUBS: 1,
    Suit.HEARTS: 2,
    Suit.SPADES: 3,
}

_BID_JOKER_ORDER: dict[Rank, int] = {
    Rank.SMALL_JOKER: 4,
    Rank.BIG_JOKER: 5,
}

DISPLAY_SUIT_ORDER: dict[Suit, int] = {
    Suit.SPADES: 0,
    Suit.HEARTS: 1,
    Suit.CLUBS: 2,
    Suit.DIAMONDS: 3,
    Suit.JOKER: -1,
}


# ---- Trump Order ----


def is_trump_card(
    card: Card, trump_suit: Suit | None, trump_rank: Rank
) -> bool:
    """Determine if a card is a trump card.

    Trump cards: jokers, cards of trump_rank, or cards of trump_suit.
    When trump_suit is None, only jokers and trump_rank are trump.
    """
    if card.is_joker:
        return True
    if card.rank == trump_rank:
        return True
    if trump_suit is not None and card.suit == trump_suit:
        return True
    return False


def trump_order(
    card: Card, trump_suit: Suit | None, trump_rank: Rank
) -> int:
    """Return the trump order value of a card.

    Higher value = stronger card. Ordering:
      BJ=100, SJ=90, trump_rank+trump_suit=80, trump_rank+other=70,
      trump_suit_non_rank=45+RANK_ORDER, non_trump=RANK_ORDER-2.

    When trump_suit is None, only jokers and trump_rank are trump; all
    trump-rank suits are equal at 70, so earlier play order breaks ties.
    """
    if card.rank == Rank.BIG_JOKER:
        return 100
    if card.rank == Rank.SMALL_JOKER:
        return 90
    if card.rank == trump_rank:
        if trump_suit is not None and card.suit == trump_suit:
            return 80
        return 70
    if trump_suit is not None and card.suit == trump_suit:
        return 45 + RANK_ORDER[card.rank]
    # Non-trump
    return RANK_ORDER[card.rank] - 2


def trump_rank_order(
    card: Card, trump_suit: Suit | None, trump_rank: Rank
) -> int:
    """Return the trump rank order of a card.

    Identical to trump_order. Named separately to distinguish semantic
    use:
    trump_order is for comparing play strength; trump_rank_order is for
    decompose's trump-group tractor detection (spec section 2.3).

    When trump_suit is None, all trump rank cards get the same order.
    """
    return trump_order(card, trump_suit, trump_rank)


# ---- Effective Suit ----


def effective_suit(
    card: Card, trump_suit: Suit | None, trump_rank: Rank
) -> EffectiveSuit:
    """Return the effective suit of a card.

    Returns "trump" if the card is a trump card, otherwise the card's
    actual suit.
    """
    if is_trump_card(card, trump_suit, trump_rank):
        return "trump"
    return card.suit


# ---- Bid Value ----


def bid_value(cards: list[Card], trump_rank: Rank) -> int:
    """Calculate the bid value for a set of cards.

    Value = count * 100 + card_rank, where card_rank ordering is:
      ♦=0, ♣=1, ♥=2, ♠=3, 小王=4, 大王=5.

    Invalid bids return 0:
      - Single joker (count=1 and joker).
      - Non-trump-rank cards.
    """
    if len(cards) == 0:
        return 0

    # All cards must be trump rank or jokers
    for c in cards:
        if c.rank != trump_rank and not c.is_joker:
            return 0

    count = len(cards)

    # Single joker is invalid
    if count == 1 and cards[0].is_joker:
        return 0

    # Pair of jokers
    if count == 2 and all(c.is_joker for c in cards):
        # Must be same type
        if cards[0].rank != cards[1].rank:
            return 0
        joker_val = _BID_JOKER_ORDER[cards[0].rank]
        return count * 100 + joker_val

    # All cards must be same suit (for trump rank cards)
    if any(c.is_joker for c in cards):
        return 0  # Mixed joker + rank cards invalid

    suits = {c.suit for c in cards}
    if len(suits) != 1:
        return 0

    suit = cards[0].suit
    suit_val = _BID_SUIT_ORDER.get(suit, -1)
    if suit_val < 0:
        return 0

    return count * 100 + suit_val


# ---- Sort ----


def sort_by_trump_order(
    cards: list[Card], trump_suit: Suit | None, trump_rank: Rank
) -> list[Card]:
    """Sort cards in descending trump order."""
    return sorted(
        cards,
        key=lambda c: trump_order(c, trump_suit, trump_rank),
        reverse=True,
    )


def display_hand_order_key(
    card: Card, trump_suit: Suit | None, trump_rank: Rank
) -> tuple[int, int, int]:
    """
    Return the player-facing hand display sort key.

    Lower key values render earlier:
      trump cards first as 大王、小王、主级牌、副级牌、其他主牌;
      then side suits as 黑桃、红桃、梅花、方片, each high to low.
    This mirrors frontend/core/card.ts sortHand().
    """
    if is_trump_card(card, trump_suit, trump_rank):
        return (
            0,
            _display_trump_priority(card, trump_suit, trump_rank),
            _display_trump_tiebreaker(card, trump_suit, trump_rank),
        )
    return (1, DISPLAY_SUIT_ORDER[card.suit], -RANK_ORDER[card.rank])


def _display_trump_priority(
    card: Card, trump_suit: Suit | None, trump_rank: Rank
) -> int:
    if card.rank == Rank.BIG_JOKER:
        return 0
    if card.rank == Rank.SMALL_JOKER:
        return 1
    if card.rank == trump_rank:
        if trump_suit is not None and card.suit == trump_suit:
            return 2
        return 3
    return 4


def _display_trump_tiebreaker(
    card: Card, trump_suit: Suit | None, trump_rank: Rank
) -> int:
    priority = _display_trump_priority(card, trump_suit, trump_rank)
    if priority == 3:
        return DISPLAY_SUIT_ORDER[card.suit]
    if priority == 4:
        return -RANK_ORDER[card.rank]
    return 0


def sort_by_display_order(
    cards: list[Card], trump_suit: Suit | None, trump_rank: Rank
) -> list[Card]:
    """
    Sort cards in the same order used by the frontend hand display.
    """
    return sorted(
        cards,
        key=lambda card: display_hand_order_key(
            card, trump_suit, trump_rank
        ),
    )
