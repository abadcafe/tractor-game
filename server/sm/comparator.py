"""Trump-aware card comparison module for Shengji/Tractor.

Provides trump ordering, effective suit determination, and play comparison.
"""

from server.sm.card_model import Card, Suit, Rank

# ---- Constants ----

SUIT_OFFSET: dict[Suit, int] = {
    Suit.HEARTS: 3,
    Suit.SPADES: 2,
    Suit.DIAMONDS: 1,
    Suit.CLUBS: 0,
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


# ---- Trump Order ----


def is_trump_card(card: Card, trump_suit: Suit | None, trump_rank: Rank) -> bool:
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


def trump_order(card: Card, trump_suit: Suit | None, trump_rank: Rank) -> int:
    """Return the trump order value of a card.

    Higher value = stronger card. Ordering:
      BJ=100, SJ=90, trump_rank+trump_suit=80, trump_rank+other=70+offset,
      trump_suit_non_rank=45+RANK_ORDER, non_trump=RANK_ORDER-2.

    When trump_suit is None, only jokers and trump_rank are trump (get high orders).
    """
    if card.rank == Rank.BIG_JOKER:
        return 100
    if card.rank == Rank.SMALL_JOKER:
        return 90
    if card.rank == trump_rank:
        if trump_suit is not None and card.suit == trump_suit:
            return 80
        # trump rank in other suit, or trump_suit is None
        return 70 + SUIT_OFFSET.get(card.suit, 0)
    if trump_suit is not None and card.suit == trump_suit:
        return 45 + RANK_ORDER[card.rank]
    # Non-trump
    return RANK_ORDER[card.rank] - 2


# ---- Effective Suit ----


def effective_suit(
    card: Card, trump_suit: Suit | None, trump_rank: Rank
) -> Suit | str:
    """Return the effective suit of a card.

    Returns "trump" if the card is a trump card, otherwise the card's actual suit.
    """
    if is_trump_card(card, trump_suit, trump_rank):
        return "trump"
    return card.suit


# ---- Compare Plays ----


def compare_plays(
    a_cards: list[Card],
    b_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
    lead_suit: Suit | None,
) -> int:
    """Compare two plays (each a list of cards).

    Returns > 0 if a wins, < 0 if b wins, 0 if tie.

    Rules:
      - Trump beats non-trump.
      - Both trump: compare by max trump_order.
      - Both non-trump, same suit: compare by max RANK_ORDER.
      - Both non-trump, different suit: lead suit wins.
    """
    a_has_trump = any(is_trump_card(c, trump_suit, trump_rank) for c in a_cards)
    b_has_trump = any(is_trump_card(c, trump_suit, trump_rank) for c in b_cards)

    if a_has_trump and not b_has_trump:
        return 1
    if b_has_trump and not a_has_trump:
        return -1

    if a_has_trump and b_has_trump:
        a_max = max(trump_order(c, trump_suit, trump_rank) for c in a_cards)
        b_max = max(trump_order(c, trump_suit, trump_rank) for c in b_cards)
        return a_max - b_max

    # Both non-trump: determine effective suits
    a_eff = effective_suit(a_cards[0], trump_suit, trump_rank)
    b_eff = effective_suit(b_cards[0], trump_suit, trump_rank)

    if a_eff == b_eff:
        # Same suit: compare by max RANK_ORDER
        a_max = max(RANK_ORDER[c.rank] for c in a_cards)
        b_max = max(RANK_ORDER[c.rank] for c in b_cards)
        return a_max - b_max

    # Different suits: lead suit wins
    if lead_suit is not None and a_eff == lead_suit:
        return 1
    if lead_suit is not None and b_eff == lead_suit:
        return -1

    # Neither is the lead suit (shouldn't happen in normal play)
    return 0


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
    return sorted(cards, key=lambda c: trump_order(c, trump_suit, trump_rank), reverse=True)
