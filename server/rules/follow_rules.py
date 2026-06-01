"""Following rules for 升级 (Shengji/Tractor) trick-taking.

When a player leads, they can play anything.
When following:
  1. Must match the lead pattern type (single->single, pair->pair, etc.)
  2. Must follow suit if possible (for the lead suit)
  3. If can't follow suit with the required pattern, can play anything

Ported from src/rules/follow-rules.ts.
"""

from __future__ import annotations

from server.engine.card import Card, Suit, Rank
from server.engine.types import PlayType, PlayAction
from server.rules.comparator import effective_suit, trump_order


# ---- Public API ----


def get_lead_suit(
    lead_action: PlayAction, trump_suit: Suit, trump_rank: Rank,
) -> Suit | str:
    """Determine the lead suit of a trick.

    For trump leads, the lead suit is 'trump'.
    For non-trump leads, it is the effective suit of the first card.
    """
    if not lead_action.cards:
        return Suit.JOKER
    return effective_suit(lead_action.cards[0], trump_suit, trump_rank)


def can_follow(
    hand: list[Card],
    lead_action: PlayAction,
    trump_suit: Suit,
    trump_rank: Rank,
) -> bool:
    """Check if a player CAN follow the lead pattern with their hand.

    Returns True if the player has cards that can follow the lead.
    """
    match lead_action.type:
        case PlayType.SINGLE:
            return _can_follow_single(hand, lead_action.cards[0], trump_suit, trump_rank)
        case PlayType.PAIR:
            return _can_follow_pair(hand, lead_action.cards[0], trump_suit, trump_rank)
        case PlayType.TRACTOR:
            return _can_follow_tractor(hand, lead_action, trump_suit, trump_rank)
        case PlayType.THROW:
            return _can_follow_throw(hand, lead_action, trump_suit, trump_rank)
    return False


def get_legal_follows(
    hand: list[Card],
    lead_action: PlayAction,
    trump_suit: Suit,
    trump_rank: Rank,
) -> list[PlayAction]:
    """Get all legal follow plays for a given lead.

    Returns a list of all valid card combinations the player can play.
    """
    match lead_action.type:
        case PlayType.SINGLE:
            return _get_follow_singles(hand, lead_action.cards[0], trump_suit, trump_rank)
        case PlayType.PAIR:
            return _get_follow_pairs(hand, lead_action.cards[0], trump_suit, trump_rank)
        case PlayType.TRACTOR:
            return _get_follow_tractors(hand, lead_action, trump_suit, trump_rank)
        case PlayType.THROW:
            return _get_follow_throws(hand, lead_action, trump_suit, trump_rank)
    return []


# ---- Single following ----


def _can_follow_single(
    hand: list[Card], lead_card: Card, trump_suit: Suit, trump_rank: Rank,
) -> bool:
    eff_suit = effective_suit(lead_card, trump_suit, trump_rank)
    if eff_suit == "trump":
        return any(effective_suit(c, trump_suit, trump_rank) == "trump" for c in hand)
    return any(
        effective_suit(c, trump_suit, trump_rank) == eff_suit and c.rank != trump_rank
        for c in hand
    )


def _get_follow_singles(
    hand: list[Card], lead_card: Card, trump_suit: Suit, trump_rank: Rank,
) -> list[PlayAction]:
    eff_suit = effective_suit(lead_card, trump_suit, trump_rank)
    if _can_follow_single(hand, lead_card, trump_suit, trump_rank):
        matching = [
            c for c in hand
            if effective_suit(c, trump_suit, trump_rank) == (
                "trump" if eff_suit == "trump" else eff_suit
            ) and (eff_suit == "trump" or c.rank != trump_rank)
        ]
        return [PlayAction(type=PlayType.SINGLE, cards=[c]) for c in matching]
    return [PlayAction(type=PlayType.SINGLE, cards=[c]) for c in hand]


# ---- Pair following ----


def _can_follow_pair(
    hand: list[Card], lead_card: Card, trump_suit: Suit, trump_rank: Rank,
) -> bool:
    eff_suit = effective_suit(lead_card, trump_suit, trump_rank)
    return _has_pair_in_effective_suit(hand, eff_suit, trump_suit, trump_rank)


def _get_follow_pairs(
    hand: list[Card], lead_card: Card, trump_suit: Suit, trump_rank: Rank,
) -> list[PlayAction]:
    eff_suit = effective_suit(lead_card, trump_suit, trump_rank)
    pairs = _find_pairs_in_effective_suit(hand, eff_suit, trump_suit, trump_rank)
    if pairs:
        return [PlayAction(type=PlayType.PAIR, cards=pair[:2]) for pair in pairs]
    # Can't follow -- play any two cards
    return _generate_any_two(hand)


# ---- Tractor following ----


def _can_follow_tractor(
    hand: list[Card], lead_action: PlayAction, trump_suit: Suit, trump_rank: Rank,
) -> bool:
    eff_suit = effective_suit(lead_action.cards[0], trump_suit, trump_rank)
    pair_count = len(lead_action.cards) // 2
    return _has_tractor_in_effective_suit(hand, eff_suit, pair_count, trump_suit, trump_rank)


def _get_follow_tractors(
    hand: list[Card], lead_action: PlayAction, trump_suit: Suit, trump_rank: Rank,
) -> list[PlayAction]:
    eff_suit = effective_suit(lead_action.cards[0], trump_suit, trump_rank)
    pair_count = len(lead_action.cards) // 2
    tractors = _find_tractors_in_effective_suit(hand, eff_suit, pair_count, trump_suit, trump_rank)
    if tractors:
        return [
            PlayAction(type=PlayType.TRACTOR, cards=cards[: pair_count * 2])
            for cards in tractors
        ]
    # Can't follow tractor -- try pairs, then fill
    pairs = _find_pairs_in_effective_suit(hand, eff_suit, trump_suit, trump_rank)
    follow_cards: list[Card] = []
    for pair in pairs:
        follow_cards.extend(pair[:2])
    needed = pair_count * 2
    if len(follow_cards) >= needed:
        return [PlayAction(type=PlayType.TRACTOR, cards=follow_cards[:needed])]
    remaining = [c for c in hand if c not in follow_cards]
    all_cards = follow_cards + remaining
    if len(all_cards) >= needed:
        return [PlayAction(type=PlayType.TRACTOR, cards=all_cards[:needed])]
    return []


# ---- Throw following ----


def _can_follow_throw(
    hand: list[Card], lead_action: PlayAction, trump_suit: Suit, trump_rank: Rank,
) -> bool:
    suit = lead_action.cards[0].suit
    if suit == Suit.JOKER:
        return False
    matching = [c for c in hand if c.suit == suit and c.rank != trump_rank]
    return len(matching) >= len(lead_action.cards)


def _get_follow_throws(
    hand: list[Card], lead_action: PlayAction, trump_suit: Suit, trump_rank: Rank,
) -> list[PlayAction]:
    suit = lead_action.cards[0].suit
    if suit == Suit.JOKER:
        return []
    matching = [c for c in hand if c.suit == suit and c.rank != trump_rank]
    required = len(lead_action.cards)
    if len(matching) >= required:
        return [PlayAction(type=PlayType.THROW, cards=matching[:required])]
    # Not enough matching -- play all matching + fill with others
    discards = [c for c in hand if c not in matching][: required - len(matching)]
    return [PlayAction(type=PlayType.THROW, cards=matching + discards)]


# ---- Helper functions ----


def _has_pair_in_effective_suit(
    hand: list[Card], eff_suit: Suit | str, trump_suit: Suit, trump_rank: Rank,
) -> bool:
    group = _group_by_effective_order(hand, eff_suit, trump_suit, trump_rank)
    return any(len(cards) >= 2 for cards in group.values())


def _find_pairs_in_effective_suit(
    hand: list[Card], eff_suit: Suit | str, trump_suit: Suit, trump_rank: Rank,
) -> list[list[Card]]:
    group = _group_by_effective_order(hand, eff_suit, trump_suit, trump_rank)
    return [cards[:2] for cards in group.values() if len(cards) >= 2]


def _has_tractor_in_effective_suit(
    hand: list[Card], eff_suit: Suit | str, pair_count: int,
    trump_suit: Suit, trump_rank: Rank,
) -> bool:
    pairs = _find_consecutive_pairs(hand, eff_suit, trump_suit, trump_rank)
    return _find_consecutive_run(pairs, pair_count, eff_suit) is not None


def _find_tractors_in_effective_suit(
    hand: list[Card], eff_suit: Suit | str, pair_count: int,
    trump_suit: Suit, trump_rank: Rank,
) -> list[list[Card]]:
    pairs = _find_consecutive_pairs(hand, eff_suit, trump_suit, trump_rank)
    if len(pairs) < pair_count:
        return []
    results: list[list[Card]] = []
    for i in range(len(pairs) - pair_count + 1):
        is_consecutive = True
        for j in range(i + 1, i + pair_count):
            step = _get_trump_step(pairs[j - 1][0], eff_suit)
            if pairs[j][0] != pairs[j - 1][0] - step:
                is_consecutive = False
                break
        if is_consecutive:
            cards: list[Card] = []
            for j in range(i, i + pair_count):
                cards.extend(pairs[j][1][:2])
            results.append(cards)
    return results


def _find_consecutive_pairs(
    hand: list[Card], eff_suit: Suit | str, trump_suit: Suit, trump_rank: Rank,
) -> list[tuple[int, list[Card]]]:
    """Find pairs in effective suit, sorted by order descending.

    Returns list of (order, cards) tuples where cards has length >= 2.
    """
    group = _group_by_effective_order(hand, eff_suit, trump_suit, trump_rank)
    pairs = [(order, cards) for order, cards in group.items() if len(cards) >= 2]
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs


def _find_consecutive_run(
    pairs: list[tuple[int, list[Card]]], length: int,
    eff_suit: Suit | str = Suit.HEARTS,
) -> list[tuple[int, list[Card]]] | None:
    if len(pairs) < length:
        return None
    for i in range(len(pairs) - length + 1):
        ok = True
        for j in range(i + 1, i + length):
            step = _get_trump_step(pairs[j - 1][0], eff_suit)
            if pairs[j][0] != pairs[j - 1][0] - step:
                ok = False
                break
        if ok:
            return pairs[i : i + length]
    return None


def _group_by_effective_order(
    hand: list[Card], eff_suit: Suit | str, trump_suit: Suit, trump_rank: Rank,
) -> dict[int, list[Card]]:
    """Group cards by their effective order in the given suit context."""
    result: dict[int, list[Card]] = {}
    for c in hand:
        if effective_suit(c, trump_suit, trump_rank) != eff_suit:
            continue
        order = trump_order(c, trump_suit, trump_rank)
        result.setdefault(order, []).append(c)
    return result


def _get_trump_step(order: int, eff_suit: Suit | str) -> int:
    """Get the step size to the next lower trump order level."""
    if eff_suit == "trump":
        if order >= 70:
            return 10  # BJ(100) -> SJ(90) -> trump-rank(80,70-73) all step=10
        return 1
    return 1


def _generate_any_two(hand: list[Card]) -> list[PlayAction]:
    """Generate a fallback play of any two cards (for when you can't follow a pair)."""
    if len(hand) < 2:
        return []
    return [PlayAction(type=PlayType.PAIR, cards=[hand[-2], hand[-1]])]
