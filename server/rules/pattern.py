"""Pattern detection for 升级 (Shengji/Tractor).

Detects singles, pairs, tractors, and throws from a player's hand.
Pure functions -- no side effects, no game state dependency beyond
trump_suit and trump_rank.

Ported from src/rules/pattern.ts.
"""

from __future__ import annotations

from server.engine.card import Card, Suit, Rank, card_display
from server.engine.card_utils import RANK_ORDER
from server.engine.types import PlayType, PlayAction
from server.rules.comparator import (
    effective_suit,
    trump_order,
    sort_by_trump_order,
)


# ---- Public API ----


def detect_singles(hand: list[Card]) -> list[PlayAction]:
    """Detect all singles in a hand."""
    return [PlayAction(type=PlayType.SINGLE, cards=[c]) for c in hand]


def detect_pairs(hand: list[Card]) -> list[PlayAction]:
    """Detect all pairs in a hand.

    A pair = two cards with same suit + same rank (different deck/id).
    """
    groups: dict[str, list[Card]] = {}
    for c in hand:
        key = f"{c.suit}-{c.rank}"
        groups.setdefault(key, []).append(c)

    pairs: list[PlayAction] = []
    for cards in groups.values():
        if len(cards) >= 2:
            pairs.append(PlayAction(type=PlayType.PAIR, cards=cards[:2]))
    return pairs


def detect_tractors(
    hand: list[Card],
    trump_suit: Suit,
    trump_rank: Rank,
) -> list[PlayAction]:
    """Detect all tractor combinations in a hand.

    A tractor = 2+ consecutive pairs of the SAME effective suit.
    """
    tractors: list[PlayAction] = []

    # Partition hand by effective suit
    trump_cards: list[Card] = []
    non_trump_by_suit: dict[Suit | str, list[Card]] = {}

    for c in hand:
        eff = effective_suit(c, trump_suit, trump_rank)
        if eff == "trump":
            trump_cards.append(c)
        else:
            non_trump_by_suit.setdefault(eff, []).append(c)

    # Find tractors in trump group
    tractors.extend(
        _find_tractors_in_group(trump_cards, trump_suit, trump_rank, is_trump=True)
    )

    # Find tractors in each non-trump suit group
    for cards in non_trump_by_suit.values():
        tractors.extend(
            _find_tractors_in_group(cards, trump_suit, trump_rank, is_trump=False)
        )

    return _deduplicate_by_card_ids(tractors)


def detect_throw_candidates(
    hand: list[Card],
    suit: Suit,
    trump_suit: Suit,
    trump_rank: Rank,
) -> list[PlayAction]:
    """Detect valid throw (甩牌) candidates from a hand for a given suit.

    Only non-trump suits with 2+ cards can be thrown.
    """
    if suit == Suit.JOKER or suit == trump_suit:
        return []

    suit_cards = [c for c in hand if c.suit == suit and c.rank != trump_rank]
    if len(suit_cards) < 2:
        return []

    # Sort by rank descending
    sorted_cards = sorted(suit_cards, key=lambda c: RANK_ORDER[c.rank], reverse=True)

    candidates: list[PlayAction] = []
    for size in range(2, len(sorted_cards) + 1):
        candidates.append(PlayAction(type=PlayType.THROW, cards=sorted_cards[:size]))

    return candidates


def describe_play(action: PlayAction) -> str:
    """Get a human-readable description of a play action."""
    if not action.cards:
        return ""

    card_strs = [card_display(c) for c in action.cards]

    if action.type == PlayType.SINGLE:
        return f"单张 {card_strs[0]}"
    if action.type == PlayType.PAIR:
        return f"对子 {card_strs[0]}{card_strs[1]}"
    if action.type == PlayType.TRACTOR:
        pair_count = len(action.cards) // 2
        return f"拖拉机 {' '.join(card_strs)} ({pair_count}对)"
    if action.type == PlayType.THROW:
        return f"甩牌 {' '.join(card_strs)}"
    return " ".join(card_strs)


# ---- Private helpers ----


def _find_tractors_in_group(
    cards: list[Card],
    trump_suit: Suit,
    trump_rank: Rank,
    is_trump: bool,
) -> list[PlayAction]:
    """Find all tractor combinations within a group of same-effective-suit cards."""
    if len(cards) < 4:
        return []  # Minimum tractor = 2 pairs = 4 cards

    # Sort cards by appropriate ordering
    if is_trump:
        sorted_cards = sort_by_trump_order(cards, trump_suit, trump_rank)
    else:
        sorted_cards = sorted(cards, key=lambda c: RANK_ORDER[c.rank], reverse=True)

    # Find pairs at each order level
    pair_levels: dict[int, list[Card]] = {}
    for c in sorted_cards:
        order = trump_order(c, trump_suit, trump_rank) if is_trump else RANK_ORDER[c.rank]
        existing = pair_levels.get(order)
        if existing is not None and len(existing) < 2:
            existing.append(c)
        elif existing is None:
            pair_levels[order] = [c]

    # Filter to only levels with pairs (2 cards)
    pair_entries = [
        {"order": order, "cards": pair_cards[:2]}
        for order, pair_cards in pair_levels.items()
        if len(pair_cards) >= 2
    ]

    # Sort by order descending
    pair_entries.sort(key=lambda e: e["order"], reverse=True)

    # Find consecutive runs of pairs
    tractors: list[PlayAction] = []
    i = 0
    while i < len(pair_entries):
        j = i + 1
        while j < len(pair_entries) and _is_consecutive(
            pair_entries[j - 1]["order"], pair_entries[j]["order"], is_trump
        ):
            j += 1

        pair_count = j - i
        if pair_count >= 2:
            # Emit sub-tractors
            for start in range(i, j - 1):
                for end in range(start + 2, j + 1):
                    tractor_cards: list[Card] = []
                    for k in range(start, end):
                        tractor_cards.extend(pair_entries[k]["cards"])
                    tractors.append(
                        PlayAction(type=PlayType.TRACTOR, cards=tractor_cards)
                    )

        i = j  # Skip to next non-consecutive group

    return tractors


def _is_consecutive(order_a: int, order_b: int, is_trump: bool) -> bool:
    """Check if two order values are consecutive.

    Non-trump: natural ranks, consecutive if diff is 1.
    Trump: check adjacency in trump order groups.
    """
    if not is_trump:
        return order_a - order_b == 1

    # Trump: check if they're adjacent in trump ordering
    step = _get_trump_order_step(order_a)
    return order_a - order_b == step


def _get_trump_order_step(current_order: int) -> int:
    """Get the step size to the next lower trump order level.

    Trump order groups:
      100:    Big Joker
       90:    Small Joker
       80:    Trump rank + trump suit (主牌)
      70-73:  Trump rank + other suits (副级牌), 70 + suit-offset 0-3
      47-59:  Trump suit cards (non-trump-rank), 45 + RANK_ORDER
       0-12:  Non-trump suit cards, RANK_ORDER - 2
    """
    if current_order == 100:
        return 10  # BJ → SJ
    if current_order == 90:
        return 10  # SJ → 主牌
    if current_order == 80:
        return 7    # 主牌(80) → highest 副级牌(73, Hearts), diff=7
    if 70 <= current_order < 80:
        if current_order == 70:
            return 11  # Last 副级牌(70) → top trump suit (45+14=59), diff=11
        return 1   # Between 副级牌 suits
    if 45 <= current_order < 60:
        return 1   # Within trump suit ranks
    return 0       # Non-trump (not used in is_consecutive for trump)


def _deduplicate_by_card_ids(actions: list[PlayAction]) -> list[PlayAction]:
    """Deduplicate play actions by sorted card ID concatenation."""
    seen: set[str] = set()
    result: list[PlayAction] = []
    for action in actions:
        key = ",".join(sorted(c.id for c in action.cards))
        if key not in seen:
            seen.add(key)
            result.append(action)
    return result
