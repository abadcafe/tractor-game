"""Play rules: pattern detection, following rules, and legal play enumeration.

Implements the Shengji/Tractor play rules for singles, pairs, tractors, throws,
and the legal play enumeration for leading and following.
"""

from server.sm.card_model import Card, Suit, Rank, _SUITED_RANKS
from server.sm.comparator import effective_suit, is_trump_card, trump_order
from server.sm.types import PlayAction, PlayType


# ---- Helpers ----


def _non_trump_rank_order(rank: Rank, trump_rank: Rank) -> int:
    """Return rank order for non-trump suits, skipping the trump_rank.

    Spec: "级牌Rank从非主排序中移除"
    This means for non-trump suits, the trump rank is excluded from the
    consecutive rank ordering used for tractor detection.

    For trump suit itself, the full rank ordering applies including trump_rank.
    """
    # Build the non-trump ordering: all suited ranks except trump_rank
    order = 0
    for r in _SUITED_RANKS:
        if r == trump_rank:
            continue
        order += 1
        if r == rank:
            return order
    return 0


def _trump_rank_order(rank: Rank) -> int:
    """Return rank ordering for the trump suit (full ordering including all ranks)."""
    for i, r in enumerate(_SUITED_RANKS):
        if r == rank:
            return i + 1
    # Jokers in trump
    if rank == Rank.SMALL_JOKER:
        return 14
    if rank == Rank.BIG_JOKER:
        return 15
    return 0


def _rank_order_for_suit(
    rank: Rank, suit: Suit, trump_suit: Suit | None, trump_rank: Rank
) -> int:
    """Return the rank ordering for a card's rank within its effective suit group.

    For trump cards, use trump rank ordering.
    For non-trump cards, use non-trump ordering (skipping trump_rank).
    """
    if suit == Suit.JOKER:
        return _trump_rank_order(rank)
    if trump_suit is not None and suit == trump_suit:
        return _trump_rank_order(rank)
    return _non_trump_rank_order(rank, trump_rank)


def _group_cards_by_effective_suit(
    hand: list[Card], trump_suit: Suit | None, trump_rank: Rank
) -> dict[str | Suit, list[Card]]:
    """Partition cards by effective suit."""
    groups: dict[str | Suit, list[Card]] = {}
    for card in hand:
        eff = effective_suit(card, trump_suit, trump_rank)
        groups.setdefault(eff, []).append(card)
    return groups


def _find_consecutive_pairs(
    cards: list[Card], suit: Suit | str, trump_suit: Suit | None, trump_rank: Rank
) -> list[PlayAction]:
    """Find all consecutive pair runs in a group of cards of the same effective suit.

    Returns PlayAction of type TRACTOR for each run of 2+ consecutive pairs.
    """
    result: list[PlayAction] = []
    if not cards:
        return result

    # Build rank -> list of cards mapping
    rank_cards: dict[Rank, list[Card]] = {}
    for c in cards:
        rank_cards.setdefault(c.rank, []).append(c)

    # Find all pairs (exactly 2 cards of same rank)
    pair_ranks = sorted(
        [r for r, cs in rank_cards.items() if len(cs) >= 2],
        key=lambda r: _rank_order_for_suit(r, suit if isinstance(suit, Suit) else Suit.HEARTS, trump_suit, trump_rank),
    )

    if len(pair_ranks) < 1:
        return result

    # Find consecutive runs in pair_ranks
    # For non-trump suits, skip trump_rank in consecutive detection
    def _is_consecutive(r1: Rank, r2: Rank) -> bool:
        """Check if two ranks are consecutive in the appropriate ordering."""
        if suit == "trump":
            # In trump, use full ordering including trump_rank
            o1 = _trump_rank_order(r1)
            o2 = _trump_rank_order(r2)
            return o2 == o1 + 1
        else:
            # In non-trump, use non-trump ordering (skip trump_rank)
            o1 = _non_trump_rank_order(r1, trump_rank)
            o2 = _non_trump_rank_order(r2, trump_rank)
            return o2 == o1 + 1

    # Find runs of consecutive pair ranks
    runs: list[list[Rank]] = []
    current_run: list[Rank] = [pair_ranks[0]]
    for i in range(1, len(pair_ranks)):
        if _is_consecutive(current_run[-1], pair_ranks[i]):
            current_run.append(pair_ranks[i])
        else:
            if len(current_run) >= 2:
                runs.append(current_run)
            current_run = [pair_ranks[i]]
    if len(current_run) >= 2:
        runs.append(current_run)

    # Emit tractors for runs of 2+ consecutive pairs
    for run in runs:
        tractor_cards: list[Card] = []
        for r in run:
            tractor_cards.extend(rank_cards[r][:2])
        result.append(PlayAction(type=PlayType.TRACTOR, cards=tractor_cards))

    return result


# ---- Public API ----


def detect_singles(hand: list[Card]) -> list[PlayAction]:
    """Every card is a valid single."""
    return [PlayAction(type=PlayType.SINGLE, cards=[c]) for c in hand]


def detect_pairs(hand: list[Card]) -> list[PlayAction]:
    """Group by suit+rank; groups of 2+ become pairs (take first 2)."""
    rank_groups: dict[tuple[Suit, Rank], list[Card]] = {}
    for c in hand:
        if c.suit == Suit.JOKER:
            # Jokers pair by rank only
            key = (Suit.JOKER, c.rank)
        else:
            key = (c.suit, c.rank)
        rank_groups.setdefault(key, []).append(c)

    result: list[PlayAction] = []
    for cards in rank_groups.values():
        if len(cards) >= 2:
            result.append(PlayAction(type=PlayType.PAIR, cards=cards[:2]))
    return result


def detect_tractors(
    hand: list[Card], trump_suit: Suit | None, trump_rank: Rank
) -> list[PlayAction]:
    """Find consecutive pair tractors by effective suit.

    Partitions by effective suit, finds consecutive pairs using trump-aware
    ordering, and emits tractors (2+ consecutive pairs).
    """
    groups = _group_cards_by_effective_suit(hand, trump_suit, trump_rank)

    result: list[PlayAction] = []
    for eff_suit, cards in groups.items():
        pair_actions = _find_consecutive_pairs(cards, eff_suit, trump_suit, trump_rank)
        result.extend(pair_actions)
    return result


def detect_throws(
    hand: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
    known_remaining_cards: list[Card] | None = None,
) -> list[PlayAction]:
    """Identify THROW plays: multiple cards of same non-trump suit where each card
    is the highest remaining rank in that suit.

    If known_remaining_cards is None, assume all cards not in hand are potentially
    remaining (conservative: no THROW possible unless we can verify).
    If known_remaining_cards is provided, only throw cards that are higher than
    all remaining cards of that suit.
    THROW does not apply to trump cards.
    """
    # Group hand cards by suit (non-trump only)
    suit_groups: dict[Suit, list[Card]] = {}
    for c in hand:
        eff = effective_suit(c, trump_suit, trump_rank)
        if eff == "trump":
            continue
        if isinstance(eff, Suit):
            suit_groups.setdefault(eff, []).append(c)

    if not suit_groups:
        return []

    # Build remaining cards by suit (if provided)
    remaining_by_suit: dict[Suit, set[Rank]] = {}
    if known_remaining_cards is not None:
        for c in known_remaining_cards:
            eff = effective_suit(c, trump_suit, trump_rank)
            if eff == "trump":
                continue
            if isinstance(eff, Suit):
                remaining_by_suit.setdefault(eff, set()).add(c.rank)

    result: list[PlayAction] = []
    for suit, cards in suit_groups.items():
        if len(cards) < 2:
            continue

        # A card can be thrown if no remaining card of the same suit has a higher rank
        throwable: list[Card] = []
        remaining_ranks = remaining_by_suit.get(suit, set())
        for c in cards:
            # Check if any remaining card has higher rank
            has_higher = any(
                _non_trump_rank_order(r, trump_rank) > _non_trump_rank_order(c.rank, trump_rank)
                for r in remaining_ranks
            )
            if not has_higher:
                throwable.append(c)

        # THROW requires 2+ cards
        if len(throwable) >= 2:
            result.append(PlayAction(type=PlayType.THROW, cards=throwable))

    return result


def get_legal_plays(
    hand: list[Card],
    is_leading: bool,
    lead_action: PlayAction | None,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> list[PlayAction]:
    """Enumerate all legal plays given the hand and game context.

    If leading: all singles, pairs, tractors, and throws from hand.
    If following: delegate to follow rules based on lead_action.type.
    """
    if is_leading:
        result: list[PlayAction] = []
        result.extend(detect_singles(hand))
        result.extend(detect_pairs(hand))
        result.extend(detect_tractors(hand, trump_suit, trump_rank))
        result.extend(detect_throws(hand, trump_suit, trump_rank))
        return result

    # Following rules
    assert lead_action is not None, "lead_action required when following"
    lead_type = lead_action.type
    lead_cards = lead_action.cards

    if lead_type == PlayType.SINGLE:
        return _follow_single(hand, lead_cards, trump_suit, trump_rank)
    elif lead_type == PlayType.PAIR:
        return _follow_pair(hand, lead_cards, trump_suit, trump_rank)
    elif lead_type == PlayType.TRACTOR:
        return _follow_tractor(hand, lead_cards, trump_suit, trump_rank)
    elif lead_type == PlayType.THROW:
        return _follow_throw(hand, lead_cards, trump_suit, trump_rank)
    else:
        # Unknown lead type: play anything
        return detect_singles(hand)


def infer_play_type(
    cards: list[Card],
    trump_suit: Suit | None = None,
    trump_rank: Rank = Rank.TWO,
) -> PlayType:
    """Infer the play type from a set of cards.

    1 card = SINGLE
    2 same-suit same-rank = PAIR
    4+ cards forming consecutive pairs = TRACTOR
    3+ same non-trump suit, non-consecutive, non-pair = THROW
    else = SINGLE
    """
    n = len(cards)
    if n == 0:
        return PlayType.SINGLE
    if n == 1:
        return PlayType.SINGLE

    if n == 2:
        c1, c2 = cards
        # Same suit and same rank
        if c1.suit == c2.suit and c1.rank == c2.rank:
            return PlayType.PAIR
        return PlayType.SINGLE

    if n >= 4 and n % 2 == 0:
        # Check if it's a tractor: all cards same effective suit, forming consecutive pairs
        eff_suits = {effective_suit(c, trump_suit, trump_rank) for c in cards}
        if len(eff_suits) == 1:
            eff_suit_val = next(iter(eff_suits))
            # Check if cards form consecutive pairs
            rank_groups: dict[Rank, list[Card]] = {}
            for c in cards:
                rank_groups.setdefault(c.rank, []).append(c)
            pair_ranks = [r for r, cs in rank_groups.items() if len(cs) == 2]
            if len(pair_ranks) == n // 2:
                # Check consecutive
                pair_ranks_sorted = sorted(
                    pair_ranks,
                    key=lambda r: _rank_order_for_suit(
                        r,
                        eff_suit_val if isinstance(eff_suit_val, Suit) else Suit.HEARTS,
                        trump_suit,
                        trump_rank,
                    ),
                )
                is_consecutive = True
                for i in range(1, len(pair_ranks_sorted)):
                    o_prev = _rank_order_for_suit(
                        pair_ranks_sorted[i - 1],
                        eff_suit_val if isinstance(eff_suit_val, Suit) else Suit.HEARTS,
                        trump_suit,
                        trump_rank,
                    )
                    o_curr = _rank_order_for_suit(
                        pair_ranks_sorted[i],
                        eff_suit_val if isinstance(eff_suit_val, Suit) else Suit.HEARTS,
                        trump_suit,
                        trump_rank,
                    )
                    if o_curr != o_prev + 1:
                        is_consecutive = False
                        break
                if is_consecutive and len(pair_ranks_sorted) >= 2:
                    return PlayType.TRACTOR

    # Check for throw: 3+ cards, same suit, not all pairs, not trump
    if n >= 3:
        eff_suits = {effective_suit(c, trump_suit, trump_rank) for c in cards}
        if len(eff_suits) == 1:
            eff_suit_val = next(iter(eff_suits))
            if eff_suit_val != "trump":
                # Same non-trump suit
                return PlayType.THROW

    return PlayType.SINGLE


# ---- Follow Rules ----


def _follow_single(
    hand: list[Card], lead_cards: list[Card], trump_suit: Suit | None, trump_rank: Rank
) -> list[PlayAction]:
    """Following a single: must play same effective suit if possible, else anything."""
    lead_eff = effective_suit(lead_cards[0], trump_suit, trump_rank)

    # Find cards that match the lead effective suit
    matching = [
        c for c in hand
        if effective_suit(c, trump_suit, trump_rank) == lead_eff
    ]

    if matching:
        return [PlayAction(type=PlayType.SINGLE, cards=[c]) for c in matching]

    # No matching suit: play anything
    return detect_singles(hand)


def _follow_pair(
    hand: list[Card], lead_cards: list[Card], trump_suit: Suit | None, trump_rank: Rank
) -> list[PlayAction]:
    """Following a pair: must play pair of same effective suit if possible, else any 2 cards."""
    lead_eff = effective_suit(lead_cards[0], trump_suit, trump_rank)

    # Find pairs that match the lead effective suit
    suit_cards = [
        c for c in hand
        if effective_suit(c, trump_suit, trump_rank) == lead_eff
    ]

    rank_groups: dict[tuple[Suit, Rank], list[Card]] = {}
    for c in suit_cards:
        key = (c.suit, c.rank)
        rank_groups.setdefault(key, []).append(c)

    matching_pairs: list[PlayAction] = []
    for cards in rank_groups.values():
        if len(cards) >= 2:
            matching_pairs.append(PlayAction(type=PlayType.PAIR, cards=cards[:2]))

    if matching_pairs:
        return matching_pairs

    # No matching pair: play any 2 cards
    result: list[PlayAction] = []
    for i in range(len(hand)):
        for j in range(i + 1, len(hand)):
            result.append(PlayAction(type=PlayType.PAIR, cards=[hand[i], hand[j]]))
    return result


def _follow_tractor(
    hand: list[Card], lead_cards: list[Card], trump_suit: Suit | None, trump_rank: Rank
) -> list[PlayAction]:
    """Following a tractor: match length with pairs+fill, or any equivalent number of cards.

    Lead tractor has N cards (N/2 pairs).
    Rules:
    - Have consecutive pairs of same effective suit of same length: must play them
    - Have pairs but not enough consecutive: play all pairs + fill with singles
    - No pairs: play any N cards
    """
    lead_eff = effective_suit(lead_cards[0], trump_suit, trump_rank)
    tractor_len = len(lead_cards)  # Must play same number of cards

    # Find cards of same effective suit
    suit_cards = [
        c for c in hand
        if effective_suit(c, trump_suit, trump_rank) == lead_eff
    ]

    # Try to find matching tractor (consecutive pairs of same length)
    if suit_cards:
        pair_actions = _find_consecutive_pairs(suit_cards, lead_eff, trump_suit, trump_rank)
        # Find tractors of matching length
        matching_tractors = [
            a for a in pair_actions
            if a.type == PlayType.TRACTOR and len(a.cards) == tractor_len
        ]
        if matching_tractors:
            return matching_tractors

    # Find pairs in matching suit
    rank_groups: dict[Suit | tuple, list[Card]] = {}
    for c in suit_cards:
        key = (c.suit, c.rank)
        rank_groups.setdefault(key, []).append(c)

    pairs_in_suit: list[PlayAction] = []
    singles_in_suit: list[Card] = []
    for key, cards in rank_groups.items():
        if len(cards) >= 2:
            pairs_in_suit.append(PlayAction(type=PlayType.PAIR, cards=cards[:2]))
        else:
            singles_in_suit.extend(cards)

    # If we have pairs, play all pairs + fill
    if pairs_in_suit:
        result: list[PlayAction] = []
        pair_count = len(pairs_in_suit) * 2
        fill_needed = tractor_len - pair_count
        if fill_needed < 0:
            # Too many pairs; just play the first needed pairs
            fill_needed = 0
            # Return first enough pairs
            pairs_needed = tractor_len // 2
            combo_cards: list[Card] = []
            for a in pairs_in_suit[:pairs_needed]:
                combo_cards.extend(a.cards)
            result.append(PlayAction(type=PlayType.TRACTOR, cards=combo_cards))
        else:
            combo_cards: list[Card] = []
            for a in pairs_in_suit:
                combo_cards.extend(a.cards)
            # Fill from same-suit singles first, then from any remaining hand cards
            used_ids = {c.id for c in combo_cards}
            fill_cards = [c for c in singles_in_suit if c.id not in used_ids]
            remaining = [c for c in hand if c.id not in used_ids and c not in fill_cards]
            extra_fill = remaining[: max(0, fill_needed - len(fill_cards))]
            fill_cards.extend(extra_fill)
            combo_cards.extend(fill_cards[:fill_needed])
            result.append(PlayAction(type=PlayType.TRACTOR, cards=combo_cards))
        return result

    # No pairs at all: play any tractor_len cards
    if len(hand) >= tractor_len:
        return [PlayAction(type=PlayType.TRACTOR, cards=hand[:tractor_len])]
    return []


def _follow_throw(
    hand: list[Card], lead_cards: list[Card], trump_suit: Suit | None, trump_rank: Rank
) -> list[PlayAction]:
    """Following a throw: must play same-suit cards if possible, fill otherwise.

    Lead throw is N cards of a specific non-trump suit.
    Rules:
    - Have N or more cards of same suit: play exactly N of them
    - Have fewer than N: play all of them + fill from other cards
    - No cards of suit: play any N cards
    """
    lead_eff = effective_suit(lead_cards[0], trump_suit, trump_rank)
    throw_len = len(lead_cards)

    # Must be non-trump
    if lead_eff == "trump":
        return detect_singles(hand) if throw_len == 1 else []

    # Find cards of the lead suit
    suit_cards = [
        c for c in hand
        if effective_suit(c, trump_suit, trump_rank) == lead_eff
    ]

    if len(suit_cards) >= throw_len:
        # Play exactly throw_len cards of the suit (highest ones)
        sorted_suit = sorted(
            suit_cards,
            key=lambda c: _non_trump_rank_order(c.rank, trump_rank),
            reverse=True,
        )
        return [PlayAction(type=PlayType.THROW, cards=sorted_suit[:throw_len])]

    if suit_cards:
        # Play all suit cards + fill from other cards
        other_cards = [c for c in hand if c not in suit_cards]
        fill = other_cards[: throw_len - len(suit_cards)]
        return [PlayAction(type=PlayType.THROW, cards=suit_cards + fill)]

    # No cards of suit: play any throw_len cards
    if len(hand) >= throw_len:
        return [PlayAction(type=PlayType.THROW, cards=hand[:throw_len])]
    return []
