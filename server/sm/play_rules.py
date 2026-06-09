"""Play rules: pattern detection, following rules, and legal play enumeration.

Implements the Shengji/Tractor play rules for singles, pairs, tractors, throws,
and the legal play enumeration for leading and following.
"""

from itertools import combinations

from server.sm.card_model import Card, Suit, Rank, SUITED_RANKS
from server.sm.comparator import SUIT_OFFSET, effective_suit, trump_rank_order
from server.sm.types import PlayAction, PlayType, SubPlay


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
    for r in SUITED_RANKS:
        if r == trump_rank:
            continue
        order += 1
        if r == rank:
            return order
    return 0


def _trump_rank_order(rank: Rank) -> int:
    """Return rank ordering for the trump suit (full ordering including all ranks)."""
    for i, r in enumerate(SUITED_RANKS):
        if r == rank:
            return i + 1
    # Jokers in trump
    if rank == Rank.SMALL_JOKER:
        return 14
    if rank == Rank.BIG_JOKER:
        return 15
    return 0


def _rank_order_for_suit(
    rank: Rank, suit: Suit | str, trump_suit: Suit | None, trump_rank: Rank
) -> int:
    """Return the rank ordering for a card's rank within its effective suit group.

    For trump cards (effective suit "trump" or actual trump suit), use trump rank ordering.
    For non-trump cards, use non-trump ordering (skipping trump_rank).
    """
    if suit == "trump":
        return _trump_rank_order(rank)
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
        key=lambda r: _rank_order_for_suit(r, suit, trump_suit, trump_rank),
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


def detect_pairs(
    hand: list[Card],
    trump_suit: Suit | None = None,
    trump_rank: Rank = Rank.TWO,
) -> list[PlayAction]:
    """Group by effective suit+rank; groups of 2+ become pairs (pairs of 2).

    For trump cards, effective suit is 'trump' regardless of actual suit,
    so cross-suit same-rank trump cards form pairs per spec line 696.
    """
    rank_groups: dict[tuple[Suit | str, Rank], list[Card]] = {}
    for c in hand:
        eff = effective_suit(c, trump_suit, trump_rank)
        key = (eff, c.rank)
        rank_groups.setdefault(key, []).append(c)

    result: list[PlayAction] = []
    for cards in rank_groups.values():
        for i in range(0, len(cards) - 1, 2):
            result.append(PlayAction(type=PlayType.PAIR, cards=[cards[i], cards[i + 1]]))
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
    # Conservative: if we don't know what's remaining, we can't verify highest
    if known_remaining_cards is None:
        return []

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

    # Build remaining cards by suit
    remaining_by_suit: dict[Suit, set[Rank]] = {}
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
        result.extend(detect_pairs(hand, trump_suit, trump_rank))
        result.extend(detect_tractors(hand, trump_suit, trump_rank))
        result.extend(detect_throws(hand, trump_suit, trump_rank))
        return result

    # Following rules
    assert lead_action is not None, "lead_action required when following"
    if not lead_action.cards:
        # Lead player hasn't played yet; following players must wait.
        return []
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
        # Same effective suit and same rank
        eff1 = effective_suit(c1, trump_suit, trump_rank)
        eff2 = effective_suit(c2, trump_suit, trump_rank)
        if eff1 == eff2 and c1.rank == c2.rank:
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
                        eff_suit_val,
                        trump_suit,
                        trump_rank,
                    ),
                )
                is_consecutive = True
                for i in range(1, len(pair_ranks_sorted)):
                    o_prev = _rank_order_for_suit(
                        pair_ranks_sorted[i - 1],
                        eff_suit_val,
                        trump_suit,
                        trump_rank,
                    )
                    o_curr = _rank_order_for_suit(
                        pair_ranks_sorted[i],
                        eff_suit_val,
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

    rank_groups: dict[tuple[Suit | str, Rank], list[Card]] = {}
    for c in suit_cards:
        eff = effective_suit(c, trump_suit, trump_rank)
        key = (eff, c.rank)
        rank_groups.setdefault(key, []).append(c)

    matching_pairs: list[PlayAction] = []
    for cards in rank_groups.values():
        if len(cards) >= 2:
            matching_pairs.append(PlayAction(type=PlayType.PAIR, cards=cards[:2]))

    if matching_pairs:
        return matching_pairs

    # No matching pair: play any 2 cards
    # Return only the first hand as a representative option to avoid O(n^2) explosion
    if len(hand) >= 2:
        return [PlayAction(type=PlayType.PAIR, cards=hand[:2])]
    return []


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
    rank_groups: dict[tuple[Suit | str, Rank], list[Card]] = {}
    for c in suit_cards:
        eff = effective_suit(c, trump_suit, trump_rank)
        key = (eff, c.rank)
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
        pair_count = len(pairs_in_suit) * 2
        fill_needed = tractor_len - pair_count
        if fill_needed < 0:
            # Too many pairs; just play the first needed pairs
            pairs_needed = tractor_len // 2
            combo_cards: list[Card] = []
            for a in pairs_in_suit[:pairs_needed]:
                combo_cards.extend(a.cards)
            return [PlayAction(type=PlayType.TRACTOR, cards=combo_cards)]
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
            # Must have exactly tractor_len cards; return empty if insufficient
            if len(fill_cards) < fill_needed:
                return []
            combo_cards.extend(fill_cards[:fill_needed])
            return [PlayAction(type=PlayType.TRACTOR, cards=combo_cards)]

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
        # Play exactly throw_len cards of the suit: player chooses which ones
        result: list[PlayAction] = []
        for combo in combinations(suit_cards, throw_len):
            result.append(PlayAction(type=PlayType.THROW, cards=list(combo)))
        return result

    if suit_cards:
        # Play all suit cards + fill from other cards
        other_cards = [c for c in hand if c not in suit_cards]
        fill_needed = throw_len - len(suit_cards)
        if len(other_cards) < fill_needed:
            # Insufficient cards to fill; cannot play this throw
            return []
        fill = other_cards[:fill_needed]
        return [PlayAction(type=PlayType.THROW, cards=suit_cards + fill)]

    # No cards of suit: play any throw_len cards
    if len(hand) >= throw_len:
        return [PlayAction(type=PlayType.THROW, cards=hand[:throw_len])]
    return []


# ---- Decompose ----


def decompose(
    cards: list[Card], trump_suit: Suit | None, trump_rank: Rank
) -> list[SubPlay]:
    """Decompose a set of same-effective-suit cards into SubPlay structures.

    Takes a list of cards that all share the same effective suit and returns
    a list of SubPlay structures by extracting tractors first (longest,
    non-overlapping), then pairs, then singles.

    Returns the list sorted by sub_level descending (tractors first,
    then pairs, then singles).
    """
    if not cards:
        return []

    # Determine effective suit
    eff_suit = effective_suit(cards[0], trump_suit, trump_rank)
    is_trump_group = (eff_suit == "trump")

    # For trump group: group by (rank, suit) to detect cross-suit same-rank pairs
    # For non-trump: group by rank (all cards share the same suit)
    if is_trump_group:
        # Group by (rank, suit) for trump
        rank_suit_groups: dict[tuple[Rank, Suit], list[Card]] = {}
        for c in cards:
            key = (c.rank, c.suit)
            rank_suit_groups.setdefault(key, []).append(c)

        # Find pairs: (rank, suit) groups with >= 2 cards
        # Each such group forms a pair; if >= 4 cards, multiple pairs
        pair_keys: list[tuple[Rank, Suit]] = []
        single_cards: list[Card] = []
        pair_key_cards: dict[tuple[Rank, Suit], list[Card]] = {}
        for key, group_cards in rank_suit_groups.items():
            n_pairs = len(group_cards) // 2
            remainder = len(group_cards) % 2
            pair_key_cards[key] = group_cards
            for _ in range(n_pairs):
                pair_keys.append(key)
            if remainder == 1:
                single_cards.append(group_cards[-1])

        # Create a flat list of pair entries, each pointing to its key and
        # owning a distinct pair of cards from the group.  When a (rank, suit)
        # group has 4+ cards (2-deck game), it contributes multiple pairs; each
        # pair must reference different card instances.
        pair_entries: list[tuple[tuple[Rank, Suit], list[Card]]] = []
        for key, group_cards in rank_suit_groups.items():
            n_pairs = len(group_cards) // 2
            remainder = len(group_cards) % 2
            for i in range(n_pairs):
                pair_entries.append((key, group_cards[i * 2 : i * 2 + 2]))
            if remainder == 1:
                single_cards.append(group_cards[-1])

        # Sort pairs by trump_rank_order position of representative card
        rep_cards: dict[tuple[Rank, Suit], Card] = {}
        for key, pair_cards in pair_entries:
            if key not in rep_cards:
                rep_cards[key] = pair_cards[0]
        pair_entries.sort(
            key=lambda e: trump_rank_order(rep_cards[e[0]], trump_suit, trump_rank)
        )

        # Adjacency for trump group:
        # - Different ranks: adjacent if no other pair's position falls between them
        # - Same rank (different suits): adjacent only if SUIT_OFFSET difference > 1
        #   (non-adjacent suits, with a gap between their position values)
        all_positions = sorted(
            {trump_rank_order(rep_cards[e[0]], trump_suit, trump_rank) for e in pair_entries}
        )

        def _are_adjacent_t(
            e1: tuple[tuple[Rank, Suit], list[Card]],
            e2: tuple[tuple[Rank, Suit], list[Card]],
        ) -> bool:
            rank1, suit1 = e1[0]
            rank2, suit2 = e2[0]
            pos1 = trump_rank_order(rep_cards[e1[0]], trump_suit, trump_rank)
            pos2 = trump_rank_order(rep_cards[e2[0]], trump_suit, trump_rank)

            # Same rank: require non-adjacent suits (SUIT_OFFSET difference > 1)
            if rank1 == rank2:
                offset_diff = abs(SUIT_OFFSET.get(suit1, 0) - SUIT_OFFSET.get(suit2, 0))
                return offset_diff > 1

            # Different ranks: adjacent if no other pair's position falls between them
            lo, hi = min(pos1, pos2), max(pos1, pos2)
            for p in all_positions:
                if lo < p < hi:
                    return False
            return True

        # Find runs of consecutive adjacent pairs
        runs: list[list[int]] = []  # indices into pair_entries
        if pair_entries:
            current_run: list[int] = [0]
            for i in range(1, len(pair_entries)):
                if _are_adjacent_t(pair_entries[current_run[-1]], pair_entries[i]):
                    current_run.append(i)
                else:
                    if len(current_run) >= 2:
                        runs.append(current_run)
                    current_run = [i]
            if len(current_run) >= 2:
                runs.append(current_run)

        # Greedy: extract longest tractors first
        runs.sort(key=lambda r: len(r), reverse=True)
        used_indices: set[int] = set()
        tractor_runs: list[list[int]] = []
        for run in runs:
            if any(idx in used_indices for idx in run):
                continue
            tractor_runs.append(run)
            used_indices.update(run)

        # Build SubPlay list
        result: list[SubPlay] = []
        for run in tractor_runs:
            tractor_cards: list[Card] = []
            for idx in run:
                tractor_cards.extend(pair_entries[idx][1])
            result.append(SubPlay(pair_count=len(run), cards=tractor_cards, suit=eff_suit))

        for idx, (key, pair_cards) in enumerate(pair_entries):
            if idx in used_indices:
                continue
            result.append(SubPlay(pair_count=1, cards=pair_cards, suit=eff_suit))

        for c in single_cards:
            result.append(SubPlay(pair_count=0, cards=[c], suit=eff_suit))

    else:
        # Non-trump: group by rank
        rank_groups: dict[Rank, list[Card]] = {}
        for c in cards:
            rank_groups.setdefault(c.rank, []).append(c)

        # Create a flat list of pair entries, each owning a distinct pair of cards.
        pair_entries: list[tuple[Rank, list[Card]]] = []
        single_cards: list[Card] = []
        for rank, rank_cards in rank_groups.items():
            n_pairs = len(rank_cards) // 2
            remainder = len(rank_cards) % 2
            for i in range(n_pairs):
                pair_entries.append((rank, rank_cards[i * 2 : i * 2 + 2]))
            if remainder == 1:
                single_cards.append(rank_cards[-1])

        pair_entries.sort(key=lambda e: _non_trump_rank_order(e[0], trump_rank))

        # Adjacency for non-trump: consecutive rank order
        def _are_adjacent_nt(e1: tuple[Rank, list[Card]], e2: tuple[Rank, list[Card]]) -> bool:
            o1 = _non_trump_rank_order(e1[0], trump_rank)
            o2 = _non_trump_rank_order(e2[0], trump_rank)
            return abs(o1 - o2) == 1

        # Find runs (indices into pair_entries)
        runs: list[list[int]] = []
        if pair_entries:
            current_run: list[int] = [0]
            for i in range(1, len(pair_entries)):
                if _are_adjacent_nt(pair_entries[current_run[-1]], pair_entries[i]):
                    current_run.append(i)
                else:
                    if len(current_run) >= 2:
                        runs.append(current_run)
                    current_run = [i]
            if len(current_run) >= 2:
                runs.append(current_run)

        # Greedy: extract longest tractors first
        runs.sort(key=lambda r: len(r), reverse=True)
        used_indices: set[int] = set()
        tractor_runs: list[list[int]] = []
        for run in runs:
            if any(idx in used_indices for idx in run):
                continue
            tractor_runs.append(run)
            used_indices.update(run)

        # Build SubPlay list
        result = []
        for run in tractor_runs:
            tractor_cards: list[Card] = []
            for idx in run:
                tractor_cards.extend(pair_entries[idx][1])
            result.append(SubPlay(pair_count=len(run), cards=tractor_cards, suit=eff_suit))

        for idx, (rank, pair_cards) in enumerate(pair_entries):
            if idx in used_indices:
                continue
            result.append(SubPlay(pair_count=1, cards=pair_cards, suit=eff_suit))

        for c in single_cards:
            result.append(SubPlay(pair_count=0, cards=[c], suit=eff_suit))

    # Sort by sub_level descending (tractors first, then pairs, then singles)
    result.sort(key=lambda s: s.sub_level, reverse=True)

    return result
