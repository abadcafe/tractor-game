"""Play rules: pattern detection, following rules, and legal play enumeration.

Implements the Shengji/Tractor play rules for singles, pairs, tractors, throws,
and the legal play enumeration for leading and following.
"""

from itertools import product as iterproduct

from server.sm.card_model import Card, Suit, Rank, SUITED_RANKS
from server.sm.comparator import SUIT_OFFSET, effective_suit, trump_rank_order
from server.sm.types import SubPlay

MAX_LEGAL_PLAY_HINTS = 5


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


def detect_throws(
    hand: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
    other_hands: list[Card],
) -> list[list[Card]]:
    """Detect throw options using decompose + is_biggest verification.

    Per spec sections 7 and 9.7:
    1. Group hand cards by effective suit (INCLUDE trump per spec 7.4).
    2. For each suit group:
       a. If fewer than 2 cards -> skip (not a throw).
       b. Decompose the group into sub-plays.
       c. Verify each sub-play is biggest using _is_biggest.
       d. If all pass -> add all cards in the group as a throw option.
    3. Return list of throw options (at most one per suit).
    """
    groups = _group_cards_by_effective_suit(hand, trump_suit, trump_rank)

    result: list[list[Card]] = []
    for _eff_suit, cards in groups.items():
        # A throw requires 2+ cards in the same effective suit
        if len(cards) < 2:
            continue

        subs = decompose(cards, trump_suit, trump_rank)

        # Verify each sub-play is biggest (from low to high level per spec 7.3)
        sorted_subs = sorted(subs, key=lambda s: s.sub_level)
        all_biggest = all(
            _is_biggest(sub, other_hands, trump_suit, trump_rank)
            for sub in sorted_subs
        )

        if all_biggest:
            result.append(cards)

    return result


def get_legal_plays(
    hand: list[Card],
    is_leading: bool,
    lead_cards: list[Card] | None,
    trump_suit: Suit | None,
    trump_rank: Rank,
    other_hands: list[Card],
) -> list[list[Card]]:
    """Enumerate all legal plays using SubPlay-based decomposition.

    Leading: all singles, pairs, tractors, and throws from hand.
    Following: enumerate options respecting sub-play priority rules.
    """
    if is_leading:
        return _leading_plays(hand, trump_suit, trump_rank, other_hands)

    # Following
    if lead_cards is None or len(lead_cards) == 0:
        return []

    return _following_plays(hand, lead_cards, trump_suit, trump_rank)


def _leading_plays(
    hand: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
    other_hands: list[Card],
) -> list[list[Card]]:
    """Enumerate leading options: singles, pairs, tractors, throws."""
    result: list[list[Card]] = []

    # Group by effective suit
    groups = _group_cards_by_effective_suit(hand, trump_suit, trump_rank)

    for _eff_suit, cards in groups.items():
        subs = decompose(cards, trump_suit, trump_rank)
        # Emit each SubPlay as an option
        for sub in subs:
            result.append(sub.cards)

    # Add throw options
    result.extend(detect_throws(hand, trump_suit, trump_rank, other_hands))

    return result


def _following_plays(
    hand: list[Card],
    lead_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> list[list[Card]]:
    """Enumerate following options respecting sub-play priority rules.

    1. Compute lead_pair_count from decompose(lead_cards)
    2. Separate hand into suit_cards and other_cards
    3. Decompose suit_cards, enumerate pair extraction with branching
    4. Fill remaining slots with singles, then other-suit cards
    5. Validate each option with is_legal_follow
    """
    lead_eff = effective_suit(lead_cards[0], trump_suit, trump_rank)
    lead_count = len(lead_cards)

    # Separate hand into same effective suit vs other
    suit_cards = [c for c in hand if effective_suit(c, trump_suit, trump_rank) == lead_eff]
    other_cards = [c for c in hand if effective_suit(c, trump_suit, trump_rank) != lead_eff]

    # Compute lead pair count
    lead_subs = decompose(lead_cards, trump_suit, trump_rank)
    lead_pair_count = sum(s.pair_count for s in lead_subs)

    # Decompose suit cards
    suit_subs = decompose(suit_cards, trump_suit, trump_rank) if suit_cards else []

    # Enumerate all valid pair extraction branches
    branches = _enumerate_follow_branches(
        suit_subs, other_cards, lead_count, lead_pair_count, suit_cards
    )

    # Validate each branch with is_legal_follow
    result: list[list[Card]] = []
    for play_cards in branches:
        if is_legal_follow(hand, play_cards, lead_cards, trump_suit, trump_rank):
            # Deduplicate
            key = frozenset(c.id for c in play_cards)
            if not any(frozenset(c.id for c in existing) == key for existing in result):
                result.append(play_cards)

    return result


def _enumerate_follow_branches(
    suit_subs: list[SubPlay],
    other_cards: list[Card],
    lead_count: int,
    lead_pair_count: int,
    all_suit_cards: list[Card],
) -> list[list[Card]]:
    """Enumerate branches for following a lead.

    Each branch represents a choice of how many pairs to extract from which
    sub-plays (with branching for tractor starting positions), then fills
    the remaining slots.
    """
    # Separate suit subs by type
    tractor_subs = [s for s in suit_subs if s.pair_count >= 2]
    pair_subs = [s for s in suit_subs if s.pair_count == 1]
    single_subs = [s for s in suit_subs if s.pair_count == 0]

    # Available pair count from suit cards
    avail_pair_count = sum(s.pair_count for s in suit_subs)

    # How many pairs MUST be played (minimum)
    pairs_must = min(avail_pair_count, lead_pair_count)

    # Generate all valid pair extraction combos
    extraction_combos = _generate_extractions(
        tractor_subs, pair_subs, single_subs, pairs_must, lead_count
    )

    # Expand each combo into branches, considering all tractor starting positions.
    # Each combo is a list of (SubPlay, extracted_count). For tractor sub-plays,
    # there are (N - K + 1) possible consecutive K-pair starting positions.
    all_branches: list[list[Card]] = []
    for combo in extraction_combos:
        # Build a list of "options" for each active extraction in the combo.
        # Each option is a list of cards extracted from that sub-play.
        extraction_options: list[list[list[Card]]] = []
        for sub, extracted in combo:
            if extracted == 0:
                continue
            if sub.pair_count == 0 and extracted == 1:
                # Single: one option containing the one card
                extraction_options.append([list(sub.cards)])
            elif sub.pair_count == 1 and extracted == 1:
                # Pair: one option containing the pair
                extraction_options.append([list(sub.cards)])
            elif sub.pair_count >= 2:
                # Tractor: multiple valid starting positions
                all_starts = _extract_from_tractor_all(sub, extracted)
                extraction_options.append(all_starts)

        # Cartesian product of all extraction starting positions
        for branch_cards in iterproduct(*extraction_options):
            used_card_ids: set[str] = set()
            pair_cards_played: list[Card] = []
            for cards in branch_cards:
                pair_cards_played.extend(cards)
                used_card_ids.update(c.id for c in cards)

            remaining_needed = lead_count - len(pair_cards_played)
            if remaining_needed < 0:
                continue
            if remaining_needed == 0:
                if len(pair_cards_played) == lead_count:
                    all_branches.append(pair_cards_played)
                continue

            # Fill remaining with same-suit singles first
            fill_suit = [
                c for c in all_suit_cards
                if c.id not in used_card_ids
            ][:remaining_needed]

            used_card_ids.update(c.id for c in fill_suit)
            fill = list(fill_suit)
            remaining_needed -= len(fill)

            # Fill with other-suit cards
            fill_other = [c for c in other_cards if c.id not in used_card_ids][:remaining_needed]
            fill.extend(fill_other)

            play_cards = pair_cards_played + fill
            if len(play_cards) == lead_count:
                all_branches.append(play_cards)

    # If no suit cards at all (can't follow), play any lead_count cards from hand
    if not all_suit_cards:
        if len(other_cards) >= lead_count:
            all_branches.append(list(other_cards[:lead_count]))

    return all_branches


def _generate_extractions(
    tractor_subs: list[SubPlay],
    pair_subs: list[SubPlay],
    single_subs: list[SubPlay],
    pairs_must: int,
    lead_count: int,
) -> list[list[tuple[SubPlay, int]]]:
    """Generate all valid pair extraction combinations.

    Returns list of (sub, extracted_count) tuples for each sub-play.
    """
    all_subs = tractor_subs + pair_subs + single_subs
    if not all_subs:
        return [[]]

    # For each sub-play, the number of pairs we can extract is:
    #   tractor: 0 to sub.pair_count (but extraction must be contiguous)
    #   pair: 0 or 1
    #   single: 0 or 1 (but counts as a single, not a pair)

    # Generate ranges for each sub
    ranges: list[list[int]] = []
    for s in all_subs:
        if s.pair_count >= 2:
            # Tractor: can extract 0 to pair_count pairs
            ranges.append(list(range(0, s.pair_count + 1)))
        elif s.pair_count == 1:
            # Pair: 0 or 1
            ranges.append([0, 1])
        else:
            # Single: 0 or 1 (but doesn't count as a pair for pair_count)
            ranges.append([0, 1])

    # Generate Cartesian product
    combos: list[list[tuple[SubPlay, int]]] = []

    for extraction_values in iterproduct(*ranges):
        total_pair_count = 0
        total_card_count = 0
        combo: list[tuple[SubPlay, int]] = []

        for s, extracted in zip(all_subs, extraction_values):
            combo.append((s, extracted))
            # Only pairs (pair_count >= 1) contribute to the pair count floor
            if s.pair_count >= 1:
                total_pair_count += extracted
                total_card_count += extracted * 2
            else:
                # Singles contribute card count but not pair count
                total_card_count += extracted

        # Must extract at least pairs_must pairs
        if total_pair_count < pairs_must:
            continue

        # Total cards from pairs must not exceed lead_count
        if total_card_count > lead_count:
            continue

        combos.append(combo)

    return combos


def _extract_from_tractor_all(sub: SubPlay, count: int) -> list[list[Card]]:
    """Extract 'count' pairs from a tractor sub-play (all valid starting positions).

    Returns a list of card lists, one per valid contiguous starting position.
    If count == 0, returns [[]]. If count >= sub.pair_count, returns [all_cards].
    """
    if count == 0:
        return [[]]
    if count >= sub.pair_count:
        return [list(sub.cards)]

    # Get unique ranks in order (they're already in tractor order from decompose)
    unique_ranks: list[Rank] = []
    for c in sub.cards:
        if c.rank not in unique_ranks:
            unique_ranks.append(c.rank)

    # There are (len(unique_ranks) - count + 1) possible starting positions.
    results: list[list[Card]] = []
    num_starts = len(unique_ranks) - count + 1
    for start in range(num_starts):
        extracted: list[Card] = []
        for i in range(start, start + count):
            rank = unique_ranks[i]
            rank_cards = [c for c in sub.cards if c.rank == rank]
            extracted.extend(rank_cards[:2])
        results.append(extracted)

    return results


# ---- Follow Rules ----


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

        # Create a flat list of pair entries, each pointing to its key and
        # owning a distinct pair of cards from the group.  When a (rank, suit)
        # group has 4+ cards (2-deck game), it contributes multiple pairs; each
        # pair must reference different card instances.
        pair_entries: list[tuple[tuple[Rank, Suit], list[Card]]] = []
        single_cards: list[Card] = []
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
        pair_entries: list[tuple[tuple[Rank, Suit], list[Card]]] = []
        single_cards: list[Card] = []
        for rank, rank_cards in rank_groups.items():
            n_pairs = len(rank_cards) // 2
            remainder = len(rank_cards) % 2
            for i in range(n_pairs):
                pair_entries.append(((rank, rank_cards[0].suit), rank_cards[i * 2 : i * 2 + 2]))
            if remainder == 1:
                single_cards.append(rank_cards[-1])

        pair_entries.sort(key=lambda e: _non_trump_rank_order(e[0][0], trump_rank))

        # Adjacency for non-trump: consecutive rank order
        def _are_adjacent_nt(
            e1: tuple[tuple[Rank, Suit], list[Card]],
            e2: tuple[tuple[Rank, Suit], list[Card]],
        ) -> bool:
            o1 = _non_trump_rank_order(e1[0][0], trump_rank)
            o2 = _non_trump_rank_order(e2[0][0], trump_rank)
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

        for idx, ((_rank, _suit), pair_cards) in enumerate(pair_entries):
            if idx in used_indices:
                continue
            result.append(SubPlay(pair_count=1, cards=pair_cards, suit=eff_suit))

        for c in single_cards:
            result.append(SubPlay(pair_count=0, cards=[c], suit=eff_suit))

    # Sort by sub_level descending (tractors first, then pairs, then singles)
    result.sort(key=lambda s: s.sub_level, reverse=True)

    return result


# ---- can_win / compare_plays ----


def can_win(
    played_cards: list[Card],
    lead_eff: Suit | str,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> bool:
    """Check whether a player's cards are eligible to win the trick (spec 8.2).

    For each card: if effective_suit is neither lead_eff nor "trump" -> False.
    Otherwise True.  When lead_eff is "trump", only trump cards are eligible.
    """
    for card in played_cards:
        eff = effective_suit(card, trump_suit, trump_rank)
        if eff != lead_eff and eff != "trump":
            return False
    return True


def _compare_same_suit(
    a_cards: list[Card],
    b_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
    is_trump: bool,
) -> int:
    """Compare two plays that share the same effective suit (spec 8.3-8.4).

    Uses decompose to extract sub-plays, then compares:
    1. Max sub_level (tractor > pair > single)
    2. Same level: max rank of the highest-level sub-plays

    Returns >0 if a wins, <0 if b wins, 0 if tie.
    """
    a_subs = decompose(a_cards, trump_suit, trump_rank)
    b_subs = decompose(b_cards, trump_suit, trump_rank)

    if not a_subs and not b_subs:
        return 0
    if not a_subs:
        return -1
    if not b_subs:
        return 1

    # Find max sub_level for each
    a_max_level = max(s.sub_level for s in a_subs)
    b_max_level = max(s.sub_level for s in b_subs)

    if a_max_level != b_max_level:
        return a_max_level - b_max_level

    # Same max level: compare max rank of the highest-level sub-plays
    a_high_subs = [s for s in a_subs if s.sub_level == a_max_level]
    b_high_subs = [s for s in b_subs if s.sub_level == b_max_level]

    # Get the max rank order across all highest-level sub-plays
    if is_trump:
        # Use trump_rank_order (comparator) which distinguishes sub-types:
        #   trump-suit level=80, other-suit level=70+offset, etc.
        # _rank_order_for_suit only gives组内 position (1-15), which
        # collapses different sub-types at the same rank.
        a_max_rank = max(
            trump_rank_order(c, trump_suit, trump_rank)
            for s in a_high_subs
            for c in s.cards
        )
        b_max_rank = max(
            trump_rank_order(c, trump_suit, trump_rank)
            for s in b_high_subs
            for c in s.cards
        )
    else:
        a_max_rank = max(
            _non_trump_rank_order(c.rank, trump_rank)
            for s in a_high_subs
            for c in s.cards
        )
        b_max_rank = max(
            _non_trump_rank_order(c.rank, trump_rank)
            for s in b_high_subs
            for c in s.cards
        )

    return a_max_rank - b_max_rank


def compare_plays(
    a_cards: list[Card],
    b_cards: list[Card],
    lead_eff: Suit | str,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> int:
    """Compare two plays using sub-level decomposition (spec 8.3-8.4).

    Returns >0 if a wins, <0 if b wins, 0 if tie.

    1. can_win eligibility gating (spec 8.2)
    2. Trump vs non-trump
    3. Same suit: decompose-based comparison
    """
    a_eligible = can_win(a_cards, lead_eff, trump_suit, trump_rank)
    b_eligible = can_win(b_cards, lead_eff, trump_suit, trump_rank)

    if a_eligible and not b_eligible:
        return 1
    if b_eligible and not a_eligible:
        return -1
    if not a_eligible and not b_eligible:
        return 0

    # Both eligible: determine effective suit groups
    a_all_trump = all(
        effective_suit(c, trump_suit, trump_rank) == "trump" for c in a_cards
    )
    b_all_trump = all(
        effective_suit(c, trump_suit, trump_rank) == "trump" for c in b_cards
    )

    if a_all_trump and not b_all_trump:
        return 1  # trump beats non-trump
    if b_all_trump and not a_all_trump:
        return -1

    if a_all_trump and b_all_trump:
        return _compare_same_suit(a_cards, b_cards, trump_suit, trump_rank, is_trump=True)

    # Both lead-suit (non-trump)
    return _compare_same_suit(a_cards, b_cards, trump_suit, trump_rank, is_trump=False)


# ---- is_legal_lead ----


def _is_biggest(
    sub: SubPlay,
    other_hands: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> bool:
    """Check if a sub-play is the biggest of its type in its effective suit.

    For single (pair_count=0): no other card of same effective suit has higher
    RANK_ORDER (or higher trump_rank_order for trump group).
    For pair (pair_count=1): no other pair of same effective suit has higher RANK_ORDER.
    For tractor (pair_count>=2): no other tractor of same effective suit with
    same or greater length beats it.
    """
    eff = sub.suit
    cards = sub.cards
    pair_count = sub.pair_count

    # Filter other_hands to same effective suit
    others = [c for c in other_hands if effective_suit(c, trump_suit, trump_rank) == eff]

    if pair_count == 0:
        # Single: check if any other card has higher rank
        sub_rank = cards[0].rank
        sub_order = _rank_order_for_suit(sub_rank, eff, trump_suit, trump_rank)
        for c in others:
            other_order = _rank_order_for_suit(c.rank, eff, trump_suit, trump_rank)
            if other_order > sub_order:
                return False
        return True

    if pair_count == 1:
        # Pair: check if any other pair has higher rank
        sub_rank = cards[0].rank
        sub_order = _rank_order_for_suit(sub_rank, eff, trump_suit, trump_rank)

        # Group others by rank
        other_by_rank: dict[Rank, list[Card]] = {}
        for c in others:
            other_by_rank.setdefault(c.rank, []).append(c)

        for rank, rank_cards in other_by_rank.items():
            if len(rank_cards) >= 2:
                other_order = _rank_order_for_suit(rank, eff, trump_suit, trump_rank)
                if other_order > sub_order:
                    return False
        return True

    # pair_count >= 2: tractor
    # Decompose others to find all tractors
    other_subs = decompose(others, trump_suit, trump_rank) if others else []
    other_tractors = [s for s in other_subs if s.pair_count >= 2]

    for ot in other_tractors:
        # If other tractor is longer, it contains a same-length sub-tractor
        # that can beat sub (longer tractor always wins).
        if ot.pair_count > pair_count:
            return False
        # Same length: compare max rank order
        if ot.pair_count == pair_count:
            sub_max_order = max(
                _rank_order_for_suit(c.rank, eff, trump_suit, trump_rank)
                for c in cards
            )
            ot_max_order = max(
                _rank_order_for_suit(c.rank, eff, trump_suit, trump_rank)
                for c in ot.cards
            )
            if ot_max_order > sub_max_order:
                return False

    return True


def is_legal_lead(
    hand: list[Card],
    played_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
    other_hands: list[Card],
) -> bool:
    """Verify that a leading play is legal per spec section 6.1.

    1. played_cards must be a subset of hand
    2. All played cards must have the same effective suit
    3. If decomposition yields multiple sub-plays (throw), verify each sub-play
       is the biggest of its type using verify_throw logic (spec 7.3).
    """
    if not played_cards:
        return False

    # Step 1: played_cards must be a subset of hand
    hand_ids = {c.id for c in hand}
    for c in played_cards:
        if c.id not in hand_ids:
            return False

    # Step 2: all played cards must have the same effective suit
    eff_suits = {effective_suit(c, trump_suit, trump_rank) for c in played_cards}
    if len(eff_suits) != 1:
        return False

    # Step 3: decompose and verify throw if multi-sub-play
    subs = decompose(played_cards, trump_suit, trump_rank)
    if len(subs) > 1:
        # Verify from lowest level to highest (spec 7.3)
        sorted_subs = sorted(subs, key=lambda s: s.sub_level)
        for sub in sorted_subs:
            if not _is_biggest(sub, other_hands, trump_suit, trump_rank):
                return False

    return True


# ---- is_legal_follow ----


def is_legal_follow(
    hand: list[Card],
    played_cards: list[Card],
    lead_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> bool:
    """Verify that a following play is legal per spec section 6.2.

    Validates:
    1. played count == lead count
    2. played_cards are all in hand
    3. Suit-following rules
    4. Sub-play priority rules (spec 6.2 steps 7a/7b/7c)
    """
    # Step 1: count must match
    if len(played_cards) != len(lead_cards):
        return False

    if not played_cards:
        return False

    # Step 2: all played cards must be in hand
    hand_ids = {c.id for c in hand}
    for c in played_cards:
        if c.id not in hand_ids:
            return False

    # Step 3: compute lead effective suit
    lead_eff = effective_suit(lead_cards[0], trump_suit, trump_rank)

    # Step 4: separate hand into same effective suit vs other
    suit_in_hand = [c for c in hand if effective_suit(c, trump_suit, trump_rank) == lead_eff]

    # Step 5: separate played into same effective suit vs other
    suit_in_played = [c for c in played_cards if effective_suit(c, trump_suit, trump_rank) == lead_eff]

    # Step 6: suit-following check
    if len(suit_in_hand) >= len(lead_cards):
        # Must play exactly lead_count cards of the lead suit
        if len(suit_in_played) != len(lead_cards):
            return False
    else:
        # Don't have enough lead-suit cards: must play ALL of them
        if len(suit_in_played) < len(suit_in_hand):
            return False

    # Step 7: sub-play priority verification
    return _verify_follow_sub_play_priority(
        suit_in_hand, suit_in_played, lead_cards, lead_eff, trump_suit, trump_rank,
    )


def _verify_follow_sub_play_priority(
    hand_suit_cards: list[Card],
    played_suit_cards: list[Card],
    lead_cards: list[Card],
    lead_eff: Suit | str,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> bool:
    """Verify sub-play priority rules when following (spec 6.2 steps 7a/7b/7c).

    7a. Pair count floor: played must use at least min(available_pairs, lead_pairs) pairs.
    7b. Level-by-level: higher-level sub-plays must be used before lower ones.
    7c. Tractor continuity: partial extraction from a tractor must be contiguous.
    """
    if not played_suit_cards:
        return True

    # Decompose hand and played into SubPlay structures
    hand_subs = decompose(hand_suit_cards, trump_suit, trump_rank) if hand_suit_cards else []
    played_subs = decompose(played_suit_cards, trump_suit, trump_rank) if played_suit_cards else []

    # 7a. Pair count floor check
    lead_pair_count = sum(
        s.pair_count for s in decompose(lead_cards, trump_suit, trump_rank)
    )
    hand_avail_pair_count = sum(s.pair_count for s in hand_subs)
    played_pair_count = sum(s.pair_count for s in played_subs)

    pair_floor = min(hand_avail_pair_count, lead_pair_count)
    if played_pair_count < pair_floor:
        return False

    # 7b. Level-by-level priority check
    # For each hand sub-play, determine how many of its cards were played.
    # Then check that from highest pair_count level to lowest, the played set
    # uses pairs from higher-level sub-plays first.

    played_card_ids = {c.id for c in played_suit_cards}

    # For each hand sub-play, count pairs played from it
    # pair_count=0 (single): 1 card played if its ID is in played set
    # pair_count=1 (pair): pair_count played if both cards' IDs are in played set
    # pair_count>=2 (tractor): count how many of its ranks have both cards played

    # Group hand sub-plays by pair_count level, counting how many pairs
    # from each sub-play are present in the played set
    hand_pairs_by_level: dict[int, int] = {}  # pair_count_level -> total pairs available in hand
    played_pairs_from_hand_by_level: dict[int, int] = {}  # pair_count_level -> pairs actually played from hand subs

    for sub in hand_subs:
        pc = sub.pair_count
        hand_pairs_by_level[pc] = hand_pairs_by_level.get(pc, 0) + pc

        # Count how many cards from this sub-play were played
        cards_played = sum(1 for c in sub.cards if c.id in played_card_ids)

        if pc == 0:
            # Single: 1 card = 1 pair at level 0
            if cards_played == 1:
                played_pairs_from_hand_by_level[pc] = played_pairs_from_hand_by_level.get(pc, 0) + 1
        elif pc == 1:
            # Pair: 2 cards played = 1 pair at level 1
            if cards_played == 2:
                played_pairs_from_hand_by_level[pc] = played_pairs_from_hand_by_level.get(pc, 0) + 1
        else:
            # Tractor: pc pairs. Count how many ranks have both cards played
            rank_played_count: dict[Rank, int] = {}
            for c in sub.cards:
                if c.id in played_card_ids:
                    rank_played_count[c.rank] = rank_played_count.get(c.rank, 0) + 1
            pairs_from_tractor = sum(1 for count in rank_played_count.values() if count >= 2)
            played_pairs_from_hand_by_level[pc] = played_pairs_from_hand_by_level.get(pc, 0) + pairs_from_tractor

    # remaining_needed starts at total pairs in lead
    remaining_needed = lead_pair_count

    # Process from highest level to lowest
    all_levels = sorted(set(list(hand_pairs_by_level.keys()) + list(played_pairs_from_hand_by_level.keys())), reverse=True)

    for level in all_levels:
        hand_count = hand_pairs_by_level.get(level, 0)
        played_count = played_pairs_from_hand_by_level.get(level, 0)

        if hand_count == 0:
            continue

        expected = min(hand_count, remaining_needed)
        if played_count < expected:
            return False

        remaining_needed -= played_count

    # 7c. Tractor continuity check
    # For each hand sub-play with pair_count >= 2 (tractor):
    # if 0 < played_pairs_in_sub < sub.pair_count (partial extraction),
    # verify the extracted pairs form a contiguous block.
    for sub in hand_subs:
        if sub.pair_count < 2:
            continue

        # Count how many cards from this sub-play were played
        rank_played_count: dict[Rank, int] = {}
        for c in sub.cards:
            if c.id in played_card_ids:
                rank_played_count[c.rank] = rank_played_count.get(c.rank, 0) + 1
        pairs_played = sum(1 for count in rank_played_count.values() if count >= 2)

        if pairs_played == 0 or pairs_played == sub.pair_count:
            # Not extracted at all, or fully extracted -- both fine
            continue

        # Partial extraction: check contiguity
        # Get unique ranks that were played as pairs
        unique_ranks = list(dict.fromkeys(c.rank for c in sub.cards))
        played_ranks = [r for r in unique_ranks if rank_played_count.get(r, 0) >= 2]

        if not played_ranks:
            continue

        # Get positions of the extracted ranks within the tractor's sorted rank list
        is_trump_group = (lead_eff == "trump")
        if is_trump_group:
            sorted_ranks = sorted(
                unique_ranks,
                key=lambda rr: _trump_rank_order(rr),
            )
        else:
            sorted_ranks = sorted(
                unique_ranks,
                key=lambda rr: _non_trump_rank_order(rr, trump_rank),
            )

        positions = [sorted_ranks.index(r) for r in played_ranks]
        positions.sort()

        # Check contiguity: positions must form a range [min, max] with no gaps
        if positions[-1] - positions[0] != len(positions) - 1:
            return False

    return True
