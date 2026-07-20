"""Play decomposition rules for singles, pairs, and tractors."""

from __future__ import annotations

from .cards import SUITED_RANKS, Card, Rank, Suit
from .ordering import SUIT_OFFSET, effective_suit, trump_rank_order
from .types import EffectiveSuit, SubPlay


def non_trump_rank_order(rank: Rank, trump_rank: Rank) -> int:
    """Return rank order for non-trump suits, skipping the trump_rank.

    Spec: "级牌Rank从非主排序中移除"
    This means for non-trump suits, the trump rank is excluded from the
    consecutive rank ordering used for tractor detection.

    For trump suit itself, the full rank ordering applies including
    trump_rank.
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


def rank_order_in_trump_group(rank: Rank) -> int:
    """
    Return rank ordering for the trump suit (full ordering including all
    ranks).
    """
    for i, r in enumerate(SUITED_RANKS):
        if r == rank:
            return i + 1
    # Jokers in trump
    if rank == Rank.SMALL_JOKER:
        return 14
    if rank == Rank.BIG_JOKER:
        return 15
    return 0


def rank_order_for_effective_suit(
    rank: Rank,
    suit: EffectiveSuit,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> int:
    """
    Return the rank ordering for a card's rank within its effective suit
    group.

    For trump cards (effective suit "trump" or actual trump suit), use
    trump rank ordering.
    For non-trump cards, use non-trump ordering (skipping trump_rank).
    """
    if suit == "trump":
        return rank_order_in_trump_group(rank)
    if suit == Suit.JOKER:
        return rank_order_in_trump_group(rank)
    if trump_suit is not None and suit == trump_suit:
        return rank_order_in_trump_group(rank)
    return non_trump_rank_order(rank, trump_rank)


def decompose(
    cards: list[Card], trump_suit: Suit | None, trump_rank: Rank
) -> list[SubPlay]:
    """
    Decompose a set of same-effective-suit cards into SubPlay
    structures.

    Takes a list of cards that all share the same effective suit and
    returns
    a list of SubPlay structures by extracting tractors first (longest,
    non-overlapping), then pairs, then singles.

    Returns the list sorted by sub_level descending (tractors first,
    then pairs, then singles).
    """
    if not cards:
        return []

    # Determine effective suit
    eff_suit = effective_suit(cards[0], trump_suit, trump_rank)
    is_trump_group = eff_suit == "trump"

    # For trump group: group by (rank, suit) to detect cross-suit
    # same-rank pairs
    # For non-trump: group by rank (all cards share the same suit)
    if is_trump_group:
        # Group by (rank, suit) for trump
        rank_suit_groups: dict[tuple[Rank, Suit], list[Card]] = {}
        for c in cards:
            key = (c.rank, c.suit)
            rank_suit_groups.setdefault(key, []).append(c)

        # Create a flat list of pair entries, each pointing to its key
        # and
        # owning a distinct pair of cards from the group.  When a (rank,
        # suit)
        # group has 4+ cards (2-deck game), it contributes multiple
        # pairs; each
        # pair must reference different card instances.
        pair_entries: list[tuple[tuple[Rank, Suit], list[Card]]] = []
        single_cards: list[Card] = []
        for key, group_cards in rank_suit_groups.items():
            n_pairs = len(group_cards) // 2
            remainder = len(group_cards) % 2
            for i in range(n_pairs):
                pair_entries.append(
                    (key, group_cards[i * 2 : i * 2 + 2])
                )
            if remainder == 1:
                single_cards.append(group_cards[-1])

        # Sort pairs by trump_rank_order position of representative card
        rep_cards: dict[tuple[Rank, Suit], Card] = {}
        for key, pair_cards in pair_entries:
            if key not in rep_cards:
                rep_cards[key] = pair_cards[0]
        pair_entries.sort(
            key=lambda entry: (
                trump_rank_order(
                    rep_cards[entry[0]], trump_suit, trump_rank
                ),
                SUIT_OFFSET[entry[0][1]],
            )
        )

        # Adjacency for trump group:
        # - Different ranks: adjacent if no other pair's position falls
        # between them
        # - Same rank (different suits): adjacent only if SUIT_OFFSET
        # difference > 1
        # (non-adjacent suits, with a gap between their position values)
        all_positions = sorted(
            {
                trump_rank_order(
                    rep_cards[e[0]], trump_suit, trump_rank
                )
                for e in pair_entries
            }
        )

        def _are_adjacent_t(
            e1: tuple[tuple[Rank, Suit], list[Card]],
            e2: tuple[tuple[Rank, Suit], list[Card]],
        ) -> bool:
            rank1, suit1 = e1[0]
            rank2, suit2 = e2[0]
            pos1 = trump_rank_order(
                rep_cards[e1[0]], trump_suit, trump_rank
            )
            pos2 = trump_rank_order(
                rep_cards[e2[0]], trump_suit, trump_rank
            )

            # Same rank: require non-adjacent suits (SUIT_OFFSET
            # difference > 1)
            if rank1 == rank2:
                offset_diff = abs(
                    SUIT_OFFSET.get(suit1, 0)
                    - SUIT_OFFSET.get(suit2, 0)
                )
                return offset_diff > 1

            # Different ranks: adjacent if no other pair's position
            # falls between them
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
                if _are_adjacent_t(
                    pair_entries[current_run[-1]], pair_entries[i]
                ):
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
            result.append(
                SubPlay(
                    pair_count=len(run),
                    cards=tractor_cards,
                    suit=eff_suit,
                )
            )

        for idx, (key, pair_cards) in enumerate(pair_entries):
            if idx in used_indices:
                continue
            result.append(
                SubPlay(pair_count=1, cards=pair_cards, suit=eff_suit)
            )

        for c in single_cards:
            result.append(
                SubPlay(pair_count=0, cards=[c], suit=eff_suit)
            )

    else:
        # Non-trump: group by rank
        rank_groups: dict[Rank, list[Card]] = {}
        for c in cards:
            rank_groups.setdefault(c.rank, []).append(c)

        # Create a flat list of pair entries, each owning a distinct
        # pair of cards.
        pair_entries: list[tuple[tuple[Rank, Suit], list[Card]]] = []
        single_cards: list[Card] = []
        for rank, rank_cards in rank_groups.items():
            n_pairs = len(rank_cards) // 2
            remainder = len(rank_cards) % 2
            for i in range(n_pairs):
                pair_entries.append(
                    (
                        (rank, rank_cards[0].suit),
                        rank_cards[i * 2 : i * 2 + 2],
                    )
                )
            if remainder == 1:
                single_cards.append(rank_cards[-1])

        pair_entries.sort(
            key=lambda e: non_trump_rank_order(e[0][0], trump_rank)
        )

        # Adjacency for non-trump: consecutive rank order
        def _are_adjacent_nt(
            e1: tuple[tuple[Rank, Suit], list[Card]],
            e2: tuple[tuple[Rank, Suit], list[Card]],
        ) -> bool:
            o1 = non_trump_rank_order(e1[0][0], trump_rank)
            o2 = non_trump_rank_order(e2[0][0], trump_rank)
            return abs(o1 - o2) == 1

        # Find runs (indices into pair_entries)
        runs: list[list[int]] = []
        if pair_entries:
            current_run: list[int] = [0]
            for i in range(1, len(pair_entries)):
                if _are_adjacent_nt(
                    pair_entries[current_run[-1]], pair_entries[i]
                ):
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
            result.append(
                SubPlay(
                    pair_count=len(run),
                    cards=tractor_cards,
                    suit=eff_suit,
                )
            )

        for idx, ((_rank, _suit), pair_cards) in enumerate(
            pair_entries
        ):
            if idx in used_indices:
                continue
            result.append(
                SubPlay(pair_count=1, cards=pair_cards, suit=eff_suit)
            )

        for c in single_cards:
            result.append(
                SubPlay(pair_count=0, cards=[c], suit=eff_suit)
            )

    # Sort by sub_level descending (tractors first, then pairs, then
    # singles)
    result.sort(key=lambda s: s.sub_level, reverse=True)

    return result


def extract_tractor_pairs(sub: SubPlay, count: int) -> list[list[Card]]:
    """
    Extract 'count' pairs from a tractor sub-play (all valid starting
    positions).

    Returns a list of card lists, one per valid contiguous starting
    position.
    If count == 0, returns [[]]. If count >= sub.pair_count, returns
    [all_cards].
    """
    if count == 0:
        return [[]]
    if count >= sub.pair_count:
        return [list(sub.cards)]

    # Get unique ranks in order (they're already in tractor order from
    # decompose)
    unique_ranks: list[Rank] = []
    for c in sub.cards:
        if c.rank not in unique_ranks:
            unique_ranks.append(c.rank)

    # There are (len(unique_ranks) - count + 1) possible starting
    # positions.
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
