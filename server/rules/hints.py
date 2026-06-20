"""Bounded legal play hint enumeration and sorting."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations, product as iterproduct

from server.result import Ok, Rejected

from .cards import Card, Rank, Suit
from .decompose import extract_tractor_pairs, decompose
from .follow import is_legal_follow
from .ordering import effective_suit, trump_order
from .rejections import TooManyPlayHintsRejected
from .types import SubPlay

MAX_PLAY_ACTION_HINTS: int = 25
TOO_MANY_PLAY_HINTS: str = TooManyPlayHintsRejected.reason_text

type PlayActionHintSortKey = tuple[int, int, tuple[int, ...], tuple[str, ...]]


def get_legal_play_hints(
    hand: list[Card],
    lead_cards: list[Card] | None,
    trump_suit: Suit | None,
    trump_rank: Rank,
    max_hints: int,
) -> Ok[list[list[Card]]] | Rejected:
    """Return a bounded complete hint set for following plays.

    Empty lead cards return no hints because leading/throwing is free-form.
    Following candidates are enumerated until either the complete unique legal
    set is found within max_hints or the max_hints + 1-th unique legal
    candidate is discovered, in which case no closed hint set should be shown.
    """
    assert max_hints >= 0
    if lead_cards is None or len(lead_cards) == 0:
        return Ok([])

    return _collect_bounded_follow_hints(
        _iter_follow_play_candidates(hand, lead_cards, trump_suit, trump_rank),
        hand,
        lead_cards,
        trump_suit,
        trump_rank,
        max_hints,
    )

def sort_play_action_hints(
    hints: list[list[Card]],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> list[list[Card]]:
    """Sort player-facing play hints from weakest to strongest."""
    return sorted(
        [list(cards) for cards in hints],
        key=lambda cards: _play_action_hint_sort_key(
            cards, trump_suit, trump_rank
        ),
    )

def _play_action_hint_sort_key(
    cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> PlayActionHintSortKey:
    subs = decompose(cards, trump_suit, trump_rank) if cards else []
    max_sub_level = max((sub.sub_level for sub in subs), default=0)
    card_orders = tuple(
        sorted(trump_order(card, trump_suit, trump_rank) for card in cards)
    )
    card_ids = tuple(sorted(card.id for card in cards))
    return (len(cards), max_sub_level, card_orders, card_ids)

def _collect_bounded_follow_hints(
    candidates: Iterator[list[Card]],
    hand: list[Card],
    lead_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
    max_hints: int,
) -> Ok[list[list[Card]]] | Rejected:
    result: list[list[Card]] = []
    seen: set[frozenset[str]] = set()
    for candidate in candidates:
        if not is_legal_follow(hand, candidate, lead_cards, trump_suit, trump_rank):
            continue
        key = frozenset(card.id for card in candidate)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
        if len(result) > max_hints:
            return TooManyPlayHintsRejected()
    return Ok(result)

def _iter_follow_play_candidates(
    hand: list[Card],
    lead_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> Iterator[list[Card]]:
    """Yield following candidates respecting sub-play priority rules.

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

    if not suit_cards:
        for cards in combinations(other_cards, lead_count):
            yield list(cards)
        return

    # Compute lead pair count
    lead_subs = decompose(lead_cards, trump_suit, trump_rank)
    lead_pair_count = sum(s.pair_count for s in lead_subs)

    # Decompose suit cards
    suit_subs = decompose(suit_cards, trump_suit, trump_rank) if suit_cards else []

    # Enumerate all valid pair extraction branches
    yield from _iter_follow_branches(
        suit_subs, other_cards, lead_count, lead_pair_count, suit_cards
    )

def _iter_follow_branches(
    suit_subs: list[SubPlay],
    other_cards: list[Card],
    lead_count: int,
    lead_pair_count: int,
    all_suit_cards: list[Card],
) -> Iterator[list[Card]]:
    """Yield branches for following a lead.

    Each branch represents a choice of how many pairs to extract from which
    sub-plays (with branching for tractor starting positions), then fills
    the remaining slots.
    """
    # Separate suit subs by type
    tractor_subs = [s for s in suit_subs if s.pair_count >= 2]
    pair_subs = [s for s in suit_subs if s.pair_count == 1]

    # Available pair count from suit cards
    avail_pair_count = sum(s.pair_count for s in suit_subs)

    # How many pairs MUST be played (minimum)
    pairs_must = min(avail_pair_count, lead_pair_count)

    # Generate all valid pair extraction combos
    # Expand each combo into branches, considering all tractor starting positions.
    # Each combo is a list of (SubPlay, extracted_count). For tractor sub-plays,
    # there are (N - K + 1) possible consecutive K-pair starting positions.
    for combo in _iter_extractions(tractor_subs, pair_subs, pairs_must, lead_count):
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
                all_starts = extract_tractor_pairs(sub, extracted)
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
                    yield pair_cards_played
                continue

            yield from _iter_fill_branches(
                pair_cards_played,
                used_card_ids,
                all_suit_cards,
                other_cards,
                remaining_needed,
            )

def _iter_fill_branches(
    selected_cards: list[Card],
    used_card_ids: set[str],
    all_suit_cards: list[Card],
    other_cards: list[Card],
    remaining_needed: int,
) -> Iterator[list[Card]]:
    remaining_suit_cards = [
        card for card in all_suit_cards
        if card.id not in used_card_ids
    ]
    if len(remaining_suit_cards) >= remaining_needed:
        for fill_suit in combinations(remaining_suit_cards, remaining_needed):
            yield selected_cards + list(fill_suit)
        return

    off_suit_needed = remaining_needed - len(remaining_suit_cards)
    remaining_other_cards = [
        card for card in other_cards
        if card.id not in used_card_ids
    ]
    for fill_other in combinations(remaining_other_cards, off_suit_needed):
        yield selected_cards + remaining_suit_cards + list(fill_other)

def _iter_extractions(
    tractor_subs: list[SubPlay],
    pair_subs: list[SubPlay],
    pairs_must: int,
    lead_count: int,
) -> Iterator[list[tuple[SubPlay, int]]]:
    """Yield valid pair extraction combinations.

    Yields lists of (sub, extracted_count) tuples for each pair/tractor
    sub-play. Singles are filled later by _iter_fill_branches.
    """
    all_subs = tractor_subs + pair_subs
    if not all_subs:
        yield []
        return

    # For each sub-play, the number of pairs we can extract is:
    #   tractor: 0 to sub.pair_count (but extraction must be contiguous)
    #   pair: 0 or 1

    # Generate ranges for each sub
    ranges: list[list[int]] = []
    for s in all_subs:
        if s.pair_count >= 2:
            # Tractor: can extract 0 to pair_count pairs
            ranges.append(list(range(0, s.pair_count + 1)))
        elif s.pair_count == 1:
            # Pair: 0 or 1
            ranges.append([0, 1])

    # Generate Cartesian product
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

        yield combo
