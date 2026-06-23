"""Lead and follow legality rules."""

from __future__ import annotations

from server.result import Rejected

from .cards import Card, Rank, Suit
from .decompose import (
    decompose,
    non_trump_rank_order,
    rank_order_in_trump_group,
)
from .ordering import effective_suit
from .rejections.card import CardNotInHandRejected
from .rejections.play import (
    EmptyFollowRejected,
    EmptyLeadRejected,
    IllegalFollowShapeRejected,
    MustExhaustLeadSuitRejected,
    MustExhaustTrumpRejected,
    MustFollowHigherPatternRejected,
    MustFollowLeadSuitRejected,
    MustFollowPairsRejected,
    MustFollowTrumpRejected,
    WrongFollowCountRejected,
)
from .types import EffectiveSuit, PlayShapeInfo


def is_legal_lead(
    hand: list[Card],
    played_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
    _other_players_hands: list[list[Card]],
) -> bool:
    """Verify that a leading play attempt can be submitted.

    All leading plays are throw attempts. A failed throw is still
    accepted by
    the protocol and resolved to the forced small sub-play by
    resolve_lead_throw.
    This predicate only checks the input constraints that can still
    reject the
    message: non-empty, in hand, and one effective suit.
    """
    if not played_cards:
        return False

    # Step 1: played_cards must be a subset of hand
    hand_ids = {c.id for c in hand}
    for c in played_cards:
        if c.id not in hand_ids:
            return False

    # Step 2: all played cards must have the same effective suit
    eff_suits = {
        effective_suit(c, trump_suit, trump_rank) for c in played_cards
    }
    if len(eff_suits) != 1:
        return False

    return True


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
    suit_in_hand = [
        c
        for c in hand
        if effective_suit(c, trump_suit, trump_rank) == lead_eff
    ]

    # Step 5: separate played into same effective suit vs other
    suit_in_played = [
        c
        for c in played_cards
        if effective_suit(c, trump_suit, trump_rank) == lead_eff
    ]

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
        suit_in_hand,
        suit_in_played,
        lead_cards,
        lead_eff,
        trump_suit,
        trump_rank,
    )


def illegal_follow_rejection(
    hand: list[Card],
    played_cards: list[Card],
    lead_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> Rejected:
    """Return a structured rejection for an illegal following play."""
    if is_legal_follow(
        hand, played_cards, lead_cards, trump_suit, trump_rank
    ):
        return IllegalFollowShapeRejected(
            _play_shape_info(lead_cards, trump_suit, trump_rank)
        )

    if not lead_cards:
        return EmptyLeadRejected()

    lead_count = len(lead_cards)
    if len(played_cards) != lead_count:
        return WrongFollowCountRejected(lead_count)

    if not played_cards:
        return EmptyFollowRejected()

    hand_ids = {card.id for card in hand}
    for card in played_cards:
        if card.id not in hand_ids:
            return CardNotInHandRejected(card.id, current=True)

    lead_eff = effective_suit(lead_cards[0], trump_suit, trump_rank)
    suit_in_hand = [
        card
        for card in hand
        if effective_suit(card, trump_suit, trump_rank) == lead_eff
    ]
    suit_in_played = [
        card
        for card in played_cards
        if effective_suit(card, trump_suit, trump_rank) == lead_eff
    ]

    if (
        len(suit_in_hand) >= lead_count
        and len(suit_in_played) != lead_count
    ):
        if lead_eff == "trump":
            return MustFollowTrumpRejected()
        return MustFollowLeadSuitRejected(lead_eff)

    if len(suit_in_hand) < lead_count and len(suit_in_played) < len(
        suit_in_hand
    ):
        if lead_eff == "trump":
            return MustExhaustTrumpRejected(len(suit_in_hand))
        return MustExhaustLeadSuitRejected(lead_eff, len(suit_in_hand))

    pair_reason = _explain_follow_pair_priority(
        suit_in_hand,
        suit_in_played,
        lead_cards,
        trump_suit,
        trump_rank,
    )
    if pair_reason is not None:
        return pair_reason

    return IllegalFollowShapeRejected(
        _play_shape_info(lead_cards, trump_suit, trump_rank)
    )


def _explain_follow_pair_priority(
    hand_suit_cards: list[Card],
    played_suit_cards: list[Card],
    lead_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> Rejected | None:
    hand_subs = (
        decompose(hand_suit_cards, trump_suit, trump_rank)
        if hand_suit_cards
        else []
    )
    played_subs = (
        decompose(played_suit_cards, trump_suit, trump_rank)
        if played_suit_cards
        else []
    )
    lead_subs = decompose(lead_cards, trump_suit, trump_rank)
    lead_pair_count = sum(sub.pair_count for sub in lead_subs)
    hand_avail_pair_count = sum(sub.pair_count for sub in hand_subs)
    played_pair_count = sum(sub.pair_count for sub in played_subs)

    pair_floor = min(hand_avail_pair_count, lead_pair_count)
    if pair_floor > 0 and played_pair_count < pair_floor:
        return MustFollowPairsRejected(
            lead_pair_count=lead_pair_count,
            lead_suit=effective_suit(
                lead_cards[0], trump_suit, trump_rank
            ),
            hand_pair_count=hand_avail_pair_count,
            pair_floor=pair_floor,
        )

    if not _verify_follow_sub_play_priority(
        hand_suit_cards,
        played_suit_cards,
        lead_cards,
        effective_suit(lead_cards[0], trump_suit, trump_rank),
        trump_suit,
        trump_rank,
    ):
        return MustFollowHigherPatternRejected(
            _play_shape_info(lead_cards, trump_suit, trump_rank)
        )
    return None


def _play_shape_info(
    cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> PlayShapeInfo:
    if not cards:
        return PlayShapeInfo(kind="empty", suit=None, card_count=0)
    suit = effective_suit(cards[0], trump_suit, trump_rank)
    subs = decompose(cards, trump_suit, trump_rank)
    if len(cards) == 1:
        return PlayShapeInfo(kind="single", suit=suit, card_count=1)
    if len(subs) == 1:
        sub = subs[0]
        if sub.pair_count >= 2:
            return PlayShapeInfo(
                kind="tractor",
                suit=suit,
                card_count=len(cards),
                pair_count=sub.pair_count,
            )
        if sub.pair_count == 1:
            return PlayShapeInfo(
                kind="pair",
                suit=suit,
                card_count=len(cards),
                pair_count=1,
            )
    return PlayShapeInfo(kind="cards", suit=suit, card_count=len(cards))


def _verify_follow_sub_play_priority(
    hand_suit_cards: list[Card],
    played_suit_cards: list[Card],
    lead_cards: list[Card],
    lead_eff: EffectiveSuit,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> bool:
    """
    Verify sub-play priority rules when following (spec 6.2 steps
    7a/7b/7c).

    7a. Pair count floor: played must use at least min(available_pairs,
    lead_pairs) pairs.
    7b. Level-by-level: higher-level sub-plays must be used before lower
    ones.
    7c. Tractor continuity: partial extraction from a tractor must be
    contiguous.
    """
    if not played_suit_cards:
        return True

    # Decompose hand and played into SubPlay structures
    hand_subs = (
        decompose(hand_suit_cards, trump_suit, trump_rank)
        if hand_suit_cards
        else []
    )
    played_subs = (
        decompose(played_suit_cards, trump_suit, trump_rank)
        if played_suit_cards
        else []
    )

    # 7a. Pair count floor check
    lead_pair_count = sum(
        s.pair_count
        for s in decompose(lead_cards, trump_suit, trump_rank)
    )
    hand_avail_pair_count = sum(s.pair_count for s in hand_subs)
    played_pair_count = sum(s.pair_count for s in played_subs)

    pair_floor = min(hand_avail_pair_count, lead_pair_count)
    if played_pair_count < pair_floor:
        return False

    # 7b. Level-by-level priority check
    # For each hand sub-play, determine how many of its cards were
    # played.
    # Then check that from highest pair_count level to lowest, the
    # played set
    # uses pairs from higher-level sub-plays first.

    played_card_ids = {c.id for c in played_suit_cards}

    # For each hand sub-play, count pairs played from it
    # pair_count=0 (single): 1 card played if its ID is in played set
    # pair_count=1 (pair): pair_count played if both cards' IDs are in
    # played set
    # pair_count>=2 (tractor): count how many of its ranks have both
    # cards played

    # Group hand sub-plays by pair_count level, counting how many pairs
    # from each sub-play are present in the played set
    hand_pairs_by_level: dict[
        int, int
    ] = {}  # pair_count_level -> total pairs available in hand
    played_pairs_from_hand_by_level: dict[
        int, int
    ] = {}  # pair_count_level -> pairs actually played from hand subs

    for sub in hand_subs:
        pc = sub.pair_count
        hand_pairs_by_level[pc] = hand_pairs_by_level.get(pc, 0) + pc

        # Count how many cards from this sub-play were played
        cards_played = sum(
            1 for c in sub.cards if c.id in played_card_ids
        )

        if pc == 0:
            # Single: 1 card = 1 pair at level 0
            if cards_played == 1:
                played_pairs_from_hand_by_level[pc] = (
                    played_pairs_from_hand_by_level.get(pc, 0) + 1
                )
        elif pc == 1:
            # Pair: 2 cards played = 1 pair at level 1
            if cards_played == 2:
                played_pairs_from_hand_by_level[pc] = (
                    played_pairs_from_hand_by_level.get(pc, 0) + 1
                )
        else:
            # Tractor: pc pairs. Count how many ranks have both cards
            # played
            rank_played_count: dict[Rank, int] = {}
            for c in sub.cards:
                if c.id in played_card_ids:
                    rank_played_count[c.rank] = (
                        rank_played_count.get(c.rank, 0) + 1
                    )
            pairs_from_tractor = sum(
                1 for count in rank_played_count.values() if count >= 2
            )
            played_pairs_from_hand_by_level[pc] = (
                played_pairs_from_hand_by_level.get(pc, 0)
                + pairs_from_tractor
            )

    # remaining_needed starts at total pairs in lead
    remaining_needed = lead_pair_count

    # Process from highest level to lowest
    all_levels = sorted(
        set(
            list(hand_pairs_by_level.keys())
            + list(played_pairs_from_hand_by_level.keys())
        ),
        reverse=True,
    )

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
                rank_played_count[c.rank] = (
                    rank_played_count.get(c.rank, 0) + 1
                )
        pairs_played = sum(
            1 for count in rank_played_count.values() if count >= 2
        )

        if pairs_played == 0 or pairs_played == sub.pair_count:
            # Not extracted at all, or fully extracted -- both fine
            continue

        # Partial extraction: check contiguity
        # Get unique ranks that were played as pairs
        unique_ranks = list(dict.fromkeys(c.rank for c in sub.cards))
        played_ranks = [
            r for r in unique_ranks if rank_played_count.get(r, 0) >= 2
        ]

        if not played_ranks:
            continue

        # Get positions of the extracted ranks within the tractor's
        # sorted rank list
        is_trump_group = lead_eff == "trump"
        if is_trump_group:
            sorted_ranks = sorted(
                unique_ranks,
                key=lambda rr: rank_order_in_trump_group(rr),
            )
        else:
            sorted_ranks = sorted(
                unique_ranks,
                key=lambda rr: non_trump_rank_order(rr, trump_rank),
            )

        positions = [sorted_ranks.index(r) for r in played_ranks]
        positions.sort()

        # Check contiguity: positions must form a range [min, max] with
        # no gaps
        if positions[-1] - positions[0] != len(positions) - 1:
            return False

    return True
