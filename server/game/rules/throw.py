"""Leading throw detection and resolution rules."""

from __future__ import annotations

from dataclasses import dataclass

from server.foundation.result import Ok, Rejected

from .cards import Card, Rank, Suit
from .decompose import (
    decompose,
    extract_tractor_pairs,
    rank_order_for_effective_suit,
)
from .ordering import effective_suit, trump_rank_order
from .rejections.card import CardsNotInHandRejected
from .rejections.play import EmptyPlayRejected, MixedLeadSuitRejected
from .types import EffectiveSuit, SubPlay


@dataclass(frozen=True, slots=True)
class LeadThrowResolution:
    """
    Resolved leading throw: attempted cards and actual played cards.
    """

    attempted_cards: list[Card]
    played_cards: list[Card]


def _group_cards_by_effective_suit(
    hand: list[Card], trump_suit: Suit | None, trump_rank: Rank
) -> dict[EffectiveSuit, list[Card]]:
    """Partition cards by effective suit."""
    groups: dict[EffectiveSuit, list[Card]] = {}
    for card in hand:
        eff = effective_suit(card, trump_suit, trump_rank)
        groups.setdefault(eff, []).append(card)
    return groups


def detect_throws(
    hand: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
    other_players_hands: list[list[Card]],
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
    groups = _group_cards_by_effective_suit(
        hand, trump_suit, trump_rank
    )

    result: list[list[Card]] = []
    for _eff_suit, cards in groups.items():
        # A throw requires 2+ cards in the same effective suit
        if len(cards) < 2:
            continue

        subs = decompose(cards, trump_suit, trump_rank)

        # Verify each sub-play is biggest (from low to high level per
        # spec 7.3)
        sorted_subs = sorted(subs, key=lambda s: s.sub_level)
        all_biggest = all(
            _is_biggest(
                sub, other_players_hands, trump_suit, trump_rank
            )
            for sub in sorted_subs
        )

        if all_biggest:
            result.append(cards)

    return result


def _is_biggest(
    sub: SubPlay,
    other_players_hands: list[list[Card]],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> bool:
    """
    Check if a sub-play is the biggest of its type in its effective
    suit.

    For single (pair_count=0): no other card of same effective suit has
    higher
    RANK_ORDER (or higher trump_rank_order for trump group).
    For pair (pair_count=1): no other pair of same effective suit has
    higher RANK_ORDER.
    For tractor (pair_count>=2): no other tractor of same effective suit
    with
    same or greater length beats it.
    """
    eff = sub.suit
    cards = sub.cards
    pair_count = sub.pair_count

    if pair_count == 0:
        # Single: check if any other card has higher rank
        sub_order = _card_order_for_effective_suit(
            cards[0],
            eff,
            trump_suit,
            trump_rank,
        )
        for player_hand in other_players_hands:
            others = [
                c
                for c in player_hand
                if effective_suit(c, trump_suit, trump_rank) == eff
            ]
            for c in others:
                other_order = _card_order_for_effective_suit(
                    c,
                    eff,
                    trump_suit,
                    trump_rank,
                )
                if other_order > sub_order:
                    return False
        return True

    if pair_count == 1:
        # Pair: check if any other pair has higher rank
        sub_order = _card_order_for_effective_suit(
            cards[0],
            eff,
            trump_suit,
            trump_rank,
        )

        for player_hand in other_players_hands:
            others = [
                c
                for c in player_hand
                if effective_suit(c, trump_suit, trump_rank) == eff
            ]
            for pair_cards in _player_pair_groups(others):
                other_order = _card_order_for_effective_suit(
                    pair_cards[0],
                    eff,
                    trump_suit,
                    trump_rank,
                )
                if other_order > sub_order:
                    return False
        return True

    # pair_count >= 2: tractor
    sub_max_order = max(
        _card_order_for_effective_suit(c, eff, trump_suit, trump_rank)
        for c in cards
    )
    for player_hand in other_players_hands:
        others = [
            c
            for c in player_hand
            if effective_suit(c, trump_suit, trump_rank) == eff
        ]
        other_subs = (
            decompose(others, trump_suit, trump_rank) if others else []
        )
        other_tractors = [
            s for s in other_subs if s.pair_count >= pair_count
        ]

        for ot in other_tractors:
            for candidate_cards in extract_tractor_pairs(
                ot, pair_count
            ):
                ot_max_order = max(
                    _card_order_for_effective_suit(
                        c,
                        eff,
                        trump_suit,
                        trump_rank,
                    )
                    for c in candidate_cards
                )
                if ot_max_order > sub_max_order:
                    return False

    return True


def _card_order_for_effective_suit(
    card: Card,
    suit: EffectiveSuit,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> int:
    if suit == "trump":
        return trump_rank_order(card, trump_suit, trump_rank)
    return rank_order_for_effective_suit(
        card.rank, suit, trump_suit, trump_rank
    )


def _player_pair_groups(cards: list[Card]) -> list[list[Card]]:
    groups: dict[tuple[Rank, Suit], list[Card]] = {}
    for card in cards:
        groups.setdefault((card.rank, card.suit), []).append(card)

    pairs: list[list[Card]] = []
    for group_cards in groups.values():
        pair_count = len(group_cards) // 2
        for i in range(pair_count):
            pairs.append(group_cards[i * 2 : i * 2 + 2])
    return pairs


def _throw_verification_order(
    sub: SubPlay,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> tuple[int, int]:
    """Return low-to-high throw verification order for one sub-play."""
    if not sub.cards:
        return (sub.sub_level, 0)
    rank_order = max(
        rank_order_for_effective_suit(
            c.rank, sub.suit, trump_suit, trump_rank
        )
        for c in sub.cards
    )
    return (sub.sub_level, rank_order)


def _sorted_throw_subplays(
    subs: list[SubPlay],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> list[SubPlay]:
    """
    Sort throw sub-plays from the smallest required fallback upward.
    """
    return sorted(
        subs,
        key=lambda sub: _throw_verification_order(
            sub, trump_suit, trump_rank
        ),
    )


def resolve_lead_throw(
    hand: list[Card],
    attempted_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
    other_players_hands: list[list[Card]],
) -> Ok[LeadThrowResolution] | Rejected:
    """Resolve any leading play as a throw attempt.

    If every sub-play can be thrown, all attempted cards are played. If
    a
    sub-play cannot be thrown, the smallest failing sub-play is forced.
    """
    if not attempted_cards:
        return EmptyPlayRejected()

    hand_ids = {c.id for c in hand}
    for c in attempted_cards:
        if c.id not in hand_ids:
            return CardsNotInHandRejected()

    eff_suits: set[EffectiveSuit] = {
        effective_suit(c, trump_suit, trump_rank)
        for c in attempted_cards
    }
    if len(eff_suits) != 1:
        return MixedLeadSuitRejected(eff_suits)

    subs = decompose(attempted_cards, trump_suit, trump_rank)
    for sub in _sorted_throw_subplays(subs, trump_suit, trump_rank):
        if not _is_biggest(
            sub, other_players_hands, trump_suit, trump_rank
        ):
            return Ok(
                LeadThrowResolution(
                    attempted_cards=list(attempted_cards),
                    played_cards=list(sub.cards),
                )
            )

    return Ok(
        LeadThrowResolution(
            attempted_cards=list(attempted_cards),
            played_cards=list(attempted_cards),
        )
    )
