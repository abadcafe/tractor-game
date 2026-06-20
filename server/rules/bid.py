"""Bid declaration rules shared by deal-bid state and players."""

from __future__ import annotations

from typing import Literal

from server.result import Ok, Rejected

from .cards import Card, Rank, Suit
from .ordering import bid_value
from .rejections import (
    BidCardsCountMismatchRejected,
    BidCardSuitMismatchRejected,
    BidCardWrongRankRejected,
    BidCountRejected,
    BidPriorityTooLowRejected,
    DuplicateBidCardsRejected,
    JokerBidCountRejected,
    JokerBidMustBePairRejected,
    JokerBidSuitRejected,
    MissingBidSuitRejected,
    MixedJokerPairRejected,
    NotJokerRejected,
    ZeroBidValueRejected,
)

type BidKind = Literal["trump_rank", "joker"]

_BID_SUIT_ORDER: tuple[Suit, ...] = (
    Suit.SPADES,
    Suit.HEARTS,
    Suit.CLUBS,
    Suit.DIAMONDS,
)
MAX_BID_ACTION_HINTS: int = 10


def bid_card_candidates(hand: list[Card], trump_rank: Rank) -> list[list[Card]]:
    """Compute logical bid card groups from a player's hand."""
    suit_groups: dict[Suit, list[Card]] = {}
    small_jokers: list[Card] = []
    big_jokers: list[Card] = []

    for card in hand:
        if card.is_joker:
            if card.is_big_joker:
                big_jokers.append(card)
            else:
                small_jokers.append(card)
        elif card.rank == trump_rank:
            suit_groups.setdefault(card.suit, []).append(card)

    result: list[list[Card]] = []

    if len(big_jokers) >= 2:
        result.append(big_jokers[:2])
    if len(small_jokers) >= 2:
        result.append(small_jokers[:2])
    for suit in _BID_SUIT_ORDER:
        cards = suit_groups.get(suit, [])
        if len(cards) >= 2:
            result.append(cards[:2])

    for suit in _BID_SUIT_ORDER:
        cards = suit_groups.get(suit, [])
        if len(cards) >= 1:
            result.append([cards[0]])

    return result


def bid_hint_sort_key(cards: list[Card], trump_rank: Rank) -> tuple[int, tuple[str, ...]]:
    return (bid_value(cards, trump_rank), tuple(sorted(card.id for card in cards)))


def sort_bid_action_hints(hints: list[list[Card]], trump_rank: Rank) -> list[list[Card]]:
    return sorted([list(cards) for cards in hints], key=lambda cards: bid_hint_sort_key(cards, trump_rank))


def legal_bid_hints(
    hand: list[Card],
    trump_rank: Rank,
    current_bid_cards: list[Card] | None,
) -> list[list[Card]]:
    """Return legal bid hints from a hand, ordered weakest to strongest."""
    result: list[list[Card]] = []
    for candidate in bid_card_candidates(hand, trump_rank):
        match bid_beats_current(candidate, current_bid_cards, trump_rank):
            case Ok():
                result.append(candidate)
            case Rejected():
                continue
    return sort_bid_action_hints(result, trump_rank)


def validate_distinct_bid_cards(cards: list[Card]) -> Ok[None] | Rejected:
    if len(set(card.id for card in cards)) != len(cards):
        return DuplicateBidCardsRejected()
    return Ok(None)


def validate_bid_cards(
    *,
    kind: BidKind,
    cards: list[Card],
    declared_suit: Suit | None,
    declared_count: int,
    trump_rank: Rank,
) -> Ok[None] | Rejected:
    if kind == "trump_rank":
        if declared_suit is None:
            return MissingBidSuitRejected()
        for card in cards:
            if card.rank != trump_rank:
                return BidCardWrongRankRejected(card.id, trump_rank)
            if card.suit != declared_suit:
                return BidCardSuitMismatchRejected(card.suit, declared_suit)
        if declared_count not in (1, 2):
            return BidCountRejected(declared_count)
        if len(cards) != declared_count:
            return BidCardsCountMismatchRejected(len(cards), declared_count)
        return Ok(None)

    if declared_count != 2:
        return JokerBidMustBePairRejected()
    if len(cards) != 2:
        return JokerBidCountRejected(len(cards))
    for card in cards:
        if not card.is_joker:
            return NotJokerRejected(card.id)
    if cards[0].rank != cards[1].rank:
        return MixedJokerPairRejected()
    if declared_suit is not None:
        return JokerBidSuitRejected()
    return Ok(None)


def bid_beats_current(
    cards: list[Card],
    current_bid_cards: list[Card] | None,
    trump_rank: Rank,
) -> Ok[None] | Rejected:
    new_value = bid_value(cards, trump_rank)
    if new_value == 0:
        return ZeroBidValueRejected()

    if current_bid_cards is not None:
        current_value = bid_value(current_bid_cards, trump_rank)
        if new_value <= current_value:
            return BidPriorityTooLowRejected()

    return Ok(None)
