"""Declaration rules for bidding and stirring."""

from __future__ import annotations

from dataclasses import dataclass

from server.foundation.result import Ok, Rejected

from . import _bid
from ._ordering import bid_value as _bid_value
from .cards import Card, Rank, Suit

__all__ = [
    "Declaration",
    "is_maximum",
    "legal_reveals",
    "priority",
    "validate_reveal",
    "validate_stir",
]


@dataclass(frozen=True, slots=True)
class Declaration:
    """A public declaration made with physical cards."""

    cards: tuple[Card, ...]

    def __post_init__(self) -> None:
        assert self.cards

    @property
    def suit(self) -> Suit | None:
        first = self.cards[0]
        if first.is_joker:
            return None
        return first.suit


def legal_reveals(
    hand: tuple[Card, ...],
    trump_rank: Rank,
    current: Declaration | None,
) -> tuple[Declaration, ...]:
    """Return every currently legal reveal, weakest first."""
    current_cards = None if current is None else list(current.cards)
    return tuple(
        Declaration(cards=tuple(cards))
        for cards in _bid.legal_bid_hints(
            list(hand),
            trump_rank,
            current_cards,
        )
    )


def validate_reveal(
    cards: tuple[Card, ...],
    trump_rank: Rank,
    current: Declaration | None,
) -> Ok[Declaration] | Rejected:
    """Validate and normalize one bid reveal."""
    if not cards:
        return Rejected("亮牌不能为空")
    match _bid.validate_distinct_bid_cards(list(cards)):
        case Ok():
            pass
        case Rejected() as rejected:
            return rejected

    first = cards[0]
    kind: _bid.BidKind = "joker" if first.is_joker else "trump_rank"
    suit = None if first.is_joker else first.suit
    match _bid.validate_bid_cards(
        kind=kind,
        cards=list(cards),
        declared_suit=suit,
        declared_count=len(cards),
        trump_rank=trump_rank,
    ):
        case Ok():
            pass
        case Rejected() as rejected:
            return rejected

    current_cards = None if current is None else list(current.cards)
    match _bid.bid_beats_current(
        list(cards),
        current_cards,
        trump_rank,
    ):
        case Ok():
            return Ok(Declaration(cards=cards))
        case Rejected() as rejected:
            return rejected


def validate_stir(
    cards: tuple[Card, ...],
    trump_rank: Rank,
    current: Declaration | None,
) -> Ok[Declaration] | Rejected:
    """Validate a two-card stir against the current declaration."""
    if len(cards) != 2:
        return Rejected("炒牌必须恰好亮出两张牌")
    return validate_reveal(cards, trump_rank, current)


def priority(
    declaration: Declaration,
    trump_rank: Rank,
) -> int:
    """Return the declaration's strict comparison priority."""
    return _bid_value(list(declaration.cards), trump_rank)


def is_maximum(declaration: Declaration) -> bool:
    """Return whether no legal declaration can overcall this one."""
    return len(declaration.cards) == 2 and all(
        card.rank == Rank.BIG_JOKER for card in declaration.cards
    )
