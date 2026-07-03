"""Semantic card-face records shared by rules and training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from server.result import Ok, Rejected
from server.rules.cards import POINTS_MAP, Card, Rank, Suit

type CardColor = Literal["red", "black", "none"]

SUIT_ORDER: tuple[Suit, ...] = (
    Suit.HEARTS,
    Suit.SPADES,
    Suit.DIAMONDS,
    Suit.CLUBS,
    Suit.JOKER,
)
RANK_ORDER: tuple[Rank, ...] = (
    Rank.TWO,
    Rank.THREE,
    Rank.FOUR,
    Rank.FIVE,
    Rank.SIX,
    Rank.SEVEN,
    Rank.EIGHT,
    Rank.NINE,
    Rank.TEN,
    Rank.JACK,
    Rank.QUEEN,
    Rank.KING,
    Rank.ACE,
    Rank.SMALL_JOKER,
    Rank.BIG_JOKER,
)
MAX_FACE_COUNT: int = 2
FACE_COUNT_CHOICES: tuple[int, ...] = (1, 2)


class FaceCountRejected(Rejected):
    """Semantic card-face selection is not available in a hand."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class CardFace:
    """A card's strategic identity without deck-copy identity."""

    suit: Suit
    rank: Rank

    @property
    def points(self) -> int:
        return POINTS_MAP[self.rank]

    @property
    def color(self) -> CardColor:
        if self.suit in (Suit.HEARTS, Suit.DIAMONDS):
            return "red"
        if self.suit in (Suit.SPADES, Suit.CLUBS):
            return "black"
        return "none"


@dataclass(frozen=True, slots=True)
class FaceCount:
    """A count of identical semantic card faces."""

    face: CardFace
    count: int

    def __post_init__(self) -> None:
        assert self.count in FACE_COUNT_CHOICES


def card_face(card: Card) -> CardFace:
    """Return the semantic face of a physical card."""
    return CardFace(suit=card.suit, rank=card.rank)


def canonical_face_counts(
    cards: list[Card] | tuple[Card, ...],
) -> tuple[FaceCount, ...]:
    """Group physical cards by semantic face in canonical order."""
    counts: dict[CardFace, int] = {}
    for card in cards:
        face = card_face(card)
        counts[face] = counts.get(face, 0) + 1
    return tuple(
        FaceCount(face=face, count=count)
        for face, count in sorted(
            counts.items(), key=lambda item: face_sort_key(item[0])
        )
    )


def face_sort_key(face: CardFace) -> tuple[int, int]:
    """Return the canonical semantic order key for a card face."""
    return (SUIT_ORDER.index(face.suit), RANK_ORDER.index(face.rank))


def face_count_width(face_counts: tuple[FaceCount, ...]) -> int:
    """Return the physical card count represented by face counts."""
    return sum(item.count for item in face_counts)


def face_count_signature(
    face_counts: tuple[FaceCount, ...],
) -> tuple[tuple[str, str, int], ...]:
    """Return a stable id-free signature for history de-duplication."""
    return tuple(
        (item.face.suit.value, item.face.rank.value, item.count)
        for item in face_counts
    )


def bind_face_counts(
    face_counts: tuple[FaceCount, ...],
    hand_cards: list[Card] | tuple[Card, ...],
) -> Ok[list[Card]] | Rejected:
    """Bind semantic face counts to concrete physical cards in hand."""
    result: list[Card] = []
    used_ids: set[str] = set()
    for requested in face_counts:
        matching = [
            card
            for card in hand_cards
            if card.id not in used_ids
            and card.suit == requested.face.suit
            and card.rank == requested.face.rank
        ]
        if len(matching) < requested.count:
            return FaceCountRejected("当前手牌没有足够的指定牌面")
        selected = matching[: requested.count]
        result.extend(selected)
        used_ids.update(card.id for card in selected)
    return Ok(value=result)
