"""Card data model and deck creation for 升级 (Shengji/Tractor) card game.

Uses 2 standard 54-card decks = 108 cards total:
  2 x (4 suits x 13 ranks + 2 jokers) = 104 suited + 4 jokers

Point cards: 5 (5pts), 10 (10pts), K (10pts)
Total points in game: 200
"""

from enum import Enum
from typing import Literal, Self, overload

from pydantic import BaseModel, ConfigDict, model_validator


# ---- Enums ----


class Suit(str, Enum):
    HEARTS = "hearts"
    SPADES = "spades"
    DIAMONDS = "diamonds"
    CLUBS = "clubs"
    JOKER = "joker"


class Rank(str, Enum):
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"
    SMALL_JOKER = "SJ"
    BIG_JOKER = "BJ"


# ---- Constants ----

_JOKER_RANKS: tuple[Rank, ...] = (Rank.SMALL_JOKER, Rank.BIG_JOKER)

SUITED_RANKS: tuple[Rank, ...] = (
    Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE,
    Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE,
    Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE,
)

_SUIT_SYMBOLS: dict[Suit, str] = {
    Suit.HEARTS: "♥",
    Suit.SPADES: "♠",
    Suit.DIAMONDS: "♦",
    Suit.CLUBS: "♣",
}

_RANK_DISPLAY: dict[Rank, str] = {
    Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5",
    Rank.SIX: "6", Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9",
    Rank.TEN: "10", Rank.JACK: "J", Rank.QUEEN: "Q", Rank.KING: "K",
    Rank.ACE: "A", Rank.SMALL_JOKER: "小", Rank.BIG_JOKER: "大",
}

POINTS_MAP: dict[Rank, int] = {
    Rank.TWO: 0, Rank.THREE: 0, Rank.FOUR: 0, Rank.FIVE: 5,
    Rank.SIX: 0, Rank.SEVEN: 0, Rank.EIGHT: 0, Rank.NINE: 0,
    Rank.TEN: 10, Rank.JACK: 0, Rank.QUEEN: 0, Rank.KING: 10,
    Rank.ACE: 0, Rank.SMALL_JOKER: 0, Rank.BIG_JOKER: 0,
}


# ---- Card Model ----


type CardKey = Literal["id", "suit", "rank", "points"]


class Card(BaseModel):
    """A single playing card visible to rules and players."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    suit: Suit
    rank: Rank
    points: int

    @model_validator(mode="after")
    def _validate_card_consistency(self) -> Self:
        id_deck, id_suit, id_rank = _parse_card_id(self.id)
        if id_deck != self.deck:
            raise ValueError("card id deck does not match parsed deck")
        if id_suit != self.suit:
            raise ValueError("card id suit does not match suit")
        if id_rank != self.rank:
            raise ValueError("card id rank does not match rank")
        if self.suit == Suit.JOKER and self.rank not in _JOKER_RANKS:
            raise ValueError("suit=JOKER requires rank to be SMALL_JOKER or BIG_JOKER")
        if self.suit != Suit.JOKER and self.rank in _JOKER_RANKS:
            raise ValueError("Joker ranks require suit=JOKER")
        if self.points != POINTS_MAP[self.rank]:
            raise ValueError("points must equal POINTS_MAP[rank]")
        return self

    @property
    def is_joker(self) -> bool:
        return self.suit == Suit.JOKER

    @property
    def is_big_joker(self) -> bool:
        return self.rank == Rank.BIG_JOKER

    @property
    def deck(self) -> Literal[1, 2]:
        return _parse_card_id(self.id)[0]

    @overload
    def __getitem__(self, key: Literal["id"]) -> str: ...

    @overload
    def __getitem__(self, key: Literal["suit"]) -> Suit: ...

    @overload
    def __getitem__(self, key: Literal["rank"]) -> Rank: ...

    @overload
    def __getitem__(self, key: Literal["points"]) -> int: ...

    def __getitem__(self, key: CardKey) -> str | Suit | Rank | int:
        if key == "id":
            return self.id
        if key == "suit":
            return self.suit
        if key == "rank":
            return self.rank
        return self.points

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in type(self).model_fields


# ---- Factory ----


def _card_id(deck: Literal[1, 2], suit: Suit, rank: Rank) -> str:
    return f"D{deck}-{suit.value}-{rank.value}"


def _parse_card_id(card_id: str) -> tuple[Literal[1, 2], Suit, Rank]:
    parts = card_id.split("-")
    if len(parts) != 3:
        raise ValueError("card id must have format D{deck}-{suit}-{rank}")
    deck_raw, suit_raw, rank_raw = parts
    if deck_raw == "D1":
        deck: Literal[1, 2] = 1
    elif deck_raw == "D2":
        deck = 2
    else:
        raise ValueError("card id must start with D1- or D2-")
    try:
        suit = Suit(suit_raw)
    except ValueError as exc:
        raise ValueError("card id has invalid suit") from exc
    try:
        rank = Rank(rank_raw)
    except ValueError as exc:
        raise ValueError("card id has invalid rank") from exc
    return deck, suit, rank


def _make_card(suit: Suit, rank: Rank, deck: Literal[1, 2]) -> Card:
    if suit == Suit.JOKER:
        return Card(
            id=_card_id(deck, suit, rank),
            suit=Suit.JOKER,
            rank=rank,
            points=0,
        )
    return Card(
        id=_card_id(deck, suit, rank),
        suit=suit,
        rank=rank,
        points=POINTS_MAP[rank],
    )


def create_decks() -> list[Card]:
    """Create 2 full 54-card decks = 108 cards."""
    cards: list[Card] = []
    for deck in (1, 2):
        for suit in (Suit.HEARTS, Suit.SPADES, Suit.DIAMONDS, Suit.CLUBS):
            for rank in SUITED_RANKS:
                cards.append(_make_card(suit, rank, deck))
        cards.append(_make_card(Suit.JOKER, Rank.SMALL_JOKER, deck))
        cards.append(_make_card(Suit.JOKER, Rank.BIG_JOKER, deck))
    return cards


# ---- Helpers ----


def card_display(card: Card) -> str:
    """Return a human-readable display string for a card.

    Examples: "♥A", "♠10", "大王", "小王"
    """
    if card.is_joker:
        return f"{_RANK_DISPLAY[card.rank]}王"
    return f"{_SUIT_SYMBOLS[card.suit]}{_RANK_DISPLAY[card.rank]}"
