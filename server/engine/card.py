"""Card data model and deck creation for 升级 (Shengji/Tractor) card game.

Uses 2 standard 54-card decks = 108 cards total:
  2 × (4 suits × 13 ranks + 2 jokers) = 104 suited + 4 jokers

Point cards: 5 (5pts), 10 (10pts), K (10pts)
Total points in game: 200
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic.alias_generators import to_camel


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

_POINTS_MAP: dict[Rank, int] = {
    Rank.TWO: 0,
    Rank.THREE: 0,
    Rank.FOUR: 0,
    Rank.FIVE: 5,
    Rank.SIX: 0,
    Rank.SEVEN: 0,
    Rank.EIGHT: 0,
    Rank.NINE: 0,
    Rank.TEN: 10,
    Rank.JACK: 0,
    Rank.QUEEN: 0,
    Rank.KING: 10,
    Rank.ACE: 0,
    Rank.SMALL_JOKER: 0,
    Rank.BIG_JOKER: 0,
}

_SUITED_RANKS: list[Rank] = [
    Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE,
    Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE,
    Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE,
]

_SUITS: list[Suit] = [Suit.HEARTS, Suit.SPADES, Suit.DIAMONDS, Suit.CLUBS]

_JOKER_RANKS: tuple[Rank, ...] = (Rank.SMALL_JOKER, Rank.BIG_JOKER)

_SUIT_SYMBOLS: dict[Suit, str] = {
    Suit.HEARTS: "♥",
    Suit.SPADES: "♠",
    Suit.DIAMONDS: "♦",
    Suit.CLUBS: "♣",
    Suit.JOKER: "\U0001f0cf",
}

_RANK_DISPLAY: dict[Rank, str] = {
    Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5",
    Rank.SIX: "6", Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9",
    Rank.TEN: "10", Rank.JACK: "J", Rank.QUEEN: "Q", Rank.KING: "K",
    Rank.ACE: "A", Rank.SMALL_JOKER: "小", Rank.BIG_JOKER: "大",
}


# ---- Card Model ----


class Card(BaseModel):
    """A single playing card with unique ID, suit, rank, and point value."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, frozen=True)

    id: str
    suit: Suit
    rank: Rank
    is_joker: bool
    is_big_joker: bool
    points: int
    deck: Literal[1, 2]

    @model_validator(mode="after")
    def _validate_joker_consistency(self) -> "Card":
        if self.is_big_joker and not self.is_joker:
            raise ValueError("is_big_joker can only be True when is_joker is True")
        if self.suit == Suit.JOKER and not self.is_joker:
            raise ValueError("suit=JOKER requires is_joker=True")
        if self.is_joker and self.suit != Suit.JOKER:
            raise ValueError("is_joker=True requires suit=JOKER")
        if self.is_joker and self.rank not in _JOKER_RANKS:
            raise ValueError("is_joker=True requires rank to be SMALL_JOKER or BIG_JOKER")
        if not self.is_joker and self.rank in _JOKER_RANKS:
            raise ValueError("Joker ranks (SMALL_JOKER, BIG_JOKER) require is_joker=True")
        if self.is_big_joker and self.rank != Rank.BIG_JOKER:
            raise ValueError("is_big_joker=True requires rank to be BIG_JOKER")
        if self.rank == Rank.BIG_JOKER and not self.is_big_joker:
            raise ValueError("rank=BIG_JOKER requires is_big_joker=True")
        return self


# ---- Factory ----


def _card_id(deck: int, suit: Suit, rank: Rank) -> str:
    return f"D{deck}-{suit.value}-{rank.value}"


def _make_card(suit: Suit, rank: Rank, deck: Literal[1, 2]) -> Card:
    if suit == Suit.JOKER:
        return Card(
            id=_card_id(deck, suit, rank),
            suit=Suit.JOKER,
            rank=rank,
            is_joker=True,
            is_big_joker=(rank == Rank.BIG_JOKER),
            points=0,
            deck=deck,
        )
    return Card(
        id=_card_id(deck, suit, rank),
        suit=suit,
        rank=rank,
        is_joker=False,
        is_big_joker=False,
        points=_POINTS_MAP[rank],
        deck=deck,
    )


def create_decks() -> list[Card]:
    """Create 2 full 54-card decks = 108 cards."""
    cards: list[Card] = []
    for deck in (1, 2):
        for suit in _SUITS:
            for rank in _SUITED_RANKS:
                cards.append(_make_card(suit, rank, deck))
        cards.append(_make_card(Suit.JOKER, Rank.SMALL_JOKER, deck))
        cards.append(_make_card(Suit.JOKER, Rank.BIG_JOKER, deck))
    return cards


# ---- Helpers ----


def card_display(card: Card) -> str:
    """Return a human-readable display string for a card.

    Examples: "♥A", "♠10", "🃏大", "🃏小"
    """
    if card.is_joker:
        return f"{_SUIT_SYMBOLS[Suit.JOKER]}{_RANK_DISPLAY[card.rank]}"
    return f"{_SUIT_SYMBOLS[card.suit]}{_RANK_DISPLAY[card.rank]}"
