"""Tests for sm.card_model module."""
import pytest
from .card_model import Card, Suit, Rank, create_decks, card_display


class TestCreateDecks:
    def test_create_decks_count(self) -> None:
        """2 decks = 108 cards total."""
        deck = create_decks()
        assert len(deck) == 108

    def test_create_decks_suit_distribution(self) -> None:
        """Each suit has 2 copies x 13 ranks = 26 cards per suit, 4 suits = 104."""
        deck = create_decks()
        for suit in (Suit.HEARTS, Suit.SPADES, Suit.DIAMONDS, Suit.CLUBS):
            count = sum(1 for c in deck if c.suit == suit)
            assert count == 26, f"{suit} has {count} cards, expected 26"

    def test_create_decks_joker_count(self) -> None:
        """4 jokers total: 2 small + 2 big."""
        deck = create_decks()
        small_jokers = [c for c in deck if c.rank == Rank.SMALL_JOKER]
        big_jokers = [c for c in deck if c.rank == Rank.BIG_JOKER]
        assert len(small_jokers) == 2
        assert len(big_jokers) == 2

    def test_create_decks_unique_ids(self) -> None:
        """Every card has a unique id."""
        deck = create_decks()
        ids = [c.id for c in deck]
        assert len(set(ids)) == 108


class TestCardModel:
    def test_card_frozen(self) -> None:
        """Card is immutable."""
        card = Card(
            id="D1-hearts-A", suit=Suit.HEARTS, rank=Rank.ACE,
            is_joker=False, is_big_joker=False, points=0, deck=1,
        )
        with pytest.raises(Exception):
            setattr(card, "points", 5)

    def test_card_joker_validation_big_joker_requires_is_joker(self) -> None:
        """is_big_joker=True requires is_joker=True."""
        with pytest.raises(Exception):
            Card(
                id="D1-hearts-A", suit=Suit.HEARTS, rank=Rank.ACE,
                is_joker=False, is_big_joker=True, points=0, deck=1,
            )

    def test_card_joker_validation_joker_requires_suit(self) -> None:
        """is_joker=True requires suit=JOKER."""
        with pytest.raises(Exception):
            Card(
                id="D1-hearts-SJ", suit=Suit.HEARTS, rank=Rank.SMALL_JOKER,
                is_joker=True, is_big_joker=False, points=0, deck=1,
            )

    def test_card_id_format(self) -> None:
        """Card id follows D{deck}-{suit}-{rank} format."""
        card = Card(
            id="D2-spades-5", suit=Suit.SPADES, rank=Rank.FIVE,
            is_joker=False, is_big_joker=False, points=5, deck=2,
        )
        assert card.id == "D2-spades-5"


class TestCardPoints:
    def test_card_points_five(self) -> None:
        """Rank 5 = 5 points."""
        card = Card(
            id="D1-hearts-5", suit=Suit.HEARTS, rank=Rank.FIVE,
            is_joker=False, is_big_joker=False, points=5, deck=1,
        )
        assert card.points == 5

    def test_card_points_ten(self) -> None:
        """Rank 10 = 10 points."""
        card = Card(
            id="D1-hearts-10", suit=Suit.HEARTS, rank=Rank.TEN,
            is_joker=False, is_big_joker=False, points=10, deck=1,
        )
        assert card.points == 10

    def test_card_points_king(self) -> None:
        """Rank K = 10 points."""
        card = Card(
            id="D1-hearts-K", suit=Suit.HEARTS, rank=Rank.KING,
            is_joker=False, is_big_joker=False, points=10, deck=1,
        )
        assert card.points == 10

    def test_card_points_non_scoring(self) -> None:
        """Non-scoring ranks = 0 points."""
        card = Card(
            id="D1-hearts-7", suit=Suit.HEARTS, rank=Rank.SEVEN,
            is_joker=False, is_big_joker=False, points=0, deck=1,
        )
        assert card.points == 0

    def test_card_points_joker_zero(self) -> None:
        """Jokers have 0 points."""
        card = Card(
            id="D1-joker-BJ", suit=Suit.JOKER, rank=Rank.BIG_JOKER,
            is_joker=True, is_big_joker=True, points=0, deck=1,
        )
        assert card.points == 0


class TestCardDisplay:
    def test_card_display_suit(self) -> None:
        """Suited card displays as {symbol}{rank}."""
        card = Card(
            id="D1-hearts-A", suit=Suit.HEARTS, rank=Rank.ACE,
            is_joker=False, is_big_joker=False, points=0, deck=1,
        )
        assert card_display(card) == "♥A"

    def test_card_display_joker_big(self) -> None:
        """Big joker displays as \U0001f0cf大."""
        card = Card(
            id="D1-joker-BJ", suit=Suit.JOKER, rank=Rank.BIG_JOKER,
            is_joker=True, is_big_joker=True, points=0, deck=1,
        )
        assert card_display(card) == "\U0001f0cf大"

    def test_card_display_joker_small(self) -> None:
        """Small joker displays as \U0001f0cf小."""
        card = Card(
            id="D1-joker-SJ", suit=Suit.JOKER, rank=Rank.SMALL_JOKER,
            is_joker=True, is_big_joker=False, points=0, deck=1,
        )
        assert card_display(card) == "\U0001f0cf小"


class TestEnums:
    def test_suit_enum_values(self) -> None:
        """Suit has exactly 5 values including JOKER."""
        assert len(Suit) == 5
        assert Suit.HEARTS.value == "hearts"
        assert Suit.SPADES.value == "spades"
        assert Suit.DIAMONDS.value == "diamonds"
        assert Suit.CLUBS.value == "clubs"
        assert Suit.JOKER.value == "joker"

    def test_rank_enum_values(self) -> None:
        """Rank has 13 suited ranks + 2 joker ranks = 15."""
        assert len(Rank) == 15

    def test_rank_joker_values(self) -> None:
        """Joker ranks are SMALL_JOKER and BIG_JOKER."""
        assert Rank.SMALL_JOKER.value == "SJ"
        assert Rank.BIG_JOKER.value == "BJ"
