"""Tests for engine.card module."""
import pytest
from pydantic import ValidationError
from server.engine.card import Card, Suit, Rank, create_decks, card_display


class TestCreateDecks:
    def test_create_decks_count(self):
        decks = create_decks()
        assert len(decks) == 108

    def test_create_decks_structure(self):
        decks = create_decks()
        deck1 = [c for c in decks if c.deck == 1]
        deck2 = [c for c in decks if c.deck == 2]
        assert len(deck1) == 54
        assert len(deck2) == 54

        deck1_suited = [c for c in deck1 if not c.is_joker]
        deck1_jokers = [c for c in deck1 if c.is_joker]
        assert len(deck1_suited) == 52
        assert len(deck1_jokers) == 2

    def test_create_decks_unique_ids(self):
        decks = create_decks()
        ids = [c.id for c in decks]
        assert len(set(ids)) == 108


class TestCardPoints:
    def test_card_points_five(self):
        decks = create_decks()
        fives = [c for c in decks if c.rank == Rank.FIVE]
        assert len(fives) == 8
        for c in fives:
            assert c.points == 5

    def test_card_points_ten(self):
        decks = create_decks()
        tens = [c for c in decks if c.rank == Rank.TEN]
        for c in tens:
            assert c.points == 10

    def test_card_points_king(self):
        decks = create_decks()
        kings = [c for c in decks if c.rank == Rank.KING]
        for c in kings:
            assert c.points == 10

    def test_card_points_others_zero(self):
        decks = create_decks()
        non_scoring = [c for c in decks if c.rank not in (Rank.FIVE, Rank.TEN, Rank.KING)]
        for c in non_scoring:
            assert c.points == 0


class TestCardDisplay:
    def test_card_display_big_joker(self):
        c = Card(id="D1-joker-BJ", suit=Suit.JOKER, rank=Rank.BIG_JOKER,
                 is_joker=True, is_big_joker=True, points=0, deck=1)
        assert card_display(c) == "🃏大"

    def test_card_display_small_joker(self):
        c = Card(id="D1-joker-SJ", suit=Suit.JOKER, rank=Rank.SMALL_JOKER,
                 is_joker=True, is_big_joker=False, points=0, deck=1)
        assert card_display(c) == "🃏小"

    def test_card_display_suited(self):
        c = Card(id="D1-hearts-A", suit=Suit.HEARTS, rank=Rank.ACE,
                 is_joker=False, is_big_joker=False, points=0, deck=1)
        assert card_display(c) == "♥A"


class TestCardValidation:
    def test_reject_big_joker_without_is_joker(self):
        with pytest.raises(ValidationError):
            Card(id="D1-hearts-A", suit=Suit.HEARTS, rank=Rank.ACE,
                 is_joker=False, is_big_joker=True, points=0, deck=1)

    def test_reject_joker_suit_without_is_joker(self):
        with pytest.raises(ValidationError):
            Card(id="D1-joker-BJ", suit=Suit.JOKER, rank=Rank.BIG_JOKER,
                 is_joker=False, is_big_joker=True, points=0, deck=1)

    def test_reject_non_joker_suit_with_is_joker(self):
        with pytest.raises(ValidationError):
            Card(id="D1-hearts-A", suit=Suit.HEARTS, rank=Rank.ACE,
                 is_joker=True, is_big_joker=False, points=0, deck=1)
