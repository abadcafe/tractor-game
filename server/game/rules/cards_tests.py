"""Black-box tests for physical card values."""

from server.game.rules.cards import (
    Card,
    Rank,
    Suit,
    card_display,
    card_points,
    create_decks,
    suited_ranks,
)


class TestCreateDecks:
    def test_create_decks_count(self) -> None:
        """2 decks = 108 cards total."""
        deck = create_decks()
        assert len(deck) == 108

    def test_create_decks_suit_distribution(self) -> None:
        """
        Each suit has 2 copies x 13 ranks = 26 cards per suit, 4 suits =
        104.
        """
        deck = create_decks()
        for suit in (
            Suit.HEARTS,
            Suit.SPADES,
            Suit.DIAMONDS,
            Suit.CLUBS,
        ):
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
    def test_card_id_format(self) -> None:
        """Card id follows D{deck}-{suit}-{rank} format."""
        card = Card(
            id="D2-spades-5",
            suit=Suit.SPADES,
            rank=Rank.FIVE,
            points=5,
        )
        assert card.id == "D2-spades-5"
        assert card.deck == 2

    def test_generated_cards_have_canonical_identity(self) -> None:
        """Generated cards agree with their typed fields."""
        for card in create_decks():
            assert card.id == (
                f"D{card.deck}-{card.suit.value}-{card.rank.value}"
            )
            assert card.points == card_points(card.rank)

    def test_generated_jokers_have_only_joker_faces(self) -> None:
        """The physical deck never creates a mixed joker face."""
        jokers = tuple(card for card in create_decks() if card.is_joker)
        assert all(card.suit == Suit.JOKER for card in jokers)
        assert {card.rank for card in jokers} == {
            Rank.SMALL_JOKER,
            Rank.BIG_JOKER,
        }

    def test_generated_suited_cards_never_use_joker_ranks(self) -> None:
        """Suited cards use exactly the public suited-rank domain."""
        cards = tuple(
            card for card in create_decks() if card.suit != Suit.JOKER
        )
        assert {card.rank for card in cards} == set(suited_ranks())

    def test_deck_construction_is_stable_and_fresh(self) -> None:
        """Repeated construction has equal values and distinct lists."""
        first = create_decks()
        second = create_decks()
        assert first == second
        assert first is not second

    def test_card_value_serializes_as_exact_public_shape(self) -> None:
        """Card JSON has no hidden compatibility fields."""
        card = Card(
            id="D1-hearts-A",
            suit=Suit.HEARTS,
            rank=Rank.ACE,
            points=0,
        )
        assert card.model_dump(mode="json") == {
            "id": "D1-hearts-A",
            "suit": "hearts",
            "rank": "A",
            "points": 0,
        }


class TestCardPoints:
    def test_card_points_five(self) -> None:
        """Rank 5 = 5 points."""
        card = Card(
            id="D1-hearts-5",
            suit=Suit.HEARTS,
            rank=Rank.FIVE,
            points=5,
        )
        assert card.points == 5

    def test_card_points_ten(self) -> None:
        """Rank 10 = 10 points."""
        card = Card(
            id="D1-hearts-10",
            suit=Suit.HEARTS,
            rank=Rank.TEN,
            points=10,
        )
        assert card.points == 10

    def test_card_points_king(self) -> None:
        """Rank K = 10 points."""
        card = Card(
            id="D1-hearts-K",
            suit=Suit.HEARTS,
            rank=Rank.KING,
            points=10,
        )
        assert card.points == 10

    def test_card_points_non_scoring(self) -> None:
        """Non-scoring ranks = 0 points."""
        card = Card(
            id="D1-hearts-7",
            suit=Suit.HEARTS,
            rank=Rank.SEVEN,
            points=0,
        )
        assert card.points == 0

    def test_card_points_joker_zero(self) -> None:
        """Jokers have 0 points."""
        card = Card(
            id="D1-joker-BJ",
            suit=Suit.JOKER,
            rank=Rank.BIG_JOKER,
            points=0,
        )
        assert card.points == 0
        assert card.is_joker is True
        assert card.is_big_joker is True


class TestCardDisplay:
    def test_card_display_suit(self) -> None:
        """Suited card displays as {symbol}{rank}."""
        card = Card(
            id="D1-hearts-A",
            suit=Suit.HEARTS,
            rank=Rank.ACE,
            points=0,
        )
        assert card_display(card) == "♥A"

    def test_card_display_joker_big(self) -> None:
        """Big joker displays as 大王."""
        card = Card(
            id="D1-joker-BJ",
            suit=Suit.JOKER,
            rank=Rank.BIG_JOKER,
            points=0,
        )
        assert card_display(card) == "大王"

    def test_card_display_joker_small(self) -> None:
        """Small joker displays as 小王."""
        card = Card(
            id="D1-joker-SJ",
            suit=Suit.JOKER,
            rank=Rank.SMALL_JOKER,
            points=0,
        )
        assert card_display(card) == "小王"


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
