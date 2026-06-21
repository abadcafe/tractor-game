"""Tests for sm.deal_bid_sm module."""

from typing import Literal

from server.result import Ok, Rejected
from server.rules.cards import (
    POINTS_MAP,
    Card,
    Rank,
    Suit,
    create_decks,
)

from .deal_bid_sm import (
    MAX_BID_ACTION_HINTS,
    DealBidInput,
    DealBidResult,
    DealBidState,
    create_deal_bid,
    deal_next_card,
    finalize_dealing,
    get_bid_action_hints,
    reveal,
)
from .types import BidEvent


def _card(suit: Suit, rank: Rank, deck: Literal[1, 2] = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit,
        rank=rank,
        points=POINTS_MAP[rank],
    )


def _deal(state: DealBidState) -> DealBidState:
    """Unwrap deal_next_card result, asserting Ok."""
    result = deal_next_card(state)
    assert isinstance(result, Ok), (
        f"deal_next_card rejected: {result.reason}"
    )
    return result.value


def _make_deck_with_specific_cards() -> tuple[list[Card], list[Card]]:
    """
    Create a deck and bottom cards for testing. Returns (deck_100,
    bottom_8).
    """
    all_cards = create_decks()
    import random

    random.seed(42)
    random.shuffle(all_cards)
    return all_cards[8:], all_cards[:8]


def _make_deterministic_deck() -> tuple[list[Card], list[Card]]:
    """
    Create a deterministic deck where specific trump-rank cards and
    jokers
    land in specific player hands.

    Layout (100-card deck, CCW order: 0->1->2->3->0):
      Position 0 (player 0): ♠TWO deck1
      Position 1 (player 1): ♥TWO deck1
      Position 2 (player 2): ♣TWO deck1
      Position 3 (player 3): ♦TWO deck1
      Position 4 (player 0): ♠TWO deck2   <- gives player 0 a ♠ pair
      Position 5 (player 1): ♥TWO deck2   <- gives player 1 a ♥ pair
      Position 6 (player 2): ♣TWO deck2   <- gives player 2 a ♣ pair
      Position 7 (player 3): ♦TWO deck2   <- gives player 3 a ♦ pair
      Positions 8-9 (player 0): big jokers  <- for joker pair tests
      Positions 10+: remaining non-trump-rank non-joker cards shuffled
    Bottom cards: taken from the end of the remaining pool.
    """
    all_cards = create_decks()
    big_jokers = [c for c in all_cards if c.rank == Rank.BIG_JOKER]
    small_jokers = [c for c in all_cards if c.rank == Rank.SMALL_JOKER]
    remaining = [
        c
        for c in all_cards
        if c.rank not in (Rank.BIG_JOKER, Rank.SMALL_JOKER)
    ]
    import random

    random.seed(99)
    random.shuffle(remaining)

    # Specific trump-rank cards for each player
    spade_two_1 = _card(Suit.SPADES, Rank.TWO, 1)
    spade_two_2 = _card(Suit.SPADES, Rank.TWO, 2)
    heart_two_1 = _card(Suit.HEARTS, Rank.TWO, 1)
    heart_two_2 = _card(Suit.HEARTS, Rank.TWO, 2)
    club_two_1 = _card(Suit.CLUBS, Rank.TWO, 1)
    club_two_2 = _card(Suit.CLUBS, Rank.TWO, 2)
    diamond_two_1 = _card(Suit.DIAMONDS, Rank.TWO, 1)
    diamond_two_2 = _card(Suit.DIAMONDS, Rank.TWO, 2)

    # Remove the specific cards from remaining pool (by id)
    specific_ids = {
        c.id
        for c in [
            spade_two_1,
            spade_two_2,
            heart_two_1,
            heart_two_2,
            club_two_1,
            club_two_2,
            diamond_two_1,
            diamond_two_2,
        ]
    }
    remaining = [c for c in remaining if c.id not in specific_ids]

    # Build the 100-card deck with known positions
    deck: list[Card] = [
        spade_two_1,  # pos 0 -> player 0
        heart_two_1,  # pos 1 -> player 1
        club_two_1,  # pos 2 -> player 2
        diamond_two_1,  # pos 3 -> player 3
        spade_two_2,  # pos 4 -> player 0 (now has ♠ pair)
        heart_two_2,  # pos 5 -> player 1 (now has ♥ pair)
        club_two_2,  # pos 6 -> player 2 (now has ♣ pair)
        diamond_two_2,  # pos 7 -> player 3 (now has ♦ pair)
        big_jokers[0],  # pos 8 -> player 0
        big_jokers[1],  # pos 9 -> player 1
    ]
    # Remove the big jokers from remaining pool
    big_joker_ids = {c.id for c in big_jokers}
    remaining = [c for c in remaining if c.id not in big_joker_ids]
    deck.extend(remaining[:90])

    # Bottom cards: from remaining pool after deck is filled
    bottom_pool = remaining[90:]
    while len(bottom_pool) < 8:
        bottom_pool.append(small_jokers.pop())
    bottom = bottom_pool[:8]

    return deck[:100], bottom


def _make_joker_pair_deck() -> list[Card]:
    """Create a 100-card deck with both big jokers at positions 0 and 4.

    This ensures both big jokers land in player 0's hand after dealing
    5 cards (CCW order: 0->1->2->3->0).
    """
    all_cards = create_decks()
    big_jokers = [c for c in all_cards if c.rank == Rank.BIG_JOKER]
    remaining = [
        c
        for c in all_cards
        if c.rank not in (Rank.BIG_JOKER, Rank.SMALL_JOKER)
    ]
    import random

    random.seed(77)
    random.shuffle(remaining)
    # Place both big jokers at positions 0 and 4 (both go to player 0)
    custom_deck: list[Card] = [
        big_jokers[0],
        remaining[0],
        remaining[1],
        remaining[2],
        big_jokers[1],
    ]
    # Remove used cards from remaining
    used_ids = {c.id for c in custom_deck}
    rest = [c for c in remaining if c.id not in used_ids]
    custom_deck.extend(rest[:95])
    return custom_deck[:100]


class TestCreateDealBid:
    def test_create_deal_bid_initial_state(self) -> None:
        """
        Initial state: DEALING, cursor=0, target=start_player, no bids.
        """
        deck, _ = _make_deck_with_specific_cards()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        assert state.phase == "DEALING"
        assert state.deal_cursor == 0
        assert state.deal_target == 0
        assert state.bid_winner is None
        assert len(state.bid_events) == 0
        assert all(len(h) == 0 for h in state.players_hand)


class TestDealNextCard:
    def test_deal_next_card_first(self) -> None:
        """First card goes to start_player."""
        deck, _ = _make_deck_with_specific_cards()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        state = _deal(state)
        assert state.deal_cursor == 1
        assert len(state.players_hand[0]) == 1
        assert state.players_hand[0][0].id == deck[0].id

    def test_deal_next_card_advances_target(self) -> None:
        """After dealing to player 0, next target is player 1 (CCW)."""
        deck, _ = _make_deck_with_specific_cards()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        state = _deal(state)
        assert state.deal_target == 1

    def test_deal_next_card_increments_cursor(self) -> None:
        """Deal cursor advances by 1 each time."""
        deck, _ = _make_deck_with_specific_cards()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        for _ in range(5):
            state = _deal(state)
        assert state.deal_cursor == 5

    def test_deal_next_card_distributes_to_hand(self) -> None:
        """Cards go to the correct players in CCW order."""
        deck, _ = _make_deck_with_specific_cards()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        # Deal 4 cards: one to each player
        for _ in range(4):
            state = _deal(state)
        # Each player should have 1 card
        for i in range(4):
            assert len(state.players_hand[i]) == 1

    def test_deal_next_card_all_dealt_with_bid(self) -> None:
        """
        After 100 cards dealt with a bid placed mid-deal,
        all_dealt=True.
        """
        deck, _ = _make_deterministic_deck()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        # Deal 5 cards so player 0 has ♠TWO (positions 0 and 4)
        for _ in range(5):
            state = _deal(state)
        # Place a bid mid-deal
        spade_twos = [
            c
            for c in state.players_hand[0]
            if c.rank == Rank.TWO and c.suit == Suit.SPADES
        ]
        assert len(spade_twos) >= 1
        event = BidEvent(
            player=0,
            cards=[spade_twos[0]],
            kind="trump_rank",
            suit=Suit.SPADES,
            joker_type=None,
            count=1,
        )
        result = reveal(state, event)
        assert isinstance(result, Ok)
        state = result.value
        # Deal remaining 95 cards
        for _ in range(95):
            if state.phase == "DEALING" and not state.all_dealt:
                state = _deal(state)
        # Phase stays DEALING with all_dealt=True — last recipient
        # hasn't acted yet
        assert state.phase == "DEALING"
        assert state.all_dealt

    def test_deal_next_card_all_dealt_no_bid(self) -> None:
        """
        After 100 cards dealt with no bids, all_dealt=True, phase stays
        DEALING.
        """
        deck, _ = _make_deck_with_specific_cards()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        for _ in range(100):
            if state.phase == "DEALING" and not state.all_dealt:
                state = _deal(state)
        # Phase stays DEALING with all_dealt=True — last recipient
        # hasn't acted yet
        assert state.phase == "DEALING"
        assert state.all_dealt


class TestReveal:
    def test_reveal_first_bid_accepted(self) -> None:
        """First reveal is always accepted (when no current winner)."""
        deck, _ = _make_deterministic_deck()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        # Deal 5 cards so player 0 has ♠TWO (positions 0 and 4)
        for _ in range(5):
            state = _deal(state)
        # Player 0 reveals single ♠TWO
        spade_twos = [
            c
            for c in state.players_hand[0]
            if c.rank == Rank.TWO and c.suit == Suit.SPADES
        ]
        assert len(spade_twos) >= 1, (
            "Player 0 should have at least one ♠TWO"
        )
        event = BidEvent(
            player=0,
            cards=[spade_twos[0]],
            kind="trump_rank",
            suit=Suit.SPADES,
            joker_type=None,
            count=1,
        )
        result = reveal(state, event)
        assert isinstance(result, Ok)
        state = result.value
        assert state.bid_winner is not None
        assert state.bid_winner.player == 0
        assert state.bid_winner.suit == Suit.SPADES

    def test_reveal_higher_bid_overrides(self) -> None:
        """Higher value bid overrides current winner."""
        deck, _ = _make_deterministic_deck()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        # Deal 8 cards so player 3 (team 1) has ♦TWO and player 0 has
        # ♠TWO
        for _ in range(8):
            state = _deal(state)
        # Player 3 reveals single ♦TWO (value 100, weakest)
        diamond_twos = [
            c
            for c in state.players_hand[3]
            if c.rank == Rank.TWO and c.suit == Suit.DIAMONDS
        ]
        assert len(diamond_twos) >= 1, (
            "Player 3 should have at least one ♦TWO"
        )
        low_bid = BidEvent(
            player=3,
            cards=[diamond_twos[0]],
            kind="trump_rank",
            suit=Suit.DIAMONDS,
            joker_type=None,
            count=1,
        )
        result = reveal(state, low_bid)
        assert isinstance(result, Ok)
        state = result.value
        assert state.bid_winner is not None
        assert state.bid_winner.player == 3
        # Player 0 reveals single ♠TWO (value 103, strongest single)
        spade_twos = [
            c
            for c in state.players_hand[0]
            if c.rank == Rank.TWO and c.suit == Suit.SPADES
        ]
        assert len(spade_twos) >= 1, (
            "Player 0 should have at least one ♠TWO"
        )
        high_bid = BidEvent(
            player=0,
            cards=[spade_twos[0]],
            kind="trump_rank",
            suit=Suit.SPADES,
            joker_type=None,
            count=1,
        )
        result = reveal(state, high_bid)
        assert isinstance(result, Ok)
        state = result.value
        assert state.bid_winner is not None
        assert state.bid_winner.player == 0
        assert state.bid_winner.suit == Suit.SPADES

    def test_reveal_same_value_rejected(self) -> None:
        """
        Bid with equal or lower value is rejected (strict greater
        required).
        """
        deck, _ = _make_deterministic_deck()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        # Deal 8 cards so player 0 has ♠TWO and player 1 has ♥TWO
        for _ in range(8):
            state = _deal(state)
        # Player 0 reveals single ♠TWO (value 103, highest single)
        spade_twos = [
            c
            for c in state.players_hand[0]
            if c.rank == Rank.TWO and c.suit == Suit.SPADES
        ]
        assert len(spade_twos) >= 1, (
            "Player 0 should have at least one ♠TWO"
        )
        bid1 = BidEvent(
            player=0,
            cards=[spade_twos[0]],
            kind="trump_rank",
            suit=Suit.SPADES,
            joker_type=None,
            count=1,
        )
        result = reveal(state, bid1)
        assert isinstance(result, Ok)
        state = result.value
        # Player 1 tries single ♥TWO (value 102 < 103) -- should be
        # rejected
        heart_twos = [
            c
            for c in state.players_hand[1]
            if c.rank == Rank.TWO and c.suit == Suit.HEARTS
        ]
        assert len(heart_twos) >= 1, (
            "Player 1 should have at least one ♥TWO"
        )
        bid2 = BidEvent(
            player=1,
            cards=[heart_twos[0]],
            kind="trump_rank",
            suit=Suit.HEARTS,
            joker_type=None,
            count=1,
        )
        result = reveal(state, bid2)
        assert isinstance(result, Rejected)
        # ♠(103) > ♥(102), so bid2 should be rejected -- winner
        # unchanged
        assert state.bid_winner is not None
        assert state.bid_winner.player == 0

    def test_reveal_wrong_phase_rejected(self) -> None:
        """Reveal after dealing is finalized is rejected."""
        deck, _ = _make_deck_with_specific_cards()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        for _ in range(100):
            if state.phase == "DEALING" and not state.all_dealt:
                state = _deal(state)
        # Finalize (no bids → NO_BID)
        result = finalize_dealing(state)
        assert isinstance(result, Ok)
        state = result.value
        # Now phase is NO_BID — reveal should be rejected
        bid = BidEvent(
            player=0,
            cards=[],
            kind="trump_rank",
            suit=Suit.HEARTS,
            joker_type=None,
            count=1,
        )
        result2 = reveal(state, bid)
        assert isinstance(result2, Rejected)

    def test_reveal_not_in_hand_rejected(self) -> None:
        """Reveal with cards not in player's hand is rejected."""
        deck, _ = _make_deck_with_specific_cards()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        for _ in range(5):
            state = _deal(state)
        # Create a fake card not in any hand
        fake_card = _card(Suit.HEARTS, Rank.TWO, 1)
        bid = BidEvent(
            player=0,
            cards=[fake_card],
            kind="trump_rank",
            suit=Suit.HEARTS,
            joker_type=None,
            count=1,
        )
        result = reveal(state, bid)
        assert isinstance(result, Rejected)

    def test_reveal_count_cards_mismatch_rejected(self) -> None:
        """Bid with count=2 but only 1 card is rejected."""
        deck, _ = _make_deterministic_deck()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        for _ in range(5):
            state = _deal(state)
        spade_twos = [
            c
            for c in state.players_hand[0]
            if c.rank == Rank.TWO and c.suit == Suit.SPADES
        ]
        assert len(spade_twos) >= 1, (
            "Player 0 should have at least one ♠TWO"
        )
        # Submit 1 card but claim count=2 -- should be rejected
        bid = BidEvent(
            player=0,
            cards=[spade_twos[0]],
            kind="trump_rank",
            suit=Suit.SPADES,
            joker_type=None,
            count=2,
        )
        result = reveal(state, bid)
        assert isinstance(result, Rejected)

    def test_reveal_count_one_with_two_cards_rejected(self) -> None:
        """Bid with count=1 but 2 cards is rejected."""
        deck, _ = _make_deterministic_deck()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        for _ in range(5):
            state = _deal(state)
        spade_twos = [
            c
            for c in state.players_hand[0]
            if c.rank == Rank.TWO and c.suit == Suit.SPADES
        ]
        assert len(spade_twos) >= 2, "Player 0 should have a ♠ pair"
        # Submit 2 cards but claim count=1 -- should be rejected
        bid = BidEvent(
            player=0,
            cards=[spade_twos[0], spade_twos[1]],
            kind="trump_rank",
            suit=Suit.SPADES,
            joker_type=None,
            count=1,
        )
        result = reveal(state, bid)
        assert isinstance(result, Rejected)

    def test_reveal_duplicate_card_ids_rejected(self) -> None:
        """
        Bid submitting the same card ID twice (duplicate physical card)
        is rejected.
        """
        deck, _ = _make_deterministic_deck()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        # Deal 5 cards so player 0 has ♠TWO
        for _ in range(5):
            state = _deal(state)
        spade_twos = [
            c
            for c in state.players_hand[0]
            if c.rank == Rank.TWO and c.suit == Suit.SPADES
        ]
        assert len(spade_twos) >= 1, (
            "Player 0 should have at least one ♠TWO"
        )
        # Submit the same card twice (duplicate ID) as a pair bid
        bid = BidEvent(
            player=0,
            cards=[spade_twos[0], spade_twos[0]],
            kind="trump_rank",
            suit=Suit.SPADES,
            joker_type=None,
            count=2,
        )
        result = reveal(state, bid)
        assert isinstance(result, Rejected)

    def test_reveal_non_trump_rank_rejected(self) -> None:
        """Revealing non-trump-rank cards is rejected."""
        deck, _ = _make_deterministic_deck()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        # Deal 13 cards so player 0 has non-trump-rank cards
        # (CCW: P0 gets pos 0,4,8,12; pos 12 is from remaining pool)
        for _ in range(13):
            state = _deal(state)
        # Find a non-trump-rank card in player 0's hand
        non_rank = [
            c
            for c in state.players_hand[0]
            if c.rank != Rank.TWO and not c.is_joker
        ]
        assert len(non_rank) >= 1, (
            "Player 0 should have at least one non-trump-rank card"
        )
        bid = BidEvent(
            player=0,
            cards=[non_rank[0]],
            kind="trump_rank",
            suit=non_rank[0].suit,
            joker_type=None,
            count=1,
        )
        result = reveal(state, bid)
        assert isinstance(result, Rejected)

    def test_reveal_single_joker_rejected(self) -> None:
        """Single joker cannot be used for reveal (must be pair)."""
        deck, _ = _make_deterministic_deck()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        # Deal 10 cards so player 0 has at least one big joker (pos 8)
        for _ in range(10):
            state = _deal(state)
        # Find a single joker in player 0's hand
        jokers = [c for c in state.players_hand[0] if c.is_joker]
        assert len(jokers) >= 1, (
            "Player 0 should have at least one joker"
        )
        bid = BidEvent(
            player=0,
            cards=[jokers[0]],
            kind="joker",
            suit=None,
            joker_type="big" if jokers[0].is_big_joker else "small",
            count=1,
        )
        result = reveal(state, bid)
        assert isinstance(result, Rejected)

    def test_reveal_joker_pair_accepted(self) -> None:
        """Pair of big jokers is valid for reveal and accepted."""
        custom_deck = _make_joker_pair_deck()
        state = create_deal_bid(
            DealBidInput(
                deck=custom_deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        for _ in range(5):
            state = _deal(state)
        # Find both big jokers in player 0's hand
        bj = [
            c for c in state.players_hand[0] if c.rank == Rank.BIG_JOKER
        ]
        assert len(bj) >= 2, "Player 0 should have 2 big jokers by now"
        event = BidEvent(
            player=0,
            cards=[bj[0], bj[1]],
            kind="joker",
            suit=None,
            joker_type="big",
            count=2,
        )
        result = reveal(state, event)
        assert isinstance(result, Ok)
        state = result.value
        # Joker pair should be accepted: bid_events grows, bid_winner
        # updated
        assert len(state.bid_events) == 1
        assert state.bid_winner is not None
        assert state.bid_winner.player == 0
        assert state.bid_winner.kind == "joker"
        assert state.bid_winner.count == 2

    def test_reveal_joker_pair_sets_no_trump(self) -> None:
        """
        Joker pair reveal sets trump_suit=None (无主) when deal
        completes.
        """
        custom_deck = _make_joker_pair_deck()
        state = create_deal_bid(
            DealBidInput(
                deck=custom_deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        # Deal 5 cards so player 0 has both big jokers
        for _ in range(5):
            state = _deal(state)
        # Reveal big joker pair
        bj = [
            c for c in state.players_hand[0] if c.rank == Rank.BIG_JOKER
        ]
        assert len(bj) >= 2
        event = BidEvent(
            player=0,
            cards=[bj[0], bj[1]],
            kind="joker",
            suit=None,
            joker_type="big",
            count=2,
        )
        result = reveal(state, event)
        assert isinstance(result, Ok)
        state = result.value
        # Deal remaining cards
        for _ in range(95):
            if state.phase == "DEALING" and not state.all_dealt:
                state = _deal(state)
        # After all cards dealt, phase stays DEALING with all_dealt=True
        assert state.phase == "DEALING"
        assert state.all_dealt
        # Finalize → COMPLETE (joker bid exists)
        result2 = finalize_dealing(state)
        assert isinstance(result2, Ok)
        state = result2.value
        assert state.phase == "COMPLETE"
        # The result should have trump_suit=None (无主)
        result_data = _get_result(state)
        assert result_data.trump_suit is None
        assert result_data.winner == 0

    def test_reveal_subsequent_round_non_declarer_team_accepted(
        self,
    ) -> None:
        """
        In subsequent rounds, all players can reveal to choose trump
        suit.
        """
        deck, _ = _make_deterministic_deck()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=1,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        # Deal 5 cards so player 0 has ♠TWO
        for _ in range(5):
            state = _deal(state)
        # Player 0 is team 0, but can still reveal in later rounds.
        spade_twos = [
            c
            for c in state.players_hand[0]
            if c.rank == Rank.TWO and c.suit == Suit.SPADES
        ]
        assert len(spade_twos) >= 1, (
            "Player 0 should have at least one ♠TWO"
        )
        bid = BidEvent(
            player=0,
            cards=[spade_twos[0]],
            kind="trump_rank",
            suit=Suit.SPADES,
            joker_type=None,
            count=1,
        )
        result = reveal(state, bid)
        assert isinstance(result, Ok)

    def test_reveal_subsequent_round_declarer_team_accepted(
        self,
    ) -> None:
        """In subsequent rounds, declarer-team players can reveal."""
        deck, _ = _make_deterministic_deck()
        # declarer_team=1 means only team1 players (1,2) can reveal
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=1,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        # Deal 6 cards so player 1 has ♥TWO (position 1 and 5)
        for _ in range(6):
            state = _deal(state)
        # Player 1 is team 1 (declarer), should be accepted
        heart_twos = [
            c
            for c in state.players_hand[1]
            if c.rank == Rank.TWO and c.suit == Suit.HEARTS
        ]
        assert len(heart_twos) >= 1, (
            "Player 1 should have at least one ♥TWO"
        )
        bid = BidEvent(
            player=1,
            cards=[heart_twos[0]],
            kind="trump_rank",
            suit=Suit.HEARTS,
            joker_type=None,
            count=1,
        )
        result = reveal(state, bid)
        assert isinstance(result, Ok)
        state = result.value
        assert state.bid_winner is not None
        assert state.bid_winner.player == 1
        assert state.bid_winner.suit == Suit.HEARTS


class TestDealBidFullFlow:
    def test_deal_bid_full_flow_no_bid(self) -> None:
        """
        Complete flow: deal all cards with no bids, finalize → NO_BID.
        """
        deck, _ = _make_deck_with_specific_cards()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        # Deal all cards
        for _ in range(100):
            if state.phase == "DEALING" and not state.all_dealt:
                state = _deal(state)
        # Phase stays DEALING with all_dealt=True
        assert state.phase == "DEALING"
        assert state.all_dealt
        # Finalize → NO_BID
        result = finalize_dealing(state)
        assert isinstance(result, Ok)
        assert result.value.phase == "NO_BID"

    def test_deal_bid_no_bid_empty_trump(self) -> None:
        """
        No bids = finalize → NO_BID phase, result has winner=None,
        trump_suit=None.
        """
        deck, _ = _make_deck_with_specific_cards()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        for _ in range(100):
            if state.phase == "DEALING" and not state.all_dealt:
                state = _deal(state)
        # Finalize
        result = finalize_dealing(state)
        assert isinstance(result, Ok)
        state = result.value
        assert state.phase == "NO_BID"
        result_data = _get_result(state)
        assert result_data.winner is None
        assert result_data.trump_suit is None
        assert result_data.bid_count == 0

    def test_deal_bid_bid_value_ordering(self) -> None:
        """
        Bid values: pair♠(203) > pair♥(202) > single♠(103) >
        single♦(100).
        """
        from server.rules.ordering import bid_value

        c_d = _card(Suit.DIAMONDS, Rank.TWO, 1)
        c_s = _card(Suit.SPADES, Rank.TWO, 1)
        c_h1 = _card(Suit.HEARTS, Rank.TWO, 1)
        c_h2 = _card(Suit.HEARTS, Rank.TWO, 2)

        assert bid_value([c_s], Rank.TWO) > bid_value([c_d], Rank.TWO)
        assert bid_value([c_h1, c_h2], Rank.TWO) > bid_value(
            [c_s], Rank.TWO
        )


class TestLastCardBidOpportunity:
    """
    Tests for the bug fix: last card recipient must get a chance to
    bid/skip.
    """

    def test_deal_next_card_last_card_stays_dealing(self) -> None:
        """
        After dealing the 100th card, phase stays DEALING with
        all_dealt=True.

        The recipient of the last card must still be able to bid or
        skip.
        """
        deck, _ = _make_deck_with_specific_cards()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        # Deal all 100 cards
        for _ in range(100):
            if state.phase == "DEALING" and not state.all_dealt:
                state = _deal(state)
        # Phase should still be DEALING (not COMPLETE/NO_BID)
        assert state.phase == "DEALING"
        assert state.all_dealt is True
        assert state.deal_cursor == 100

    def test_last_card_recipient_can_bid(self) -> None:
        """Last card recipient can still bid after all_dealt=True."""
        deck, _ = _make_deterministic_deck()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        # Deal all 100 cards
        for _ in range(100):
            if state.phase == "DEALING" and not state.all_dealt:
                state = _deal(state)
        assert state.all_dealt is True
        # The last card was dealt to a player — they should be able to
        # bid
        # if they have a trump-rank card (reveal still works in DEALING
        # phase)
        assert state.phase == "DEALING"
        # Verify reveal is not rejected due to phase
        # (the specific bid validity depends on hand contents, so just
        # verify
        # that the phase check passes — not rejected with "wrong phase")
        fake_bid = BidEvent(
            player=0,
            cards=[],
            kind="trump_rank",
            suit=Suit.SPADES,
            joker_type=None,
            count=1,
        )
        result = reveal(state, fake_bid)
        # Should NOT be rejected with "wrong phase" — other rejections
        # are fine
        if isinstance(result, Rejected):
            assert "当前阶段" not in result.reason

    def test_last_card_bid_then_finalize_complete(self) -> None:
        """After last card recipient bids, finalize → COMPLETE."""
        deck, _ = _make_deterministic_deck()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        # Deal 5 cards, place a bid mid-deal
        for _ in range(5):
            state = _deal(state)
        spade_twos = [
            c
            for c in state.players_hand[0]
            if c.rank == Rank.TWO and c.suit == Suit.SPADES
        ]
        assert len(spade_twos) >= 1
        event = BidEvent(
            player=0,
            cards=[spade_twos[0]],
            kind="trump_rank",
            suit=Suit.SPADES,
            joker_type=None,
            count=1,
        )
        result = reveal(state, event)
        assert isinstance(result, Ok)
        state = result.value
        # Deal remaining cards
        for _ in range(95):
            if state.phase == "DEALING" and not state.all_dealt:
                state = _deal(state)
        assert state.all_dealt is True
        # Finalize → COMPLETE (bid_winner exists)
        result2 = finalize_dealing(state)
        assert isinstance(result2, Ok)
        assert result2.value.phase == "COMPLETE"

    def test_last_card_skip_then_finalize_no_bid(self) -> None:
        """
        After last card recipient skips (no bids), finalize → NO_BID.
        """
        deck, _ = _make_deck_with_specific_cards()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        # Deal all cards (no bids placed)
        for _ in range(100):
            if state.phase == "DEALING" and not state.all_dealt:
                state = _deal(state)
        assert state.all_dealt is True
        # Finalize → NO_BID (no bid_winner)
        result = finalize_dealing(state)
        assert isinstance(result, Ok)
        assert result.value.phase == "NO_BID"

    def test_finalize_before_all_dealt_rejected(self) -> None:
        """finalize_dealing rejected when not all cards dealt yet."""
        deck, _ = _make_deck_with_specific_cards()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        # Deal only 50 cards
        for _ in range(50):
            state = _deal(state)
        assert state.all_dealt is False
        result = finalize_dealing(state)
        assert isinstance(result, Rejected)

    def test_all_dealt_flag_transitions(self) -> None:
        """all_dealt is False until the 100th card, then True."""
        deck, _ = _make_deck_with_specific_cards()
        state = create_deal_bid(
            DealBidInput(
                deck=deck,
                declarer_team=None,
                trump_rank=Rank.TWO,
                start_player=0,
            )
        )
        # Deal 99 cards — all_dealt should be False
        for _ in range(99):
            if state.phase == "DEALING" and not state.all_dealt:
                state = _deal(state)
        assert state.all_dealt is False
        assert state.deal_cursor == 99
        # Deal the 100th card — all_dealt becomes True
        state = _deal(state)
        assert state.all_dealt is True
        assert state.deal_cursor == 100


def _get_result(state: DealBidState) -> DealBidResult:
    """Extract result from a completed DealBidState."""
    if state.bid_winner is not None:
        return DealBidResult(
            winner=state.bid_winner.player,
            trump_suit=state.bid_winner.suit,
            bid_count=state.bid_winner.count,
            players_hand=state.players_hand,
            bid_events=state.bid_events,
        )
    return DealBidResult(
        winner=None,
        trump_suit=None,
        bid_count=0,
        players_hand=state.players_hand,
        bid_events=state.bid_events,
    )


def test_get_bid_action_hints_singles_and_pairs() -> None:
    """
    get_bid_action_hints returns complete bid hint singles and pairs.
    """
    trump_rank = Rank.TWO
    hand: list[Card] = [
        _card(Suit.CLUBS, Rank.TWO, 1),
        _card(Suit.CLUBS, Rank.TWO, 2),
        _card(Suit.SPADES, Rank.TWO, 1),
        _card(Suit.JOKER, Rank.SMALL_JOKER, 1),
        _card(Suit.HEARTS, Rank.FIVE, 1),
    ]
    state = DealBidState(
        phase="DEALING",
        deck=[],
        deal_cursor=0,
        deal_target=0,
        bid_winner=None,
        bid_events=[],
        players_hand=[hand, [], [], []],
        declarer_team=None,
        trump_rank=trump_rank,
        start_player=0,
    )

    result = get_bid_action_hints(state, player=0)

    assert isinstance(result, list)
    assert len(result) > 0

    all_bid_card_ids: set[str] = set()
    for option in result:
        assert isinstance(option, list)
        assert len(option) in (1, 2)
        for card in option:
            assert isinstance(card, Card)
            all_bid_card_ids.add(card.id)

    assert "D1-clubs-2" in all_bid_card_ids
    assert "D2-clubs-2" in all_bid_card_ids
    assert "D1-spades-2" in all_bid_card_ids
    assert "D1-hearts-5" not in all_bid_card_ids


def test_get_bid_action_hints_ordered_from_small_to_large() -> None:
    """
    Bid hints are ordered as 10 logical options from small to large.
    """
    trump_rank = Rank.TWO
    hand: list[Card] = [
        _card(Suit.DIAMONDS, Rank.TWO, 1),
        _card(Suit.DIAMONDS, Rank.TWO, 2),
        _card(Suit.CLUBS, Rank.TWO, 1),
        _card(Suit.CLUBS, Rank.TWO, 2),
        _card(Suit.HEARTS, Rank.TWO, 1),
        _card(Suit.HEARTS, Rank.TWO, 2),
        _card(Suit.SPADES, Rank.TWO, 1),
        _card(Suit.SPADES, Rank.TWO, 2),
        _card(Suit.JOKER, Rank.SMALL_JOKER, 1),
        _card(Suit.JOKER, Rank.SMALL_JOKER, 2),
        _card(Suit.JOKER, Rank.BIG_JOKER, 1),
        _card(Suit.JOKER, Rank.BIG_JOKER, 2),
    ]
    state = DealBidState(
        phase="DEALING",
        deck=[],
        deal_cursor=0,
        deal_target=0,
        bid_winner=None,
        bid_events=[],
        players_hand=[hand, [], [], []],
        declarer_team=None,
        trump_rank=trump_rank,
        start_player=0,
    )

    result = get_bid_action_hints(state, player=0)

    assert len(result) == MAX_BID_ACTION_HINTS
    assert [[card.id for card in option] for option in result] == [
        ["D1-diamonds-2"],
        ["D1-clubs-2"],
        ["D1-hearts-2"],
        ["D1-spades-2"],
        ["D1-diamonds-2", "D2-diamonds-2"],
        ["D1-clubs-2", "D2-clubs-2"],
        ["D1-hearts-2", "D2-hearts-2"],
        ["D1-spades-2", "D2-spades-2"],
        ["D1-joker-SJ", "D2-joker-SJ"],
        ["D1-joker-BJ", "D2-joker-BJ"],
    ]


def test_get_bid_action_hints_allow_all_players_in_later_rounds() -> (
    None
):
    """get_bid_action_hints includes legal reveals from any team."""
    trump_rank = Rank.TWO
    hand: list[Card] = [
        _card(Suit.CLUBS, Rank.TWO, 1),
    ]
    state = DealBidState(
        phase="DEALING",
        deck=[],
        deal_cursor=0,
        deal_target=1,
        bid_winner=None,
        bid_events=[],
        players_hand=[[], hand, [], []],
        declarer_team=0,
        trump_rank=trump_rank,
        start_player=0,
    )

    hints = get_bid_action_hints(state, player=1)
    assert len(hints) == 1
    assert hints[0][0].id == "D1-clubs-2"
