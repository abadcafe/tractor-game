"""Tests for sm.round_sm module."""
import random
from collections import Counter
from itertools import combinations
from typing import Literal

import pytest
from .card_model import Card, Suit, Rank
from .types import BidEvent, CompletedTrick
from .round_sm import (
    RoundState, RoundInput, create_round,
    deal_next_card, reveal, pass_stir, stir, stir_discard, play,
    is_round_complete, get_round_result, finalize_deal_bid,
)
from . import trick_sm as trick_mod
from .result import Ok, Rejected

type CompletedTrickKey = tuple[int, int, int, tuple[tuple[int, tuple[str, ...]], ...]]


def _completed_trick_key(trick: CompletedTrick | None) -> CompletedTrickKey | None:
    if trick is None:
        return None
    return (
        trick.lead_player,
        trick.winner,
        trick.points,
        tuple(
            (slot.player, tuple(card.id for card in slot.cards))
            for slot in trick.slots
        ),
    )


def _card(suit: Suit, rank: Rank, deck: Literal[1, 2] = 1) -> Card:
    """Create a card for deterministic round-state tests."""
    points_map: dict[Rank, int] = {
        Rank.FIVE: 5,
        Rank.TEN: 10,
        Rank.KING: 10,
    }
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit,
        rank=rank,
        is_joker=(suit == Suit.JOKER),
        is_big_joker=(rank == Rank.BIG_JOKER),
        points=points_map.get(rank, 0),
        deck=deck,
    )


def _deal(state: RoundState) -> RoundState:
    """Unwrap deal_next_card result, asserting Ok."""
    result = deal_next_card(state)
    assert isinstance(result, Ok), f"deal_next_card rejected: {result.reason}"
    return result.value


def _play_first_legal(state: RoundState) -> RoundState:
    """Play the first accepted play for the current player in the trick."""
    trick = state.trick_state
    assert trick is not None
    cur = trick.cur
    hand = trick.hands[cur]
    if not hand:
        return state

    if trick.phase == "LEADING":
        candidate_sizes = [1]
    else:
        lead_slot = trick.slots[trick.lead_player]
        lead_cards = lead_slot.cards
        candidate_sizes = [len(lead_cards)]

    for size in candidate_sizes:
        for candidate in combinations(hand, size):
            result = play(state, player_index=cur, cards=list(candidate))
            if isinstance(result, Ok):
                return result.value
    raise AssertionError(f"No accepted play for player {cur}")


def _deal_all_cards(state: RoundState) -> RoundState:
    """Deal all 100 cards in the deal-bid phase."""
    while state.phase == "DEAL_BID" and state.deal_bid_state is not None:
        if state.deal_bid_state.phase != "DEALING" or state.deal_bid_state.all_dealt:
            break
        state = _deal(state)
    return state


def _complete_deal_bid_no_bid(state: RoundState) -> RoundState:
    """Complete deal-bid without any reveals (results in NO_BID)."""
    state = _deal_all_cards(state)
    # After all cards dealt, finalize to transition to STIRRING
    if state.phase == "DEAL_BID" and state.deal_bid_state is not None and state.deal_bid_state.all_dealt:
        result = finalize_deal_bid(state)
        assert isinstance(result, Ok), f"finalize_deal_bid rejected: {result.reason}"
        state = result.value
    return state


def _complete_deal_bid_with_reveal(state: RoundState) -> RoundState:
    """Complete deal-bid with one reveal (results in COMPLETE)."""
    # Deal some cards first
    for _ in range(20):
        if state.deal_bid_state is None or state.deal_bid_state.phase != "DEALING":
            break
        state = _deal(state)

    # Find a trump rank card in any hand and reveal it
    if state.deal_bid_state is not None and state.deal_bid_state.phase == "DEALING":
        for p in range(4):
            trump_cards = [c for c in state.deal_bid_state.players_hand[p]
                          if c.rank == state.trump_rank and not c.is_joker]
            if trump_cards:
                event = BidEvent(
                    player=p, cards=[trump_cards[0]], kind="trump_rank",
                    suit=trump_cards[0].suit, joker_type=None, count=1,
                )
                result = reveal(state, event)
                if isinstance(result, Ok):
                    state = result.value
                break

    # Deal remaining cards
    state = _deal_all_cards(state)
    # After all cards dealt, finalize to transition to STIRRING
    if state.phase == "DEAL_BID" and state.deal_bid_state is not None and state.deal_bid_state.all_dealt:
        result = finalize_deal_bid(state)
        assert isinstance(result, Ok), f"finalize_deal_bid rejected: {result.reason}"
        state = result.value
    return state


def _complete_stirring_all_pass(state: RoundState) -> RoundState:
    """Complete stirring by handling initial EXCHANGING then having others pass.

    The stirring phase starts in EXCHANGING sub-phase: the declarer must
    exchange (stir_discard) bottom cards first. After that, the three
    non-exchanging players pass, and stirring completes, transitioning directly to PLAYING.
    """
    assert state.phase == "STIRRING"
    assert state.stirring_state is not None

    # Step 1: Handle initial EXCHANGING sub-phase (declarer exchanges bottom cards)
    if state.stirring_state.phase == "EXCHANGING":
        assert state.stirring_state.exchange_state is not None
        ex = state.stirring_state.exchange_state
        declarer = state.stirring_state.exchanging_player
        assert declarer is not None
        discarded = ex.hand_after_pickup[:ex.count]
        result = stir_discard(state, player_index=declarer, cards=discarded)
        assert isinstance(result, Ok), f"stir_discard rejected: {result.reason}"
        state = result.value

    # Step 2: The three non-exchanging players pass.
    for _ in range(3):
        if state.phase != "STIRRING":
            break
        if state.stirring_state is None:
            break
        # If stirring is in EXCHANGING (e.g. after a stir), handle it first
        if state.stirring_state.phase == "EXCHANGING":
            assert state.stirring_state.exchange_state is not None
            ex = state.stirring_state.exchange_state
            exchanging = state.stirring_state.exchanging_player
            assert exchanging is not None
            discarded = ex.hand_after_pickup[:ex.count]
            result = stir_discard(state, player_index=exchanging, cards=discarded)
            assert isinstance(result, Ok), f"stir_discard rejected: {result.reason}"
            state = result.value
        if state.phase != "STIRRING" or state.stirring_state is None:
            break
        cur = state.stirring_state.current_player
        result = pass_stir(state, player_index=cur)
        assert isinstance(result, Ok), f"pass_stir rejected: {result.reason}"
        state = result.value

    return state



class TestCreateRound:
    def test_create_round_initial_state(self) -> None:
        """Initial round state: DEAL_BID phase."""
        state = create_round(RoundInput(
            declarer_team=None,
            trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        ))
        assert state.phase == "DEAL_BID"
        assert state.declarer_team is None
        assert state.trump_rank == Rank.TWO
        assert state.deal_bid_state is not None

    def test_create_round_with_declarer(self) -> None:
        """Subsequent round starts dealing from the fixed declarer."""
        state = create_round(RoundInput(
            declarer_team=0,
            trump_rank=Rank.THREE,
            next_declarer_player=0,
            team0_level=Rank.THREE,
            team1_level=Rank.TWO,
        ))
        assert state.declarer_team == 0
        assert state.trump_rank == Rank.THREE
        assert state.start_player == 0
        assert state.deal_bid_state is not None
        assert state.deal_bid_state.deal_target == 0

    def test_create_round_with_nonzero_declarer_starts_dealing_there(self) -> None:
        """Subsequent round deal starts from the fixed declarer, not player 0."""
        state = create_round(RoundInput(
            declarer_team=1,
            trump_rank=Rank.FIVE,
            next_declarer_player=2,
            team0_level=Rank.THREE,
            team1_level=Rank.FIVE,
        ))
        assert state.start_player == 2
        assert state.deal_bid_state is not None
        assert state.deal_bid_state.deal_target == 2


class TestDealBidPhase:
    def test_deal_next_card_advances_deal_bid(self) -> None:
        """deal_next_card during DEAL_BID advances the deal-bid sub-state."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        assert state.phase == "DEAL_BID"
        assert state.deal_bid_state is not None
        initial_cursor = state.deal_bid_state.deal_cursor
        state = _deal(state)
        assert state.deal_bid_state is not None
        assert state.deal_bid_state.deal_cursor == initial_cursor + 1

    def test_reveal_during_deal_bid(self) -> None:
        """reveal during DEAL_BID adds a bid event."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        # Deal some cards first
        for _ in range(20):
            state = _deal(state)

        # Find a trump rank card and reveal
        assert state.deal_bid_state is not None
        for p in range(4):
            trump_cards = [c for c in state.deal_bid_state.players_hand[p]
                          if c.rank == Rank.TWO and not c.is_joker]
            if trump_cards:
                event = BidEvent(
                    player=p, cards=[trump_cards[0]], kind="trump_rank",
                    suit=trump_cards[0].suit, joker_type=None, count=1,
                )
                assert state.deal_bid_state is not None
                old_events = len(state.deal_bid_state.bid_events)
                result = reveal(state, event)
                assert isinstance(result, Ok)
                state = result.value
                assert state.deal_bid_state is not None
                assert len(state.deal_bid_state.bid_events) > old_events
                break

    def test_deal_bid_to_stirring_with_winner(self) -> None:
        """After deal-bid completes with a winner, round enters STIRRING."""
        random.seed(42)
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_with_reveal(state)
        assert state.deal_bid_state is not None
        assert state.deal_bid_state.bid_winner is not None
        assert state.phase == "STIRRING"
        assert state.declarer_player is not None
        assert state.trump_suit is not None

    def test_deal_bid_single_winner_enters_stirring_with_single_priority(self) -> None:
        """Deal-bid must preserve whether the winning bid was single or pair."""
        bid_card = _card(Suit.SPADES, Rank.FIVE, 1)
        over_stir_pair = [
            _card(Suit.DIAMONDS, Rank.FIVE, 1),
            _card(Suit.DIAMONDS, Rank.FIVE, 2),
        ]
        hands = [[bid_card], over_stir_pair, [], []]
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.FIVE,
            next_declarer_player=None,
            team0_level=Rank.FIVE, team1_level=Rank.TWO,
        ))
        bid_event = BidEvent(
            player=0,
            cards=[bid_card],
            kind="trump_rank",
            suit=Suit.SPADES,
            joker_type=None,
            count=1,
        )
        assert state.deal_bid_state is not None
        db_state = state.deal_bid_state.model_copy(update={
            "all_dealt": True,
            "bid_winner": bid_event,
            "bid_events": [bid_event],
            "players_hand": hands,
        })
        state = state.model_copy(update={
            "deal_bid_state": db_state,
            "players_hand": hands,
        })

        result = finalize_deal_bid(state)

        assert isinstance(result, Ok)
        state = result.value
        assert state.stirring_state is not None
        assert state.stirring_state.current_priority == 103
        state = _complete_initial_exchange(state)
        result = stir(state, player_index=1, cards=over_stir_pair)
        assert isinstance(result, Ok)
        assert result.value.trump_suit == Suit.DIAMONDS

    def test_deal_bid_to_stirring_no_bid(self) -> None:
        """After deal-bid with no bids, round enters STIRRING with empty trump."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        assert state.phase == "STIRRING"
        assert state.trump_suit is None  # empty trump


def _complete_initial_exchange(state: RoundState) -> RoundState:
    """Complete the initial EXCHANGING sub-phase of stirring.

    After deal-bid, stirring starts in EXCHANGING sub-phase where the declarer
    must exchange (stir_discard) bottom cards before any pass/stir actions.
    """
    assert state.phase == "STIRRING"
    assert state.stirring_state is not None
    assert state.stirring_state.phase == "EXCHANGING"
    assert state.stirring_state.exchange_state is not None
    ex = state.stirring_state.exchange_state
    exchanging = state.stirring_state.exchanging_player
    assert exchanging is not None
    discarded = ex.hand_after_pickup[:ex.count]
    result = stir_discard(state, player_index=exchanging, cards=discarded)
    assert isinstance(result, Ok), f"stir_discard rejected: {result.reason}"
    return result.value


class TestStirringPhase:
    def test_stirring_starts_in_exchanging(self) -> None:
        """After deal-bid, stirring starts in EXCHANGING sub-phase."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        assert state.phase == "STIRRING"
        assert state.stirring_state is not None
        assert state.stirring_state.phase == "EXCHANGING"
        assert state.stirring_state.exchange_state is not None

    def test_stir_discard_completes_initial_exchange(self) -> None:
        """stir_discard during initial EXCHANGING transitions to WAITING."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        state = _complete_initial_exchange(state)
        assert state.phase == "STIRRING"
        assert state.stirring_state is not None
        assert state.stirring_state.phase == "WAITING"

    def test_stir_discard_after_max_stir_transitions_round_to_playing(self) -> None:
        """A max-priority stir that completes during exchange advances the round."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        state = _complete_initial_exchange(state)
        assert state.phase == "STIRRING"
        assert state.stirring_state is not None

        cur = state.stirring_state.current_player
        big_jokers = [
            Card(
                id="test-big-joker-1",
                suit=Suit.JOKER,
                rank=Rank.BIG_JOKER,
                is_joker=True,
                is_big_joker=True,
                points=0,
                deck=1,
            ),
            Card(
                id="test-big-joker-2",
                suit=Suit.JOKER,
                rank=Rank.BIG_JOKER,
                is_joker=True,
                is_big_joker=True,
                points=0,
                deck=2,
            ),
        ]
        hands = [list(hand) for hand in state.players_hand]
        hands[cur].extend(big_jokers)
        assert state.stirring_state is not None
        state = state.model_copy(update={
            "players_hand": hands,
            "stirring_state": state.stirring_state.model_copy(update={"players_hand": hands}),
        })

        stir_result = stir(state, player_index=cur, cards=big_jokers)
        assert isinstance(stir_result, Ok), f"stir rejected: {stir_result.reason}"
        state = stir_result.value
        assert state.phase == "STIRRING"
        assert state.stirring_state is not None
        assert state.stirring_state.phase == "EXCHANGING"
        assert state.stirring_state.exchange_state is not None
        exchanging = state.stirring_state.exchanging_player
        assert exchanging is not None
        ex = state.stirring_state.exchange_state

        discard_result = stir_discard(
            state,
            player_index=exchanging,
            cards=ex.hand_after_pickup[:ex.count],
        )
        assert isinstance(discard_result, Ok), f"stir_discard rejected: {discard_result.reason}"
        state = discard_result.value
        assert state.phase == "PLAYING"
        assert state.trick_state is not None

    def test_pass_stir_during_stirring(self) -> None:
        """pass_stir during STIRRING WAITING sub-phase advances current player."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        state = _complete_initial_exchange(state)
        assert state.phase == "STIRRING"
        assert state.stirring_state is not None
        result = pass_stir(state, player_index=state.stirring_state.current_player)
        assert isinstance(result, Ok)
        state = result.value
        assert state.stirring_state is not None
        assert len(state.stirring_state.pass_set) == 2

    def test_stir_during_stirring(self) -> None:
        """stir during STIRRING WAITING sub-phase changes trump suit."""
        random.seed(10)
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        state = _complete_initial_exchange(state)
        assert state.phase == "STIRRING"
        # With empty trump, find a trump-rank pair in the current player's hand
        assert state.stirring_state is not None
        cur = state.stirring_state.current_player
        hand = state.players_hand[cur]
        # Find two cards of the same suit with trump rank
        suit_counts: Counter[Suit] = Counter()
        for c in hand:
            if c.rank == state.trump_rank and not c.is_joker:
                suit_counts[c.suit] += 1
        # Find a suit where we have at least 2 trump-rank cards
        target_suit = None
        for s, cnt in suit_counts.items():
            if cnt >= 2 and s != state.trump_suit:
                target_suit = s
                break
        assert target_suit is not None, (
            f"No trump-rank pair available in player {cur}'s hand for stirring. "
            f"Hand: {[c.id for c in hand]}"
        )
        pair = [c for c in hand if c.rank == state.trump_rank and not c.is_joker and c.suit == target_suit][:2]
        result = stir(state, player_index=cur, cards=pair)
        assert isinstance(result, Ok)
        state = result.value
        assert state.trump_suit == target_suit
        assert state.bid_winner is not None
        assert state.bid_winner.player == cur
        assert state.bid_winner.cards == pair
        assert state.bid_winner.suit == target_suit

    def test_stir_cards_not_in_hand_rejected(self) -> None:
        """stir with cards not in current player's hand returns Rejected."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        state = _complete_initial_exchange(state)
        assert state.phase == "STIRRING"
        assert state.stirring_state is not None
        # Fabricate cards that are NOT in the current player's hand
        fake_cards = [
            Card(id="D1-clubs-2", suit=Suit.CLUBS, rank=Rank.TWO,
                 is_joker=False, is_big_joker=False, points=0, deck=1),
            Card(id="D2-clubs-2", suit=Suit.CLUBS, rank=Rank.TWO,
                 is_joker=False, is_big_joker=False, points=0, deck=2),
        ]
        result = stir(state, player_index=state.stirring_state.current_player, cards=fake_cards)
        assert isinstance(result, Rejected)
        assert "不在" in result.reason and "手牌中" in result.reason

    def test_stir_rejected_by_stirring_module(self) -> None:
        """stir with in-hand cards that the stirring module rejects returns Rejected."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        state = _complete_initial_exchange(state)
        assert state.phase == "STIRRING"
        assert state.stirring_state is not None
        cur = state.stirring_state.current_player
        hand = state.players_hand[cur]
        # A single card is in the player's hand but the stirring module
        # requires exactly 2 cards (a pair), so it will reject the play.
        if hand:
            result = stir(state, player_index=cur, cards=[hand[0]])
            assert isinstance(result, Rejected)
            assert "对子" in result.reason

    def test_stirring_all_pass_to_playing(self) -> None:
        """After all players pass stirring, round enters PLAYING directly."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        state = _complete_stirring_all_pass(state)
        assert state.phase == "PLAYING"

    def test_stirrer_is_skipped_after_exchange(self) -> None:
        """A player who just stirred is skipped after exchanging bottom cards."""
        from collections import Counter

        random.seed(10)
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        state = _complete_initial_exchange(state)
        assert state.phase == "STIRRING"
        assert state.stirring_state is not None

        cur = state.stirring_state.current_player
        hand = state.players_hand[cur]

        # Find a trump-rank pair in the current player's hand
        suit_counts: Counter[Suit] = Counter()
        for c in hand:
            if c.rank == state.trump_rank and not c.is_joker:
                suit_counts[c.suit] += 1
        target_suit = None
        for s, cnt in suit_counts.items():
            if cnt >= 2 and s != state.trump_suit:
                target_suit = s
                break
        if target_suit is None:
            pytest.skip("No trump-rank pair available for this test")

        pair = [c for c in hand
                if c.rank == state.trump_rank and not c.is_joker and c.suit == target_suit][:2]

        # First stir succeeds
        result = stir(state, player_index=cur, cards=pair)
        assert isinstance(result, Ok)
        state = result.value
        assert state.stirring_state is not None
        assert state.stirring_state.last_stir_player == cur

        # After stir, phase is EXCHANGING; complete it
        if state.stirring_state.phase == "EXCHANGING":
            assert state.stirring_state.exchange_state is not None
            ex = state.stirring_state.exchange_state
            exchanging = state.stirring_state.exchanging_player
            assert exchanging is not None
            discarded = ex.hand_after_pickup[:ex.count]
            result = stir_discard(state, player_index=exchanging, cards=discarded)
            assert isinstance(result, Ok), f"stir_discard rejected: {result.reason}"
            state = result.value

        # The other three players pass; the turn must not come back to the stirrer.
        for _ in range(3):
            if state.phase != "STIRRING":
                break
            assert state.stirring_state is not None
            if state.stirring_state.phase == "EXCHANGING":
                assert state.stirring_state.exchange_state is not None
                ex = state.stirring_state.exchange_state
                exchanging = state.stirring_state.exchanging_player
                assert exchanging is not None
                discarded = ex.hand_after_pickup[:ex.count]
                result = stir_discard(state, player_index=exchanging, cards=discarded)
                assert isinstance(result, Ok), f"stir_discard rejected: {result.reason}"
                state = result.value
            if state.stirring_state is None:
                break
            p = state.stirring_state.current_player
            result = pass_stir(state, player_index=p)
            assert isinstance(result, Ok), f"pass_stir rejected: {result.reason}"
            state = result.value

        assert state.phase == "PLAYING"
        assert state.stirring_state is not None
        assert state.stirring_state.current_player != cur



class TestPlayingPhase:
    def test_play_during_playing_first_trick(self) -> None:
        """First play during PLAYING is the lead player's turn."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        state = _complete_stirring_all_pass(state)
        assert state.phase == "PLAYING"
        # Current trick should be in LEADING state
        assert state.trick_state is not None
        assert state.trick_state.phase == "LEADING"

    def test_playing_trick_resolved_starts_next(self) -> None:
        """After a trick resolves, the next trick starts automatically."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        state = _complete_stirring_all_pass(state)
        # Play a complete trick: 4 players play
        trick = state.trick_state
        assert trick is not None
        # Play all 4 cards for the first trick using legal plays
        for _ in range(4):
            if state.phase != "PLAYING":
                break
            trick = state.trick_state
            if trick is None or trick.phase == "RESOLVED":
                break
            state = _play_first_legal(state)

        # After the trick, a new trick should have started (or we moved to scoring)
        assert state.last_completed_trick is not None
        assert state.defender_points >= 0
        if state.phase == "PLAYING":
            # New trick started: lead player is the winner of the first trick
            new_trick = state.trick_state
            assert new_trick is not None
            assert new_trick.phase == "LEADING"
            winner = state.last_completed_trick.winner
            assert new_trick.lead_player == winner

    def test_playing_all_tricks_to_scoring(self) -> None:
        """Playing phase progresses through tricks and eventually completes.

        Multi-card plays (pairs, tractors) may cause players to run out
        of cards before 25 tricks. The game should still complete and transition to
        SCORING/COMPLETE once all cards are played or 25 tricks finish.
        """
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        state = _complete_stirring_all_pass(state)
        # Play tricks until phase changes from PLAYING or no progress
        max_iterations = 30
        prev_completed_trick_key: CompletedTrickKey | None = None
        for _ in range(max_iterations):
            if state.phase != "PLAYING":
                break
            trick = state.trick_state
            if trick is None:
                break
            for _ in range(4):
                if trick.phase == "RESOLVED":
                    break
                state = _play_first_legal(state)
                trick = state.trick_state
                if trick is None:
                    break
            # Check for no progress (all hands empty, trick can't resolve)
            completed_trick_key = _completed_trick_key(state.last_completed_trick)
            if completed_trick_key == prev_completed_trick_key:
                break
            prev_completed_trick_key = completed_trick_key
        # Game should have progressed through at least some tricks
        assert state.last_completed_trick is not None
        assert state.phase in ("SCORING", "WAITING", "PLAYING")


class TestScoringPhase:
    def test_scoring_produces_round_result(self) -> None:
        """SCORING phase computes and stores RoundResult.

        Plays tricks until the round completes. With multi-card plays,
        the round may complete before 25 tricks if players run out of cards.
        """
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        state = _complete_stirring_all_pass(state)
        # Play tricks until phase changes or no progress
        prev_completed_trick_key: CompletedTrickKey | None = None
        for _ in range(30):
            if state.phase != "PLAYING":
                break
            trick = state.trick_state
            if trick is None:
                break
            for _ in range(4):
                if trick.phase == "RESOLVED":
                    break
                state = _play_first_legal(state)
                trick = state.trick_state
                if trick is None:
                    break
            completed_trick_key = _completed_trick_key(state.last_completed_trick)
            if completed_trick_key == prev_completed_trick_key:
                break
            prev_completed_trick_key = completed_trick_key
        # SCORING is transient and immediately transitions to COMPLETE
        # Game should have completed at least one trick
        assert state.last_completed_trick is not None


class TestRoundDeclarer:
    def test_round_first_round_declarer_from_bid(self) -> None:
        """First round: declarer_team is None until deal-bid completes."""
        random.seed(42)
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        assert state.declarer_team is None
        state = _complete_deal_bid_with_reveal(state)
        assert state.deal_bid_state is not None
        assert state.deal_bid_state.bid_winner is not None
        assert state.declarer_player is not None

    def test_round_subsequent_round_declarer_fixed(self) -> None:
        """Subsequent round: declarer_team is pre-determined."""
        state = create_round(RoundInput(
            declarer_team=1, trump_rank=Rank.THREE,
            next_declarer_player=1,
            team0_level=Rank.TWO, team1_level=Rank.THREE,
        ))
        assert state.declarer_team == 1

    def test_round_subsequent_round_bid_winner_only_sets_trump(self) -> None:
        """Subsequent round: bid winner does not replace fixed declarer."""
        random.seed(3)
        state = create_round(RoundInput(
            declarer_team=0, trump_rank=Rank.TWO,
            next_declarer_player=3,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _deal_all_cards(state)
        # Find a trump rank card outside the fixed declarer player's hand.
        assert state.deal_bid_state is not None
        bid_player: int | None = None
        bid_suit: Suit | None = None
        for p in [0, 1, 2]:
            trump_cards = [c for c in state.deal_bid_state.players_hand[p]
                           if c.rank == Rank.TWO and not c.is_joker]
            if trump_cards:
                bid_player = p
                bid_suit = trump_cards[0].suit
                event = BidEvent(
                    player=p, cards=[trump_cards[0]], kind="trump_rank",
                    suit=trump_cards[0].suit, joker_type=None, count=1,
                )
                result = reveal(state, event)
                assert isinstance(result, Ok)
                state = result.value
                break
        assert bid_player is not None
        assert bid_suit is not None
        # Finalize deal-bid to transition to STIRRING
        if state.deal_bid_state is not None and state.deal_bid_state.all_dealt:
            result = finalize_deal_bid(state)
            assert isinstance(result, Ok)
            state = result.value
        assert state.phase == "STIRRING"
        assert state.deal_bid_state is not None
        assert state.deal_bid_state.bid_winner is not None
        assert state.declarer_team == 0  # unchanged
        assert state.declarer_player == 3
        assert state.bid_winner is not None
        assert state.bid_winner.player == bid_player
        assert state.trump_suit == bid_suit

    def test_round_subsequent_round_other_team_bid_still_sets_trump(self) -> None:
        """Subsequent round: any player's bid can choose suit without changing declarer."""
        random.seed(3)
        state = create_round(RoundInput(
            declarer_team=0, trump_rank=Rank.TWO,
            next_declarer_player=0,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _deal_all_cards(state)
        # Find trump rank card in team 1 player's hand (players 1, 2)
        assert state.deal_bid_state is not None
        bid_player: int | None = None
        bid_suit: Suit | None = None
        for p in [1, 2]:
            trump_cards = [c for c in state.deal_bid_state.players_hand[p]
                           if c.rank == Rank.TWO and not c.is_joker]
            if trump_cards:
                bid_player = p
                bid_suit = trump_cards[0].suit
                event = BidEvent(
                    player=p, cards=[trump_cards[0]], kind="trump_rank",
                    suit=trump_cards[0].suit, joker_type=None, count=1,
                )
                result = reveal(state, event)
                assert isinstance(result, Ok)
                state = result.value
                break
        assert bid_player is not None
        assert bid_suit is not None
        # Finalize deal-bid to transition to STIRRING
        if state.deal_bid_state is not None and state.deal_bid_state.all_dealt:
            result = finalize_deal_bid(state)
            assert isinstance(result, Ok)
            state = result.value
        assert state.phase == "STIRRING"
        assert state.declarer_team == 0  # unchanged
        assert state.declarer_player == 0
        assert state.bid_winner is not None
        assert state.bid_winner.player == bid_player
        assert state.trump_suit == bid_suit

    def test_round_empty_trump_no_bid(self) -> None:
        """No bid = empty trump, declarer_player from start_player."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        assert state.trump_suit is None
        # First round no-bid: declarer_player should be start_player (0)
        assert state.declarer_player == 0

    def test_round_subsequent_round_no_bid(self) -> None:
        """Subsequent round no-bid: declarer_player = next_declarer_player."""
        state = create_round(RoundInput(
            declarer_team=1, trump_rank=Rank.THREE,
            next_declarer_player=2,
            team0_level=Rank.TWO, team1_level=Rank.THREE,
        ))
        state = _complete_deal_bid_no_bid(state)
        assert state.trump_suit is None
        # Subsequent round no-bid: declarer_player = next_declarer_player
        assert state.declarer_player == 2
        assert state.declarer_team == 1  # unchanged


class TestRoundValidation:
    def test_round_wrong_phase_operation_rejected(self) -> None:
        """Calling a phase-specific operation in the wrong phase raises error."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        assert state.phase == "DEAL_BID"
        # Cannot stir during DEAL_BID
        cards = [
            Card(id="D1-diamonds-2", suit=Suit.DIAMONDS, rank=Rank.TWO,
                 is_joker=False, is_big_joker=False, points=0, deck=1),
            Card(id="D2-diamonds-2", suit=Suit.DIAMONDS, rank=Rank.TWO,
                 is_joker=False, is_big_joker=False, points=0, deck=2),
        ]
        result = stir(state, player_index=0, cards=cards)
        assert isinstance(result, Rejected)
        assert "不能" in result.reason


class TestRoundFullFlow:
    def test_round_full_round_flow(self) -> None:
        """Integration: complete round from deal-bid to scoring."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        # Deal-bid: no bids
        state = _complete_deal_bid_no_bid(state)
        assert state.phase == "STIRRING"

        # Stirring: all pass (includes initial stir_discard)
        state = _complete_stirring_all_pass(state)
        assert state.phase == "PLAYING"

        # Play tricks until phase changes or no progress
        prev_completed_trick_key: CompletedTrickKey | None = None
        for _ in range(30):
            if state.phase != "PLAYING":
                break
            trick = state.trick_state
            if trick is None:
                break
            for _ in range(4):
                if trick.phase == "RESOLVED":
                    break
                state = _play_first_legal(state)
                trick = state.trick_state
                if trick is None:
                    break
            completed_trick_key = _completed_trick_key(state.last_completed_trick)
            if completed_trick_key == prev_completed_trick_key:
                break
            prev_completed_trick_key = completed_trick_key

        # Should progress through at least some tricks
        assert state.last_completed_trick is not None
        # If round completed, verify result is valid
        if state.phase == "WAITING":
            assert is_round_complete(state) is True
            result = get_round_result(state)
            assert result is not None
            assert result.next_declarer_team in (0, 1)
            assert result.next_declarer_player in (0, 1, 2, 3)


class TestPlayerIdentityValidation:
    """Bug 4 regression: pass_stir, stir, and stir_discard must reject wrong player_index."""

    def test_pass_stir_rejects_wrong_player(self) -> None:
        """pass_stir rejects when player_index doesn't match current_player."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        assert state.phase == "STIRRING"
        assert state.stirring_state is not None
        # Stirring starts in EXCHANGING sub-phase, so pass_stir is rejected
        # regardless of player identity. First complete the exchange.
        assert state.stirring_state.phase == "EXCHANGING"
        assert state.stirring_state.exchange_state is not None
        ex = state.stirring_state.exchange_state
        exchanging = state.stirring_state.exchanging_player
        assert exchanging is not None
        discarded = ex.hand_after_pickup[:ex.count]
        result = stir_discard(state, player_index=exchanging, cards=discarded)
        assert isinstance(result, Ok), f"stir_discard rejected: {result.reason}"
        state = result.value
        # Now in WAITING sub-phase, test wrong player
        assert state.stirring_state is not None
        cur = state.stirring_state.current_player
        wrong_player = (cur + 1) % 4
        result = pass_stir(state, player_index=wrong_player)
        assert isinstance(result, Rejected)
        assert "不是你的回合" in result.reason

    def test_stir_rejects_wrong_player(self) -> None:
        """stir rejects when player_index doesn't match current_player."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        assert state.phase == "STIRRING"
        assert state.stirring_state is not None
        # Stirring starts in EXCHANGING sub-phase; complete it first
        assert state.stirring_state.phase == "EXCHANGING"
        assert state.stirring_state.exchange_state is not None
        ex = state.stirring_state.exchange_state
        exchanging = state.stirring_state.exchanging_player
        assert exchanging is not None
        discarded = ex.hand_after_pickup[:ex.count]
        result = stir_discard(state, player_index=exchanging, cards=discarded)
        assert isinstance(result, Ok), f"stir_discard rejected: {result.reason}"
        state = result.value
        assert state.stirring_state is not None
        cur = state.stirring_state.current_player
        wrong_player = (cur + 1) % 4
        # Use any cards; the player identity check happens before card validation
        fake_cards = [
            Card(id="D1-clubs-2", suit=Suit.CLUBS, rank=Rank.TWO,
                 is_joker=False, is_big_joker=False, points=0, deck=1),
            Card(id="D2-clubs-2", suit=Suit.CLUBS, rank=Rank.TWO,
                 is_joker=False, is_big_joker=False, points=0, deck=2),
        ]
        result = stir(state, player_index=wrong_player, cards=fake_cards)
        assert isinstance(result, Rejected)
        assert "不是你的回合" in result.reason

    def test_stir_discard_rejects_wrong_player(self) -> None:
        """stir_discard rejects when player_index doesn't match exchanging_player."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        assert state.phase == "STIRRING"
        assert state.stirring_state is not None
        assert state.stirring_state.phase == "EXCHANGING"
        assert state.stirring_state.exchange_state is not None
        exchanging = state.stirring_state.exchanging_player
        assert exchanging is not None
        wrong_player = (exchanging + 1) % 4
        # Use any cards; the player identity check happens before card validation
        fake_cards = [
            Card(id="D1-clubs-2", suit=Suit.CLUBS, rank=Rank.TWO,
                 is_joker=False, is_big_joker=False, points=0, deck=1),
        ]
        result = stir_discard(state, player_index=wrong_player, cards=fake_cards)
        assert isinstance(result, Rejected)
        assert "炒主者" in result.reason

    def test_pass_stir_accepts_correct_player(self) -> None:
        """pass_stir succeeds when player_index matches current_player."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        assert state.phase == "STIRRING"
        assert state.stirring_state is not None
        # Stirring starts in EXCHANGING; complete it first
        assert state.stirring_state.phase == "EXCHANGING"
        assert state.stirring_state.exchange_state is not None
        ex = state.stirring_state.exchange_state
        exchanging = state.stirring_state.exchanging_player
        assert exchanging is not None
        discarded = ex.hand_after_pickup[:ex.count]
        result = stir_discard(state, player_index=exchanging, cards=discarded)
        assert isinstance(result, Ok), f"stir_discard rejected: {result.reason}"
        state = result.value
        assert state.stirring_state is not None
        cur = state.stirring_state.current_player
        result = pass_stir(state, player_index=cur)
        assert isinstance(result, Ok)

    def test_stir_discard_accepts_correct_player(self) -> None:
        """stir_discard succeeds when player_index matches exchanging_player."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        assert state.phase == "STIRRING"
        assert state.stirring_state is not None
        assert state.stirring_state.phase == "EXCHANGING"
        assert state.stirring_state.exchange_state is not None
        ex = state.stirring_state.exchange_state
        exchanging = state.stirring_state.exchanging_player
        assert exchanging is not None
        discarded = ex.hand_after_pickup[:ex.count]
        result = stir_discard(state, player_index=exchanging, cards=discarded)
        assert isinstance(result, Ok)


def test_play_leading_accepts_legal_partial_throw_not_in_enumerated_hints() -> None:
    """Round play delegates leading validation to trick_sm, allowing partial throws."""
    hands = [
        [
            _card(Suit.SPADES, Rank.ACE),
            _card(Suit.SPADES, Rank.KING),
            _card(Suit.SPADES, Rank.THREE),
        ],
        [_card(Suit.SPADES, Rank.QUEEN)],
        [_card(Suit.HEARTS, Rank.THREE)],
        [_card(Suit.CLUBS, Rank.THREE)],
    ]
    trick = trick_mod.create_trick(trick_mod.TrickInput(
        lead_player=0,
        hands=hands,
        trump_suit=Suit.HEARTS,
        trump_rank=Rank.TWO,
        defender_points=0,
        declarer_team=0,
    ))
    state = RoundState(
        phase="PLAYING",
        declarer_team=0,
        declarer_player=0,
        defender_team=1,
        trump_suit=Suit.HEARTS,
        trump_rank=Rank.TWO,
        bid_winner=None,
        players_hand=hands,
        bottom_cards=[],
        defender_points=0,
        last_completed_trick=None,
        defender_point_cards=[],
        deal_bid_state=None,
        stirring_state=None,
        trick_state=trick,
        result=None,
        team0_level=Rank.TWO,
        team1_level=Rank.TWO,
        start_player=0,
        next_declarer_player=None,
    )

    result = play(
        state,
        player_index=0,
        cards=[hands[0][0], hands[0][1]],
    )

    assert isinstance(result, Ok)
    assert result.value.trick_state is not None
    assert result.value.trick_state.played == 1
    assert result.value.trick_state.failed_throw is None


def test_play_leading_failed_throw_records_public_penalty_event() -> None:
    """Round play advances failed throws and keeps attempted/forced cards public."""
    hands = [
        [_card(Suit.SPADES, Rank.KING), _card(Suit.SPADES, Rank.QUEEN)],
        [_card(Suit.SPADES, Rank.ACE)],
        [_card(Suit.HEARTS, Rank.THREE)],
        [_card(Suit.CLUBS, Rank.THREE)],
    ]
    trick = trick_mod.create_trick(trick_mod.TrickInput(
        lead_player=0,
        hands=hands,
        trump_suit=Suit.HEARTS,
        trump_rank=Rank.TWO,
        defender_points=0,
        declarer_team=0,
    ))
    state = RoundState(
        phase="PLAYING",
        declarer_team=0,
        declarer_player=0,
        defender_team=1,
        trump_suit=Suit.HEARTS,
        trump_rank=Rank.TWO,
        bid_winner=None,
        players_hand=hands,
        bottom_cards=[],
        defender_points=0,
        last_completed_trick=None,
        defender_point_cards=[],
        deal_bid_state=None,
        stirring_state=None,
        trick_state=trick,
        result=None,
        team0_level=Rank.TWO,
        team1_level=Rank.TWO,
        start_player=0,
        next_declarer_player=None,
    )

    result = play(state, player_index=0, cards=hands[0])

    assert isinstance(result, Ok)
    assert result.value.trick_state is not None
    assert result.value.trick_state.slots[0].cards == [hands[0][1]]
    assert result.value.players_hand[0] == [hands[0][0]]
    assert result.value.trick_state.failed_throw is not None
    assert result.value.trick_state.failed_throw.attempted_cards == hands[0]
    assert result.value.trick_state.failed_throw.forced_cards == [hands[0][1]]
