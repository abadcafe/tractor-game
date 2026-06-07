"""Tests for sm.round_sm module."""
import random
import pytest
from server.sm.card_model import Card, Suit, Rank
from server.sm.types import BidEvent, PlayAction
from server.sm.round_sm import (
    RoundState, RoundInput, create_round,
    deal_next_card, reveal, pass_stir, stir, discard, play,
    is_round_complete, get_round_result,
)


def _play_first_legal(state: RoundState) -> RoundState:
    """Play the first legal play for the current player in the trick.

    Uses get_legal_plays to find a valid play instead of blindly playing
    hand[0], which may violate follow-suit rules and cause ValueError.
    """
    from server.sm.play_rules import get_legal_plays

    trick = state.trick_state
    assert trick is not None
    cur = trick.cur
    hand = trick.hands[cur]
    if not hand:
        return state

    # Determine if leading or following
    is_leading = trick.phase == "LEADING"
    if is_leading:
        lead_action: PlayAction | None = None
    else:
        # Build the lead action from the lead player's slot
        lead_slot = trick.slots[trick.lead_player]
        assert lead_slot is not None and lead_slot.cards is not None
        lead_cards = lead_slot.cards
        lead_action = PlayAction(type=trick.lead_type, cards=lead_cards)

    legal_plays = get_legal_plays(
        hand=hand,
        is_leading=is_leading,
        lead_action=lead_action,
        trump_suit=state.trump_suit,
        trump_rank=state.trump_rank,
    )
    assert len(legal_plays) > 0, f"No legal plays for player {cur}"
    return play(state, cards=legal_plays[0].cards)


def _deal_all_cards(state: RoundState) -> RoundState:
    """Deal all 100 cards in the deal-bid phase."""
    while state.phase == "DEAL_BID" and state.deal_bid_state is not None:
        if state.deal_bid_state.phase != "DEALING":
            break
        state = deal_next_card(state)
    return state


def _complete_deal_bid_no_bid(state: RoundState) -> RoundState:
    """Complete deal-bid without any reveals (results in NO_BID)."""
    state = _deal_all_cards(state)
    return state


def _complete_deal_bid_with_reveal(state: RoundState) -> RoundState:
    """Complete deal-bid with one reveal (results in COMPLETE)."""
    # Deal some cards first
    for _ in range(20):
        if state.deal_bid_state is None or state.deal_bid_state.phase != "DEALING":
            break
        state = deal_next_card(state)

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
                state = reveal(state, event)
                break

    # Deal remaining cards
    state = _deal_all_cards(state)
    return state


def _complete_stirring_all_pass(state: RoundState) -> RoundState:
    """Complete stirring by having all players pass."""
    for _ in range(4):
        if state.phase != "STIRRING":
            break
        state = pass_stir(state)
    return state


def _complete_exchange(state: RoundState) -> RoundState:
    """Complete exchange by discarding bottom cards back."""
    if state.exchange_state is None:
        return state
    discarded = state.exchange_state.hand_after_pickup[:state.exchange_state.count]
    state = discard(state, discarded)
    return state


class TestCreateRound:
    def test_create_round_initial_state(self) -> None:
        """Initial round state: DEAL_BID phase."""
        state = create_round(RoundInput(
            declarer_team=None,
            trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        ))
        assert state.phase == "DEAL_BID"
        assert state.declarer_team is None
        assert state.trump_rank == Rank.TWO
        assert state.deal_bid_state is not None

    def test_create_round_with_declarer(self) -> None:
        """Subsequent round with known declarer_team."""
        state = create_round(RoundInput(
            declarer_team=0,
            trump_rank=Rank.THREE,
            last_declarer_player=0,
            team0_level=Rank.THREE,
            team1_level=Rank.TWO,
        ))
        assert state.declarer_team == 0
        assert state.trump_rank == Rank.THREE


class TestDealBidPhase:
    def test_deal_next_card_advances_deal_bid(self) -> None:
        """deal_next_card during DEAL_BID advances the deal-bid sub-state."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        assert state.phase == "DEAL_BID"
        initial_cursor = state.deal_bid_state.deal_cursor
        state = deal_next_card(state)
        assert state.deal_bid_state.deal_cursor == initial_cursor + 1

    def test_reveal_during_deal_bid(self) -> None:
        """reveal during DEAL_BID adds a bid event."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        # Deal some cards first
        for _ in range(20):
            state = deal_next_card(state)

        # Find a trump rank card and reveal
        for p in range(4):
            trump_cards = [c for c in state.deal_bid_state.players_hand[p]
                          if c.rank == Rank.TWO and not c.is_joker]
            if trump_cards:
                event = BidEvent(
                    player=p, cards=[trump_cards[0]], kind="trump_rank",
                    suit=trump_cards[0].suit, joker_type=None, count=1,
                )
                old_events = len(state.deal_bid_state.bid_events)
                state = reveal(state, event)
                assert len(state.deal_bid_state.bid_events) > old_events
                break

    def test_deal_bid_to_stirring_with_winner(self) -> None:
        """After deal-bid completes with a winner, round enters STIRRING."""
        random.seed(42)
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_with_reveal(state)
        assert state.deal_bid_state is not None
        assert state.deal_bid_state.bid_winner is not None
        assert state.phase == "STIRRING"
        assert state.declarer_player is not None
        assert state.trump_suit is not None

    def test_deal_bid_to_stirring_no_bid(self) -> None:
        """After deal-bid with no bids, round enters STIRRING with empty trump."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        assert state.phase == "STIRRING"
        assert state.trump_suit is None  # empty trump


class TestStirringPhase:
    def test_pass_stir_during_stirring(self) -> None:
        """pass_stir during STIRRING advances current player."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        assert state.phase == "STIRRING"
        state = pass_stir(state)
        assert state.stirring_state is not None
        assert len(state.stirring_state.pass_set) == 1

    def test_stir_during_stirring(self) -> None:
        """stir during STIRRING changes trump suit."""
        random.seed(10)
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        assert state.phase == "STIRRING"
        # With empty trump, find a trump-rank pair in the current player's hand
        cur = state.stirring_state.current_player
        hand = state.players_hand[cur]
        # Find two cards of the same suit with trump rank
        from collections import Counter
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
        state = stir(state, cards=pair)
        assert state.trump_suit == target_suit

    def test_stir_cards_not_in_hand_rejected(self) -> None:
        """stir with cards not in current player's hand is rejected."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        assert state.phase == "STIRRING"
        # Fabricate cards that are NOT in the current player's hand
        fake_cards = [
            Card(id="D1-clubs-2", suit=Suit.CLUBS, rank=Rank.TWO,
                 is_joker=False, is_big_joker=False, points=0, deck=1),
            Card(id="D2-clubs-2", suit=Suit.CLUBS, rank=Rank.TWO,
                 is_joker=False, is_big_joker=False, points=0, deck=2),
        ]
        with pytest.raises(ValueError, match="not in hand"):
            stir(state, cards=fake_cards)

    def test_stirring_to_exchange(self) -> None:
        """After all players pass stirring, round enters EXCHANGE."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        state = _complete_stirring_all_pass(state)
        assert state.phase == "EXCHANGE"
        assert state.exchange_state is not None


class TestExchangePhase:
    def test_discard_during_exchange(self) -> None:
        """discard during EXCHANGE transitions to PLAYING."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        state = _complete_stirring_all_pass(state)
        assert state.phase == "EXCHANGE"
        state = _complete_exchange(state)
        assert state.phase == "PLAYING"

    def test_exchange_to_playing(self) -> None:
        """After exchange completes, round enters PLAYING."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        state = _complete_stirring_all_pass(state)
        state = _complete_exchange(state)
        assert state.phase == "PLAYING"
        assert state.trick_state is not None


class TestPlayingPhase:
    def test_play_during_playing_first_trick(self) -> None:
        """First play during PLAYING is the lead player's turn."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        state = _complete_stirring_all_pass(state)
        state = _complete_exchange(state)
        assert state.phase == "PLAYING"
        # Current trick should be in LEADING state
        assert state.trick_state is not None
        assert state.trick_state.phase == "LEADING"

    def test_playing_trick_resolved_starts_next(self) -> None:
        """After a trick resolves, the next trick starts automatically."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        state = _complete_stirring_all_pass(state)
        state = _complete_exchange(state)
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
        assert len(state.trick_history) == 1
        assert state.defender_points >= 0
        if state.phase == "PLAYING":
            # New trick started: lead player is the winner of the first trick
            new_trick = state.trick_state
            assert new_trick is not None
            assert new_trick.phase == "LEADING"
            winner = state.trick_history[0].winner
            assert new_trick.lead_player == winner

    def test_playing_all_tricks_to_scoring(self) -> None:
        """After all tricks are played, round enters SCORING."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        state = _complete_stirring_all_pass(state)
        state = _complete_exchange(state)
        # Play all 25 tricks
        for _ in range(25):
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
        # SCORING is transient and immediately transitions to COMPLETE
        assert state.phase in ("SCORING", "COMPLETE")
        assert len(state.trick_history) == 25


class TestScoringPhase:
    def test_scoring_produces_round_result(self) -> None:
        """SCORING phase computes and stores RoundResult."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        state = _complete_stirring_all_pass(state)
        state = _complete_exchange(state)
        # Play all tricks quickly
        for _ in range(25):
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
        # SCORING is transient and immediately transitions to COMPLETE
        assert state.phase == "COMPLETE"
        assert is_round_complete(state) is True
        result = get_round_result(state)
        assert result is not None
        assert result.next_declarer_team in (0, 1)
        assert result.next_declarer_player in (0, 1, 2, 3)
        assert isinstance(result.total_defender_points, int)
        assert isinstance(result.bottom_card_bonus, int)


class TestRoundDeclarer:
    def test_round_first_round_declarer_from_bid(self) -> None:
        """First round: declarer_team is None until deal-bid completes."""
        random.seed(42)
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
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
            last_declarer_player=1,
            team0_level=Rank.TWO, team1_level=Rank.THREE,
        ))
        assert state.declarer_team == 1

    def test_round_subsequent_round_bid_winner_on_team(self) -> None:
        """Subsequent round: bid winner on declarer_team sets declarer_player."""
        random.seed(3)
        state = create_round(RoundInput(
            declarer_team=0, trump_rank=Rank.TWO,
            last_declarer_player=0,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        # Deal some cards
        for _ in range(20):
            state = deal_next_card(state)
        # Find trump rank card in team 0 player's hand (players 0, 3)
        for p in [0, 3]:
            trump_cards = [c for c in state.deal_bid_state.players_hand[p]
                           if c.rank == Rank.TWO and not c.is_joker]
            if trump_cards:
                event = BidEvent(
                    player=p, cards=[trump_cards[0]], kind="trump_rank",
                    suit=trump_cards[0].suit, joker_type=None, count=1,
                )
                state = reveal(state, event)
                break
        state = _deal_all_cards(state)
        assert state.phase == "STIRRING"
        assert state.deal_bid_state.bid_winner is not None
        assert state.declarer_team == 0  # unchanged
        assert state.declarer_player is not None

    def test_round_subsequent_round_bid_winner_wrong_team(self) -> None:
        """Subsequent round: bid winner NOT on declarer_team triggers fallback."""
        random.seed(3)
        state = create_round(RoundInput(
            declarer_team=0, trump_rank=Rank.TWO,
            last_declarer_player=2,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        # Deal some cards
        for _ in range(20):
            state = deal_next_card(state)
        # Find trump rank card in team 1 player's hand (players 1, 2)
        for p in [1, 2]:
            trump_cards = [c for c in state.deal_bid_state.players_hand[p]
                           if c.rank == Rank.TWO and not c.is_joker]
            if trump_cards:
                event = BidEvent(
                    player=p, cards=[trump_cards[0]], kind="trump_rank",
                    suit=trump_cards[0].suit, joker_type=None, count=1,
                )
                state = reveal(state, event)
                break
        state = _deal_all_cards(state)
        assert state.phase == "STIRRING"
        # Winner is on team 1 but declarer_team is 0 -> fallback
        assert state.declarer_team == 0  # unchanged
        assert state.declarer_player == 2  # falls back to last_declarer_player
        assert state.trump_suit is None  # fallback sets trump_suit to None

    def test_round_empty_trump_no_bid(self) -> None:
        """No bid = empty trump, declarer_player from start_player."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        state = _complete_deal_bid_no_bid(state)
        assert state.trump_suit is None
        # First round no-bid: declarer_player should be start_player (0)
        assert state.declarer_player == 0

    def test_round_subsequent_round_no_bid(self) -> None:
        """Subsequent round no-bid: declarer_player = last_declarer_player."""
        state = create_round(RoundInput(
            declarer_team=1, trump_rank=Rank.THREE,
            last_declarer_player=2,
            team0_level=Rank.TWO, team1_level=Rank.THREE,
        ))
        state = _complete_deal_bid_no_bid(state)
        assert state.trump_suit is None
        # Subsequent round no-bid: declarer_player = last_declarer_player
        assert state.declarer_player == 2
        assert state.declarer_team == 1  # unchanged


class TestRoundValidation:
    def test_round_wrong_phase_operation_rejected(self) -> None:
        """Calling a phase-specific operation in the wrong phase raises error."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
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
        with pytest.raises(ValueError, match="phase"):
            stir(state, cards=cards)


class TestRoundFullFlow:
    def test_round_full_round_flow(self) -> None:
        """Integration: complete round from deal-bid to scoring."""
        state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        # Deal-bid: no bids
        state = _complete_deal_bid_no_bid(state)
        assert state.phase == "STIRRING"

        # Stirring: all pass
        state = _complete_stirring_all_pass(state)
        assert state.phase == "EXCHANGE"

        # Exchange: discard
        state = _complete_exchange(state)
        assert state.phase == "PLAYING"

        # Play all 25 tricks
        for _ in range(25):
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

        # Should be COMPLETE after all 25 tricks
        assert state.phase == "COMPLETE"
        assert is_round_complete(state) is True
        result = get_round_result(state)
        assert result is not None
        assert result.next_declarer_team in (0, 1)
        assert result.next_declarer_player in (0, 1, 2, 3)
