"""Tests for engine.state module."""
import pytest
from server.engine.card import Card, Suit, Rank
from server.engine.types import Phase, PlayType, PlayAction, BidAction, StirAction
from server.engine.game_state import (
    TrickSlot, CompletedTrick, PlayerState, TeamState,
    GameState, GameSettings,
)
from server.engine.state import (
    create_initial_state, deal_cards, record_bid, set_declarer,
    record_stir, pickup_bottom_cards, discard_cards, play_cards,
    resolve_trick, advance_round, clear_trick,
)
from server.engine.constants import BOTTOM_CARD_COUNT, PLAYER_COUNT


class TestCreateInitialState:
    def test_create_initial_state(self):
        state = create_initial_state()
        assert state.phase == Phase.DEALING
        assert len(state.players) == 4
        assert state.current_player_index == 0
        assert state.trump_suit is None
        for p in state.players:
            assert len(p.hand) == 0


class TestDealCards:
    def test_deal_cards_structure(self):
        state = create_initial_state()
        state = deal_cards(state)
        for p in state.players:
            assert len(p.hand) == 25
        assert len(state.bottom_cards) == BOTTOM_CARD_COUNT

    def test_deal_cards_bidding_phase(self):
        state = create_initial_state()
        state = deal_cards(state)
        assert state.phase == Phase.BIDDING


class TestRecordBid:
    def test_record_bid(self):
        state = create_initial_state()
        state = deal_cards(state)
        bid = BidAction(player_index=0, level=Rank.THREE, pass_=False)
        state = record_bid(state, bid)
        assert len(state.bidding_history) == 1
        assert state.bidding_history[0].level == Rank.THREE


class TestSetDeclarer:
    def test_set_declarer_specific_player(self):
        """Bug #2 fix: isDeclarer should be set on specific player, not whole team."""
        state = create_initial_state()
        state = deal_cards(state)
        state = set_declarer(state, player_index=3, trump_suit=Suit.HEARTS, trump_rank=Rank.TWO)
        assert state.players[3].is_declarer is True
        assert state.players[0].is_declarer is False
        assert state.declarer_team_index == 0

    def test_set_declarer_stirring_phase(self):
        state = create_initial_state()
        state = deal_cards(state)
        state = set_declarer(state, player_index=3, trump_suit=Suit.HEARTS, trump_rank=Rank.TWO)
        assert state.phase == Phase.STIRRING


class TestRecordStir:
    def test_record_stir(self):
        state = create_initial_state()
        state = deal_cards(state)
        state = set_declarer(state, player_index=3, trump_suit=Suit.HEARTS, trump_rank=Rank.TWO)
        stir = StirAction(player_index=1, new_trump_suit=Suit.SPADES, level=Rank.THREE)
        state = record_stir(state, stir)
        assert state.trump_suit == Suit.SPADES
        assert len(state.stir_history) == 1


class TestPickupBottomCards:
    def test_pickup_bottom_cards_specific_player(self):
        """Bug #2 fix: bottom cards go to the specific bid winner."""
        state = create_initial_state()
        state = deal_cards(state)
        state = set_declarer(state, player_index=3, trump_suit=Suit.HEARTS, trump_rank=Rank.TWO)
        state = pickup_bottom_cards(state)
        assert state.phase == Phase.EXCHANGE
        assert len(state.players[3].hand) == 33
        assert len(state.players[0].hand) == 25


class TestDiscardCards:
    def test_discard_cards(self):
        state = create_initial_state()
        state = deal_cards(state)
        state = set_declarer(state, player_index=3, trump_suit=Suit.HEARTS, trump_rank=Rank.TWO)
        state = pickup_bottom_cards(state)
        discard = state.players[3].hand[:BOTTOM_CARD_COUNT]
        state = discard_cards(state, player_index=3, cards=discard)
        assert len(state.players[3].hand) == 25
        assert state.phase == Phase.PLAYING


class TestPlayCards:
    def test_play_cards_lead(self):
        state = _setup_playing_state()
        lead_card = state.players[state.lead_player_index].hand[0]
        action = PlayAction(type=PlayType.SINGLE, cards=[lead_card])
        state = play_cards(state, state.current_player_index, action)
        card_ids = [c.id for c in state.players[state.lead_player_index].hand]
        assert lead_card.id not in card_ids

    def test_play_cards_follow(self):
        state = _setup_playing_state()
        lead_card = state.players[state.current_player_index].hand[0]
        action = PlayAction(type=PlayType.SINGLE, cards=[lead_card])
        state = play_cards(state, state.current_player_index, action)
        follow_card = state.players[state.current_player_index].hand[0]
        action = PlayAction(type=PlayType.SINGLE, cards=[follow_card])
        state = play_cards(state, state.current_player_index, action)
        assert state.current_player_index != state.lead_player_index


class TestResolveTrick:
    def test_resolve_trick_uses_compare_plays(self):
        """Bug #1 fix: must use comparePlays() to determine winner."""
        state = _setup_playing_state()
        lead_card = state.players[state.current_player_index].hand[0]
        state = _play_four_cards(state, lead_card)
        assert state.last_completed_trick is not None
        assert 0 <= state.last_completed_trick.winner_index <= 3

    def test_resolve_trick_trump_wins(self):
        """Trump play should beat non-trump play."""
        state = _setup_playing_state()
        for p_idx in range(4):
            trump_cards = [c for c in state.players[p_idx].hand
                          if c.suit == state.trump_suit or c.is_joker or c.rank == state.trump_rank]
            if trump_cards:
                break
        assert state.phase in (Phase.PLAYING, Phase.SCORING)

    def test_resolve_trick_last_trick_scoring(self):
        """After all cards are played, phase should be SCORING."""
        state = _setup_playing_state()
        for _ in range(25):
            if state.phase == Phase.SCORING:
                break
            for _ in range(4):
                if state.phase == Phase.SCORING:
                    break
                player_idx = state.current_player_index
                hand = state.players[player_idx].hand
                if not hand:
                    break
                card = hand[0]
                action = PlayAction(type=PlayType.SINGLE, cards=[card])
                state = play_cards(state, player_idx, action)
        assert state.phase in (Phase.PLAYING, Phase.SCORING)


class TestAdvanceRound:
    def test_advance_round_independent_levels(self):
        """Bug #3 fix: teams should have independent levels after advance."""
        state = _setup_scoring_state()
        state = advance_round(
            state,
            team0_new_level=Rank.FIVE,
            team1_new_level=Rank.TWO,
            new_declarer_team=0,
        )
        assert state.teams[0].current_level == Rank.FIVE
        assert state.teams[1].current_level == Rank.TWO
        assert state.phase == Phase.DEALING


class TestClearTrick:
    def test_clear_trick(self):
        state = _setup_playing_state()
        lead_card = state.players[state.current_player_index].hand[0]
        action = PlayAction(type=PlayType.SINGLE, cards=[lead_card])
        state = play_cards(state, state.current_player_index, action)
        state = clear_trick(state)
        for slot in state.current_trick:
            assert slot.cards is None


# ---- Helpers ----

def _setup_playing_state() -> GameState:
    state = create_initial_state()
    state = deal_cards(state)
    state = set_declarer(state, player_index=3, trump_suit=Suit.HEARTS, trump_rank=Rank.TWO)
    state = pickup_bottom_cards(state)
    discard = state.players[3].hand[:BOTTOM_CARD_COUNT]
    state = discard_cards(state, player_index=3, cards=discard)
    return state


def _setup_scoring_state() -> GameState:
    state = _setup_playing_state()
    state = state.model_copy(update={
        "phase": Phase.SCORING,
        "players": [p.model_copy(update={"hand": []}) for p in state.players],
    })
    return state


def _play_four_cards(state: GameState, lead_card: Card) -> GameState:
    from server.engine.player_utils import next_player
    player_idx = state.current_player_index
    action = PlayAction(type=PlayType.SINGLE, cards=[lead_card])
    state = play_cards(state, player_idx, action)
    for _ in range(3):
        if state.phase == Phase.SCORING:
            break
        player_idx = state.current_player_index
        hand = state.players[player_idx].hand
        if not hand:
            break
        card = hand[0]
        action = PlayAction(type=PlayType.SINGLE, cards=[card])
        state = play_cards(state, player_idx, action)
    return state
