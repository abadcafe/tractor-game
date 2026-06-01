"""Tests for engine.state module."""
import pytest
from server.engine.card import Card, Suit, Rank
from server.engine.types import Phase, PlayType, PlayAction, BidAction, StirAction
from server.engine.game_state import GameState
from server.engine.state import (
    create_initial_state, deal_cards, record_bid, set_declarer,
    record_stir, pickup_bottom_cards, discard_cards, play_cards,
    advance_round, clear_trick,
)
from server.engine.constants import BOTTOM_CARD_COUNT


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
        # CR-008: verify is_declarer set on specific stirring player, not whole team
        assert state.players[1].is_declarer is True
        assert state.players[0].is_declarer is False
        assert state.players[2].is_declarer is False
        assert state.players[3].is_declarer is False


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

    def test_play_cards_wrong_player_raises(self):
        """CR-011: play_cards must reject wrong player_index."""
        state = _setup_playing_state()
        wrong_idx = (state.current_player_index + 1) % 4
        card = state.players[wrong_idx].hand[0]
        action = PlayAction(type=PlayType.SINGLE, cards=[card])
        with pytest.raises(ValueError, match="does not match"):
            play_cards(state, wrong_idx, action)


class TestResolveTrick:
    def test_resolve_trick_uses_compare_plays(self):
        """Bug #1 fix: must use compare_plays() to determine winner.

        Set up deterministic hands: lead plays a low non-trump card (THREE of DIAMONDS),
        follower plays a higher card of the same suit (ACE of DIAMONDS).
        The follower should win, proving compare_plays is actually called.
        Without compare_plays, the lead player would always win (original bug).
        """
        state = _setup_playing_state()
        # trump_suit=HEARTS, trump_rank=TWO
        # Lead card: low non-trump (DIAMONDS THREE, RANK_ORDER=3)
        low_card = Card(id="low1", suit=Suit.DIAMONDS, rank=Rank.THREE,
                        is_joker=False, is_big_joker=False, points=0, deck=1)
        # Follower card: high same-suit non-trump (DIAMONDS ACE, RANK_ORDER=14)
        high_card = Card(id="high1", suit=Suit.DIAMONDS, rank=Rank.ACE,
                         is_joker=False, is_big_joker=False, points=10, deck=1)
        other1 = Card(id="o1", suit=Suit.CLUBS, rank=Rank.FOUR,
                      is_joker=False, is_big_joker=False, points=0, deck=1)
        other2 = Card(id="o2", suit=Suit.SPADES, rank=Rank.FIVE,
                      is_joker=False, is_big_joker=False, points=5, deck=1)

        # Turn order after player 3 leads: 3 -> 1 -> 0 -> 2
        # Player 3 (lead): low card
        # Player 1 (first follower): high card (should win)
        # Player 0, 2: other cards
        new_players = list(state.players)
        new_players[3] = new_players[3].model_copy(update={
            "hand": [low_card] + [c for c in new_players[3].hand[1:]],
        })
        new_players[1] = new_players[1].model_copy(update={
            "hand": [high_card] + [c for c in new_players[1].hand[1:]],
        })
        new_players[0] = new_players[0].model_copy(update={
            "hand": [other1] + [c for c in new_players[0].hand[1:]],
        })
        new_players[2] = new_players[2].model_copy(update={
            "hand": [other2] + [c for c in new_players[2].hand[1:]],
        })
        state = state.model_copy(update={"players": new_players})

        # Play all 4 cards in turn order
        p3_card = state.players[3].hand[0]
        state = play_cards(state, 3, PlayAction(type=PlayType.SINGLE, cards=[p3_card]))

        p1_card = state.players[1].hand[0]  # high card, same suit
        state = play_cards(state, 1, PlayAction(type=PlayType.SINGLE, cards=[p1_card]))

        p0_card = state.players[0].hand[0]  # different suit
        state = play_cards(state, 0, PlayAction(type=PlayType.SINGLE, cards=[p0_card]))

        p2_card = state.players[2].hand[0]  # different suit
        state = play_cards(state, 2, PlayAction(type=PlayType.SINGLE, cards=[p2_card]))

        assert state.last_completed_trick is not None
        # Player 1 played the highest card of the lead suit -> should win
        assert state.last_completed_trick.winner_index == 1

    def test_resolve_trick_trump_wins(self):
        """Trump play should beat non-trump play.

        Uses deterministic hands to ensure only one player has a trump card.
        trump_suit=HEARTS, trump_rank=TWO.
        """
        state = _setup_playing_state()
        lead_idx = state.lead_player_index  # player 3

        # Build deterministic hands: player 3 leads a non-trump card,
        # player 1 (next after 3) has one trump card, rest are non-trump.
        # trump_suit=HEARTS, trump_rank=TWO
        # Trump = joker or suit=HEARTS or rank=TWO
        non_trump = Card(id="nt1", suit=Suit.DIAMONDS, rank=Rank.ACE,
                         is_joker=False, is_big_joker=False, points=10, deck=1)
        non_trump2 = Card(id="nt2", suit=Suit.CLUBS, rank=Rank.THREE,
                          is_joker=False, is_big_joker=False, points=0, deck=1)
        non_trump3 = Card(id="nt3", suit=Suit.SPADES, rank=Rank.FOUR,
                          is_joker=False, is_big_joker=False, points=0, deck=1)
        non_trump4 = Card(id="nt4", suit=Suit.DIAMONDS, rank=Rank.FIVE,
                          is_joker=False, is_big_joker=False, points=5, deck=1)
        # Trump card: hearts + non-trump-rank
        trump = Card(id="tr1", suit=Suit.HEARTS, rank=Rank.THREE,
                     is_joker=False, is_big_joker=False, points=0, deck=1)

        # Map players by turn order: lead=3 -> 1 -> 0 -> 2
        # Player 3 (lead): all non-trump
        # Player 1: one trump + rest non-trump
        # Player 0: all non-trump
        # Player 2: all non-trump
        new_players = list(state.players)
        new_players[3] = new_players[3].model_copy(update={
            "hand": [non_trump] + [c for c in new_players[3].hand[1:]],
        })
        new_players[1] = new_players[1].model_copy(update={
            "hand": [trump, non_trump2] + [c for c in new_players[1].hand[2:]],
        })
        new_players[0] = new_players[0].model_copy(update={
            "hand": [non_trump3] + [c for c in new_players[0].hand[1:]],
        })
        new_players[2] = new_players[2].model_copy(update={
            "hand": [non_trump4] + [c for c in new_players[2].hand[1:]],
        })
        state = state.model_copy(update={"players": new_players})

        # Play all 4 cards in turn order
        p3_card = state.players[3].hand[0]
        state = play_cards(state, 3, PlayAction(type=PlayType.SINGLE, cards=[p3_card]))

        p1_card = state.players[1].hand[0]  # trump card
        state = play_cards(state, 1, PlayAction(type=PlayType.SINGLE, cards=[p1_card]))

        p0_card = state.players[0].hand[0]  # non-trump
        state = play_cards(state, 0, PlayAction(type=PlayType.SINGLE, cards=[p0_card]))

        p2_card = state.players[2].hand[0]  # non-trump
        state = play_cards(state, 2, PlayAction(type=PlayType.SINGLE, cards=[p2_card]))

        assert state.last_completed_trick is not None
        # Player 1 played trump, should win
        assert state.last_completed_trick.winner_index == 1

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
        assert state.phase == Phase.SCORING


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
        # SR-001: after advance_round, no player should be declarer
        # (specific declarer not yet determined for new round)
        for p in state.players:
            assert p.is_declarer is False


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
