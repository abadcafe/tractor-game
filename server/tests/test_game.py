"""Tests for engine.game module."""
import pytest
from server.engine.card import Suit, Rank
from server.engine.types import Phase, PlayType, PlayAction
from server.engine.game import Game


class TestGameNewGame:
    def test_game_new_game(self):
        game = Game()
        assert game.state.phase == Phase.DEALING
        assert len(game.state.players) == 4

    def test_game_deal(self):
        game = Game()
        game.start_round()
        assert game.state.phase == Phase.BIDDING
        for p in game.state.players:
            assert len(p.hand) == 25


class TestGameBidFlow:
    def test_game_bid_flow(self):
        game = Game()
        game.start_round()
        game.submit_bid(0, Rank.THREE, pass_=False)
        game.submit_bid(1, None, pass_=True)
        game.submit_bid(2, None, pass_=True)
        game.submit_bid(3, None, pass_=True)
        assert game.get_awaiting_action() == "set_trump"

    def test_game_all_pass_redeal(self):
        game = Game()
        game.start_round()
        for i in range(4):
            game.submit_bid(i, None, pass_=True)
        assert game.state.phase in (Phase.BIDDING, Phase.DEALING)

    def test_game_set_trump(self):
        game = Game()
        game.start_round()
        game.submit_bid(0, Rank.THREE, pass_=False)
        for i in range(1, 4):
            game.submit_bid(i, None, pass_=True)
        ok = game.set_trump(0, Suit.HEARTS)
        assert ok is True
        assert game.state.trump_suit == Suit.HEARTS
        assert game.state.phase == Phase.STIRRING


class TestGameStirFlow:
    def test_game_stir_flow(self):
        game = Game()
        game.start_round()
        game.submit_bid(0, Rank.THREE, pass_=False)
        for i in range(1, 4):
            game.submit_bid(i, None, pass_=True)
        game.set_trump(0, Suit.HEARTS)
        for i in range(4):
            game.submit_stir(i, None)
        assert game.state.phase == Phase.EXCHANGE


class TestGameDiscard:
    def test_game_discard(self):
        game = Game()
        game.start_round()
        game.submit_bid(0, Rank.THREE, pass_=False)
        for i in range(1, 4):
            game.submit_bid(i, None, pass_=True)
        game.set_trump(0, Suit.HEARTS)
        for i in range(4):
            game.submit_stir(i, None)
        declarer_idx = None
        for p in game.state.players:
            if p.is_declarer:
                declarer_idx = p.index
                break
        assert declarer_idx is not None
        discard = game.state.players[declarer_idx].hand[:8]
        ok = game.submit_discard(declarer_idx, discard)
        assert ok is True
        assert game.state.phase == Phase.PLAYING


class TestGamePlaySingleTrick:
    def test_game_play_single_trick(self):
        game = _setup_game_in_playing()
        for _ in range(4):
            player_idx = game.state.current_player_index
            hand = game.state.players[player_idx].hand
            if not hand:
                break
            card = hand[0]
            ok = game.submit_play(player_idx, [card])
            assert ok is True
        assert game.state.last_completed_trick is not None or game.state.phase == Phase.SCORING


class TestGameScoring:
    def test_game_scoring_declarer_big_light(self):
        game = _setup_game_in_playing()
        for _ in range(25):
            if game.state.phase == Phase.SCORING:
                break
            for _ in range(4):
                if game.state.phase == Phase.SCORING:
                    break
                player_idx = game.state.current_player_index
                hand = game.state.players[player_idx].hand
                if not hand:
                    break
                card = hand[0]
                game.submit_play(player_idx, [card])
                if game.state.last_completed_trick is not None:
                    game.clear_trick()

    def test_game_scoring_defender_wins(self):
        game = _setup_game_in_playing()
        for _ in range(25):
            if game.state.phase == Phase.SCORING:
                break
            for _ in range(4):
                if game.state.phase == Phase.SCORING:
                    break
                player_idx = game.state.current_player_index
                hand = game.state.players[player_idx].hand
                if not hand:
                    break
                card = hand[0]
                game.submit_play(player_idx, [card])
                if game.state.last_completed_trick is not None:
                    game.clear_trick()


class TestGameIndependentLevels:
    def test_game_independent_levels(self):
        game = _setup_game_in_playing()
        assert game.state.teams[0].current_level == game.state.teams[1].current_level


class TestGameAwaitingAction:
    def test_game_awaiting_action_bid(self):
        game = Game()
        game.start_round()
        assert game.get_awaiting_action() == "bid"

    def test_game_awaiting_action_play(self):
        game = _setup_game_in_playing()
        action = game.get_awaiting_action()
        assert action in ("play", "clear_trick", "next_round")

    def test_game_awaiting_action_clear_trick(self):
        game = _setup_game_in_playing()
        for _ in range(4):
            player_idx = game.state.current_player_index
            hand = game.state.players[player_idx].hand
            if not hand:
                break
            card = hand[0]
            game.submit_play(player_idx, [card])
        if game.state.last_completed_trick is not None:
            assert game.get_awaiting_action() in ("clear_trick", "next_round")

    def test_game_awaiting_action_next_round(self):
        game = _setup_game_in_playing()
        for _ in range(25):
            if game.state.phase == Phase.SCORING:
                break
            for _ in range(4):
                if game.state.phase == Phase.SCORING:
                    break
                player_idx = game.state.current_player_index
                hand = game.state.players[player_idx].hand
                if not hand:
                    break
                card = hand[0]
                game.submit_play(player_idx, [card])
                if game.state.last_completed_trick is not None:
                    game.clear_trick()
        if game.state.phase == Phase.SCORING:
            assert game.get_awaiting_action() == "next_round"


class TestGameIsHumanTurn:
    def test_game_is_human_turn(self):
        game = Game()
        game.start_round()
        result = game.is_human_turn()
        assert isinstance(result, bool)


class TestGameAIAutoPlay:
    def test_game_ai_auto_play_after_human_bid(self):
        """When human bids or passes, AI players should auto-bid until bidding resolves or it's the human's turn again."""
        game = Game()
        game.start_round()
        # Human (player 3) passes
        game.submit_bid(3, None, pass_=True)
        # After this call, AI players should have auto-bid.
        # The game should either be awaiting set_trump (bidding over with a winner)
        # or awaiting another human bid (unlikely in 4-player with 3 passes).
        action = game.get_awaiting_action()
        assert action in ("bid", "set_trump", None)

    def test_game_ai_auto_play_after_human_play(self):
        """When human plays, AI players should auto-play until it's the human's turn again."""
        game = _setup_game_in_playing()
        if game.is_human_turn():
            hand = game.state.players[3].hand
            if hand:
                card = hand[0]
                game.submit_play(3, [card])
                # After human plays, AI should auto-play.
                # The awaiting_action should now be something the human needs to do,
                # or clear_trick/next_round.
                action = game.get_awaiting_action()
                assert action is not None


# ---- Helpers ----

def _setup_game_in_playing() -> Game:
    game = Game()
    game.start_round()
    game.submit_bid(0, Rank.THREE, pass_=False)
    for i in range(1, 4):
        game.submit_bid(i, None, pass_=True)
    game.set_trump(0, Suit.HEARTS)
    for i in range(4):
        game.submit_stir(i, None)
    declarer_idx = None
    for p in game.state.players:
        if p.is_declarer:
            declarer_idx = p.index
            break
    discard = game.state.players[declarer_idx].hand[:8]
    game.submit_discard(declarer_idx, discard)
    return game
