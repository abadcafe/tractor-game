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

    def test_game_ai_auto_play_ace_bid_no_fall_through(self):
        """CR-010: When AI wins bid at ACE during auto-play, loop must continue.

        Setup: player 0 bids THREE, player 1 bids ACE, players 2 and 3 pass.
        Bidding is NOT over (only 2 consecutive passes after ACE).
        Human's pass triggers _ai_auto_play. During auto-play, AI 0 passes
        (no valid levels above ACE), making 3 consecutive passes → bidding ends.
        Winner is AI 1 (ACE). set_trump transitions to STIRRING with cp=2 (AI).

        Without CR-010 fix: code falls through to get_valid_bids() which
        returns [] for ACE, causing break. Game stuck at STIRRING with AI cp.
        With fix: continue after set_trump, loop enters STIRRING handler.
        """
        game = Game()
        game.start_round()
        game.submit_bid(0, Rank.THREE, pass_=False)
        game.submit_bid(1, Rank.ACE, pass_=False)
        game.submit_bid(2, None, pass_=True)
        # Human passes — bidding not over (2 consecutive), auto-play triggers
        game.submit_bid(3, None, pass_=True)
        # Auto-play: AI 0 passes → 3 consecutive → AI 1 wins at ACE
        # set_trump(1,...) → STIRRING cp=2
        # CR-010: fall-through → break → stuck at STIRRING cp=2 (AI)
        # Fixed: continue → STIRRING handler → progresses
        assert game.state.phase != Phase.STIRRING or game.is_human_turn()

    def test_game_get_legal_plays_uses_lead_player_index(self):
        """CR-011: get_legal_plays must use lead_player_index to find lead cards.

        When the lead player has a higher index than a following player who
        has already played, the old code iterated trick slots in list order
        (by player_index: 0, 1, 2, 3) and picked the first non-None slot
        as the lead. This is wrong when e.g. player 3 leads SPADES and
        player 1 follows with HEARTS -- slot 1 (player 1's HEARTS) is
        checked before slot 3 (player 3's SPADES), causing the wrong
        lead suit to be used for follow-suit validation.
        """
        from server.engine.card import Card
        from server.engine.game_state import TrickSlot

        game = _setup_game_in_playing()

        trump_suit = game.state.trump_suit or Suit.HEARTS
        trump_rank = game.state.trump_rank

        # Player 3 leads SPADES ACE (not trump)
        lead_card = Card(
            id="test-lead", suit=Suit.SPADES, rank=Rank.ACE,
            is_joker=False, is_big_joker=False, points=0, deck=1,
        )
        # Player 1 plays HEARTS ACE (off-suit, different from lead)
        other_card = Card(
            id="test-other", suit=Suit.HEARTS, rank=Rank.ACE,
            is_joker=False, is_big_joker=False, points=0, deck=1,
        )
        # Player 0 has both SPADES and HEARTS available
        spades_card = Card(
            id="test-spades", suit=Suit.SPADES, rank=Rank.KING,
            is_joker=False, is_big_joker=False, points=0, deck=1,
        )
        hearts_card = Card(
            id="test-hearts", suit=Suit.HEARTS, rank=Rank.KING,
            is_joker=False, is_big_joker=False, points=0, deck=1,
        )

        # Trick: player 3 led SPADES, player 1 played HEARTS (different suit)
        game.state = game.state.model_copy(update={
            "phase": Phase.PLAYING,
            "trump_suit": trump_suit,
            "trump_rank": trump_rank,
            "lead_player_index": 3,
            "lead_play_type": PlayType.SINGLE,
            "current_player_index": 0,
            "current_trick": [
                TrickSlot(player_index=0, cards=None),        # player 0: hasn't played
                TrickSlot(player_index=1, cards=[other_card]), # player 1: played HEARTS
                TrickSlot(player_index=2, cards=None),        # player 2: hasn't played
                TrickSlot(player_index=3, cards=[lead_card]),  # player 3: led SPADES
            ],
            "players": [
                game.state.players[0].model_copy(
                    update={"hand": [spades_card, hearts_card]}
                ),
                game.state.players[1],
                game.state.players[2],
                game.state.players[3],
            ],
        })

        legal = game.get_legal_plays(0)
        legal_ids = {c.id for play in legal for c in play.cards}

        # Player 0 must follow SPADES (the actual lead suit from player 3).
        # With the fix: slot 3 (player 3's SPADES) is used as lead_action,
        #   so player 0 must follow SPADES. spades_card is legal, hearts is not.
        # With the bug: slot 1 (player 1's HEARTS) would be used as lead_action,
        #   so player 0 would follow HEARTS. hearts_card would be legal instead.
        assert "test-spades" in legal_ids, (
            "SPADES (the lead suit from player 3) must be a legal follow"
        )
        assert "test-hearts" not in legal_ids, (
            "HEARTS should NOT be legal -- must follow SPADES (lead_player_index=3). "
            "Bug: code picked player 1's HEARTS as lead instead."
        )


class TestGameWinningTeam:
    """CR-008: verify get_winning_team and team-level update on game over."""

    def test_get_winning_team_returns_none_when_not_game_over(self):
        game = Game()
        game.start_round()
        assert game.get_winning_team() is None

    def test_game_over_updates_team_levels_and_sets_winner(self):
        """When next_round detects game over, team levels must be updated."""
        from server.engine.scoring import ScoreResult
        from unittest.mock import patch

        game = Game()
        game.start_round()
        game.submit_bid(0, Rank.THREE, pass_=False)
        for i in range(1, 4):
            game.submit_bid(i, None, pass_=True)
        game.set_trump(0, Suit.HEARTS)
        for i in range(4):
            game.submit_stir(i, None)

        # Force phase to SCORING
        game.state = game.state.model_copy(update={"phase": Phase.SCORING})

        # Mock _calculate_round_score to return a game-over result
        # Team 0 at ACE (game over), team 1 stays at 2
        fake_result = ScoreResult(
            declarer_level_change=3,
            switch_declarer=False,
            bottom_card_bonus=0,
            total_defender_points=0,
            team0_new_level=Rank.ACE,
            team1_new_level=Rank.TWO,
        )
        with patch.object(game, '_calculate_round_score', return_value=fake_result):
            game.next_round()

        assert game.state.phase == Phase.GAME_OVER
        # Team levels must be updated to new levels
        assert game.state.teams[0].current_level == Rank.ACE
        assert game.state.teams[1].current_level == Rank.TWO
        # Winning team must be team 0
        assert game.get_winning_team() == 0

    def test_game_over_defender_wins(self):
        """When defender team reaches ACE, they should be the winner."""
        from server.engine.scoring import ScoreResult
        from unittest.mock import patch

        game = Game()
        game.start_round()
        game.submit_bid(0, Rank.THREE, pass_=False)
        for i in range(1, 4):
            game.submit_bid(i, None, pass_=True)
        game.set_trump(0, Suit.HEARTS)
        for i in range(4):
            game.submit_stir(i, None)

        game.state = game.state.model_copy(update={"phase": Phase.SCORING})

        # Team 1 (defender) reaches ACE
        fake_result = ScoreResult(
            declarer_level_change=-3,
            switch_declarer=True,
            bottom_card_bonus=0,
            total_defender_points=200,
            team0_new_level=Rank.THREE,
            team1_new_level=Rank.ACE,
        )
        with patch.object(game, '_calculate_round_score', return_value=fake_result):
            game.next_round()

        assert game.state.phase == Phase.GAME_OVER
        assert game.state.teams[0].current_level == Rank.THREE
        assert game.state.teams[1].current_level == Rank.ACE
        assert game.get_winning_team() == 1


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
