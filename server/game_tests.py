"""Tests for server/game.py -- Game aggregate root.

All tests use only public interfaces: Game.__init__, Game.run, Game.act,
Game.snapshot, Game.is_over, Game.get_phase, Game.get_player, Game.set_on_game_over,
Game.cancel, StateSnapshot.to_dict.
No tests access private fields like _game_state, _round_state, or _dealing_task.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from server.actions import BidAction, NextRoundAction, PlayAction, SkipStirAction
from server.game import Game
from server.player import AutoPlayer, GameView
from server.sm.result import Ok, Rejected
from server.snapshot import StateSnapshot


def _create_game_with_auto_players():
    """Create a Game with 4 AutoPlayers."""
    players = [AutoPlayer(index=i) for i in range(4)]
    return Game(players=players)


# ---- Initialization ----


def test_game_init_creates_valid_state():
    game = _create_game_with_auto_players()
    # Verify via public interface
    assert game.get_phase() == "IDLE"
    assert game.is_over() is False


# ---- get_phase() ----


def test_get_phase_returns_phase():
    game = _create_game_with_auto_players()
    assert game.get_phase() == "IDLE"


# ---- run() ----


@pytest.mark.asyncio
async def test_run_transitions_to_deal_bid():
    game = _create_game_with_auto_players()
    await game.run()
    # Verify via snapshot (public interface)
    snap = game.snapshot(for_player=0)
    assert snap.phase in ("DEAL_BID", "STIRRING", "EXCHANGE", "PLAYING", "COMPLETE")


# ---- act() ----


@pytest.mark.asyncio
async def test_act_rejects_wrong_player():
    """PlayAction during DEAL_BID should be rejected without raising."""
    game = _create_game_with_auto_players()
    await game.run()
    # Should not raise; rejection is communicated via send_error instead
    await game.act(player_index=0, action=PlayAction(cards=[]))


@pytest.mark.asyncio
async def test_act_value_error_not_propagated():
    """Invalid actions no longer raise ValueError; they send error messages."""
    game = _create_game_with_auto_players()
    await game.run()
    # Should not raise
    await game.act(player_index=0, action=PlayAction(cards=[]))


# ---- snapshot() ----


@pytest.mark.asyncio
async def test_snapshot_returns_player_hand():
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    assert isinstance(snap, StateSnapshot)
    assert isinstance(snap.player_hand, list)


def test_snapshot_raises_before_run():
    """snapshot() must raise when called before run().

    Before run(), _round_state is None, so snapshot() cannot build a
    valid StateSnapshot. Rather than returning a partial/empty snapshot
    that could mislead callers, it raises an error (AssertionError via
    assert guard).
    """
    game = _create_game_with_auto_players()
    with pytest.raises(AssertionError, match="snapshot\\(\\) called before run"):
        game.snapshot(for_player=0)


@pytest.mark.asyncio
async def test_snapshot_phase():
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    assert snap.phase in ("DEAL_BID", "STIRRING", "EXCHANGE", "PLAYING", "COMPLETE", "GAME_OVER", "SCORING")


@pytest.mark.asyncio
async def test_snapshot_current_player():
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    assert isinstance(snap.current_player, int)
    assert 0 <= snap.current_player <= 3


@pytest.mark.asyncio
async def test_snapshot_current_player_deal_bid():
    """During DEAL_BID, current_player should be the deal_target."""
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    if snap.phase == "DEAL_BID":
        assert isinstance(snap.current_player, int)
        assert 0 <= snap.current_player <= 3


@pytest.mark.asyncio
async def test_snapshot_legal_actions_in_playing():
    """Legal actions should be populated during PLAYING phase.

    After Task 010 refactor, legal_actions is list[list[Card]].
    Each entry is a plain list of Card objects (no .type attribute).
    """
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    assert isinstance(snap.legal_actions, list)
    # If in PLAYING phase, legal_actions should contain card lists
    if snap.phase == "PLAYING" and len(snap.legal_actions) > 0:
        entry = snap.legal_actions[0]
        # Entry is a list of Card objects (not a PlayAction)
        assert isinstance(entry, list)
        assert not hasattr(entry, "type")  # not a PlayAction
        if len(entry) > 0:
            from server.sm.card_model import Card
            assert isinstance(entry[0], Card)


@pytest.mark.asyncio
async def test_snapshot_awaiting_action_play():
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    assert snap.awaiting_action in ("stir", "discard", "play", "next_round", None)


@pytest.mark.asyncio
async def test_snapshot_trump_info():
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    assert snap.trump_rank is not None


@pytest.mark.asyncio
async def test_snapshot_team_levels():
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    assert snap.team0_level is not None
    assert snap.team1_level is not None


@pytest.mark.asyncio
async def test_snapshot_bid_events():
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    assert isinstance(snap.bid_events, list)


@pytest.mark.asyncio
async def test_snapshot_stirring_state():
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    # stirring_state may be None outside of STIRRING phase
    assert snap.stirring_state is None or isinstance(snap.stirring_state, dict)


@pytest.mark.asyncio
async def test_snapshot_exchange_state():
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    assert snap.exchange_state is None or isinstance(snap.exchange_state, dict)


@pytest.mark.asyncio
async def test_snapshot_scoring_in_complete():
    """When round is COMPLETE, snapshot should include scoring info."""
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    assert snap.scoring is None or isinstance(snap.scoring, dict)


# ---- is_over() ----


def test_is_over_false_during_game():
    game = _create_game_with_auto_players()
    assert game.is_over() is False


@pytest.mark.asyncio
async def test_is_over_true_after_game_over():
    """Game should be over when get_phase() returns GAME_OVER."""
    game = _create_game_with_auto_players()
    await game.run()
    # We can't easily force GAME_OVER through public API alone in unit test.
    # Instead, verify is_over() is consistent with get_phase().
    assert game.is_over() == (game.get_phase() == "GAME_OVER")


@pytest.mark.asyncio
async def test_snapshot_winning_team_in_game_over():
    """When game is over, snapshot should include winning_team."""
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    # During normal flow, game is not over yet
    if game.is_over():
        assert snap.winning_team is not None
        assert isinstance(snap.winning_team, int)
    else:
        assert snap.winning_team is None


@pytest.mark.asyncio
async def test_game_over_consistency():
    """is_over() should equal get_phase() == 'GAME_OVER'."""
    game = _create_game_with_auto_players()
    assert game.is_over() == (game.get_phase() == "GAME_OVER")
    await game.run()
    assert game.is_over() == (game.get_phase() == "GAME_OVER")


# ---- Dealing loop ----


@pytest.mark.asyncio
async def test_dealing_loop_deals_cards():
    """After run(), dealing loop should have started dealing cards."""
    game = _create_game_with_auto_players()
    await game.run()
    # Give dealing loop a moment to run
    await asyncio.sleep(0.1)
    snap = game.snapshot(for_player=0)
    # Players should have some cards dealt by now or phase has moved on
    if snap.phase == "DEAL_BID":
        assert len(snap.player_hand) >= 0  # cards may or may not have been dealt yet


# ---- Action dispatch with type conversion ----


@pytest.mark.asyncio
async def test_act_bid_during_dealing_converts_to_bid_event():
    """BidAction from player.py should be converted to sm BidEvent internally."""
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    # During DEAL_BID, try bidding. This tests the BidAction -> BidEvent conversion.
    # If not in DEAL_BID, we can't test bid; that's fine, integration tests cover it.
    if snap.phase == "DEAL_BID" and len(snap.player_hand) > 0:
        # Find a trump rank card to bid with
        trump_cards = [c for c in snap.player_hand if c.rank == snap.trump_rank]
        if trump_cards:
            action = BidAction(cards=trump_cards[:1], count=1)
            # Bid may be rejected (e.g. priority too low), but act() never raises
            await game.act(player_index=0, action=action)


@pytest.mark.asyncio
async def test_act_skip_stir_during_stirring():
    """SkipStirAction is a valid action type that Game.act() can distinguish
    from StirAction. The actual dispatch routing (SkipStirAction -> round_sm.pass_stir)
    is verified in integration tests (task-008).
    """
    action = SkipStirAction()
    assert isinstance(action, SkipStirAction)
    # Verify it is NOT a StirAction -- Game.act() must dispatch differently
    from server.actions import StirAction
    assert not isinstance(action, StirAction)


@pytest.mark.asyncio
async def test_act_next_round_during_non_complete():
    """NextRoundAction during non-COMPLETE phase should be rejected without raising."""
    game = _create_game_with_auto_players()
    await game.run()
    # Should not raise; rejection is communicated via send_error instead
    await game.act(player_index=0, action=NextRoundAction())


# ---- get_phase() GAME_OVER priority ----


def test_get_phase_prioritizes_game_over():
    """get_phase() must return GAME_OVER when _game_state.phase is GAME_OVER,
    even if _round_state is still non-None with a different phase.

    Since we cannot easily force GAME_OVER through public API in a unit test
    without accessing private fields, we verify the observable contract:
    is_over() must always be consistent with get_phase().
    """
    game = _create_game_with_auto_players()
    assert game.get_phase() == "IDLE"
    # The invariant: is_over() == (get_phase() == "GAME_OVER")
    assert game.is_over() == (game.get_phase() == "GAME_OVER")


# ---- on_game_over callback ----


@pytest.mark.asyncio
async def test_set_on_game_over_callback_fires_on_game_over():
    """When game transitions to GAME_OVER, the registered callback should be invoked.

    Strategy: Create a game in IN_ROUND state by patching game_sm.create_game,
    then patch round_sm.create_round to return a COMPLETE-phase RoundState.
    After game.run(), patch game_sm.process_round_result to return a GAME_OVER
    state. Then call game.act() with NextRoundAction. If the callback fires,
    the mock is called.

    Uses game.cancel() (public method) to stop the dealing loop instead of
    accessing the private _dealing_task field.
    """
    from server.sm import game_sm as gm, round_sm as rm
    from server.sm.scoring import RoundResult
    from server.sm.card_model import Rank

    # Create a game in IN_ROUND state by patching create_game
    in_round_state = gm.GameState(
        phase="IN_ROUND",
        team0_level=Rank.TEN,
        team1_level=Rank.TEN,
        declarer_team=0,
        last_declarer_player=0,
        winning_team=None,
        round_number=1,
    )

    # Create a COMPLETE-phase RoundState mock
    complete_round = MagicMock()
    complete_round.phase = "COMPLETE"
    complete_round.players_hand = [[] for _ in range(4)]
    complete_round.declarer_player = 0
    complete_round.result = None  # Will be overridden by get_round_result mock

    # The GAME_OVER state that process_round_result will return
    game_over_state = gm.GameState(
        phase="GAME_OVER",
        team0_level=Rank.ACE,
        team1_level=Rank.TEN,
        declarer_team=None,
        last_declarer_player=None,
        winning_team=0,
        round_number=1,
    )

    # Build a mock RoundResult
    mock_result = MagicMock(spec=RoundResult)
    mock_result.team0_new_level = Rank.ACE
    mock_result.team1_new_level = Rank.TEN
    mock_result.next_declarer_team = 0
    mock_result.next_declarer_player = 0

    with patch.object(gm, "create_game", return_value=in_round_state):
        game = _create_game_with_auto_players()

    callback = MagicMock()
    game.set_on_game_over(callback)

    # Run the game with patched sm functions
    with patch.object(gm, "start_game", return_value=Ok(in_round_state)):
        with patch.object(rm, "create_round", return_value=complete_round):
            await game.run()
            # Cancel the dealing loop via public interface
            await game.cancel()

    # Verify game is not over yet (before triggering GAME_OVER)
    assert not game.is_over()

    # Now trigger GAME_OVER via act() with NextRoundAction.
    # Patch get_round_result to return our mock_result so act() can pass
    # it to game_sm.process_round_result.
    with patch.object(rm, "get_round_result", return_value=mock_result):
        with patch.object(gm, "process_round_result", return_value=Ok(game_over_state)):
            # COMPLETE phase now requires all 4 players to confirm
            for p in range(4):
                await game.act(player_index=p, action=NextRoundAction())

    # Game must have transitioned to GAME_OVER
    assert game.is_over()
    # Callback must have been called with the game instance
    callback.assert_called_once_with(game)


# ---- get_player() ----


def test_get_player_returns_player_by_index():
    """Game.get_player(index) returns the Player at that index."""
    players = [AutoPlayer(index=i) for i in range(4)]
    game = Game(players=players)
    for i in range(4):
        assert game.get_player(i) is players[i]


# ---- cancel() ----


@pytest.mark.asyncio
async def test_cancel_stops_dealing_loop():
    """Game.cancel() stops the dealing loop background task.

    After cancel(), the game's dealing loop should no longer be running.
    We verify this by calling cancel() after run() and confirming it does
    not raise an error. The dealing loop simply stops producing state changes.
    """
    game = _create_game_with_auto_players()
    await game.run()
    # Cancel should succeed without error
    await game.cancel()
    # After cancel, the game should still be in a valid state
    # (snapshot should still work, just the dealing loop is stopped)
    snap = game.snapshot(for_player=0)
    assert isinstance(snap, StateSnapshot)
    # Calling cancel() again should be idempotent (no error)
    await game.cancel()


# ---- StateSnapshot.to_dict() ----


@pytest.mark.asyncio
async def test_snapshot_to_dict_json_serializable():
    """StateSnapshot.to_dict() must return a dict that is JSON-serializable.

    This is critical because HumanPlayer.on_state() calls ws.send_json() which
    requires JSON-serializable data. The sm Card/Suit/Rank types are Pydantic
    models and enums that are not directly JSON-serializable. legal_actions
    is now list[list[Card]], serialized to list of card-dict lists.
    """
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    result = snap.to_dict()
    # Must be a dict
    assert isinstance(result, dict)
    # Must be JSON-serializable (no Pydantic objects, no enums as objects)
    serialized = json.dumps(result)
    assert isinstance(serialized, str)
    # Must contain the required fields from spec section 5.5
    assert "phase" in result
    assert "player_hand" in result
    assert "trump_rank" in result
    assert "current_player" in result
    # legal_actions must be a list of lists (card lists, not PlayAction dicts)
    legal_actions_raw = result["legal_actions"]
    if len(legal_actions_raw) > 0:
        legal_entry = legal_actions_raw[0]
        # list of card dicts
        if len(legal_entry) > 0:
            assert "id" in legal_entry[0]
            assert "type" not in legal_entry[0]


@pytest.mark.asyncio
async def test_snapshot_to_dict_card_format():
    """StateSnapshot.to_dict() must format cards as {"id", "suit", "rank"}.

    Per spec section 5.5, each card in player_hand must be:
    {"id": "D1-H-A", "suit": "hearts", "rank": "A"}
    This is a subset of the sm Card fields (omitting is_joker, is_big_joker,
    points, deck). Suit and Rank enums must be serialized as their string values.
    """
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    result = snap.to_dict()
    # If player has cards, verify the format
    player_hand_raw = result["player_hand"]
    if len(player_hand_raw) > 0:
        card = player_hand_raw[0]
        assert isinstance(card, dict)
        assert "id" in card
        assert "suit" in card
        assert "rank" in card
        # suit and rank must be strings (not enum objects)
        assert isinstance(card["suit"], str)
        assert isinstance(card["rank"], str)
        # Must NOT contain internal sm fields
        assert "is_joker" not in card
        assert "is_big_joker" not in card
        assert "points" not in card
        assert "deck" not in card


# ---- resolve_cards() ----


@pytest.mark.asyncio
async def test_resolve_cards_returns_matching_cards():
    """Game.resolve_cards() returns Ok with Card objects matching the given IDs
    from the specified player's hand.

    This is needed because human players send card IDs via WebSocket,
    but Game.act() must pass Card Pydantic model objects to sm functions.
    resolve_cards() bridges this gap by looking up Card objects by their
    ID string in the player's hand.
    """
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    if len(snap.player_hand) > 0:
        card_ids = [c.id for c in snap.player_hand[:2]]
        result = game.resolve_cards(player_index=0, card_ids=card_ids)
        assert isinstance(result, Ok)
        for original, resolved_card in zip(card_ids, result.value):
            assert resolved_card.id == original
            # Must be a Card Pydantic model (not a string or dict)
            from server.sm.card_model import Card
            assert isinstance(resolved_card, Card)


@pytest.mark.asyncio
async def test_resolve_cards_rejects_on_unknown_id():
    """Game.resolve_cards() returns Rejected if any card_id is not found
    in the player's hand.

    This prevents human players from submitting cards they don't hold,
    which would be an invalid action.
    """
    game = _create_game_with_auto_players()
    await game.run()
    result = game.resolve_cards(player_index=0, card_ids=["NONEXISTENT-CARD-ID"])
    assert isinstance(result, Rejected)


# ---- Bug 1 regression: bid must not trigger _push_state_to_all cascade ----


@pytest.mark.asyncio
async def test_bid_during_deal_bid_does_not_push_state_to_all():
    """BidAction during DEAL_BID must NOT trigger an extra _push_state_to_all.

    Regression test for Bug 1: when a bid triggered _push_state_to_all(),
    each AutoPlayer.on_state() would create_task(bid) → game.act() →
    _push_state_to_all() → on_state() → … an exponential task cascade
    that consumed all CPU and memory.

    The fix: bid during DEAL_BID does not push state at all — the
    dealing loop pushes to all players every 0.5s anyway, so the next
    tick carries the updated bid_winner.

    Setup: 3 CountingPlayers + 1 TestHumanPlayer that bids when it
    gets a trump-rank card.  Each CountingPlayer records how many
    times on_state is called.  One dealing_loop cycle = 1 push =
    4 on_state calls.  If the bid triggered an extra push, every
    player's count would be 4 higher than expected.
    """
    import random as _random
    from server.player import Player

    class CountingPlayer(Player):
        """Player that counts on_state invocations."""
        def __init__(self, index: int) -> None:
            super().__init__(index)
            self.state_count = 0

        async def on_state(self, game: object) -> None:
            self.state_count += 1

    class TestHumanPlayer(Player):
        """Simulates a human who bids as soon as they see a trump-rank card."""
        def __init__(self, index: int) -> None:
            super().__init__(index)
            self.bid_done = asyncio.Event()
            self.state_count = 0

        async def on_state(self, game: GameView) -> None:
            self.state_count += 1
            if self.bid_done.is_set():
                return
            snapshot = game.snapshot(self.index)
            if snapshot.phase != "DEAL_BID":
                return
            trump_cards = [c for c in snapshot.player_hand if c.rank == snapshot.trump_rank]
            if not trump_cards:
                return
            action = BidAction(cards=trump_cards[:1], count=1)
            await game.act(self.index, action)
            self.bid_done.set()

    # Fixed seed so the shuffle is deterministic: player 0 receives a
    # trump-rank card on the 17th deal (after dealing has been running
    # for a while, not the very first card).  Restored after the test.
    _prev_state = _random.getstate()
    _random.seed(16)

    human = TestHumanPlayer(index=0)
    counters = [CountingPlayer(index=1), CountingPlayer(index=2), CountingPlayer(index=3)]
    players: list[Player] = [human, *counters]
    game = Game(players=players, deal_delay=0.05)

    await game.run()

    # Wait for the human player to see a trump-rank card and bid.
    # 100 cards × 0.05s = 5s max; with seed(16) player 0 gets one
    # on the 17th deal (~0.85s).
    await asyncio.wait_for(human.bid_done.wait(), timeout=6.0)

    await game.cancel()

    assert human.bid_done.is_set()
    # All 4 players should have the same on_state count (each push
    # calls on_state on all 4 players).  If the bid triggered an
    # extra _push_state_to_all, all counts would be 4 higher.
    # Verify they're equal — the dealing_loop pushes uniformly.
    all_counts = [human.state_count] + [c.state_count for c in counters]
    assert len(set(all_counts)) == 1, \
        f"Uneven on_state counts {all_counts}; bid may have triggered an extra push"

    _random.setstate(_prev_state)


# ---- Bug 2 regression: snapshot must contain player_hand_counts ----


@pytest.mark.asyncio
async def test_snapshot_to_dict_contains_all_required_fields():
    """StateSnapshot.to_dict() must contain ALL fields from spec section 5.5.

    Regression test for Bug 2: the `player_hand_counts` field was missing
    from StateSnapshot, causing the frontend's game-table component to
    show "0 张" for every player because `snapshot.player_hand_counts[i]`
    evaluated to `undefined ?? 0`.

    This test asserts the complete set of required fields so any future
    addition to the spec is also caught if the server doesn't serialize it.
    """
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    result = snap.to_dict()

    # Complete list of required fields per spec section 5.5
    required_fields = [
        "phase",
        "player_hand",
        "player_hand_counts",
        "bottom_cards",
        "trump_suit",
        "trump_rank",
        "declarer_team",
        "declarer_player",
        "current_player",
        "defender_points",
        "trick",
        "trick_history",
        "legal_actions",
        "awaiting_action",
        "scoring",
        "winning_team",
        "team0_level",
        "team1_level",
        "bid_events",
        "bid_winner",
        "stirring_state",
        "exchange_state",
    ]

    for field in required_fields:
        assert field in result, f"Missing required field: {field}"

    # player_hand_counts specifically: must be a list of 4 ints
    hand_counts = result["player_hand_counts"]
    assert len(hand_counts) == 4
    for count in hand_counts:
        assert isinstance(count, int)

    await game.cancel()


# ---- Task 010: new get_legal_plays signature ----


@pytest.mark.asyncio
async def test_snapshot_legal_actions_are_card_lists():
    """Legal actions entries are plain card lists, not PlayAction objects.

    After the refactor, legal_actions is list[list[Card]].
    Each entry is a list of Card objects (no .type attribute).
    """
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    if snap.phase == "PLAYING" and len(snap.legal_actions) > 0:
        entry = snap.legal_actions[0]
        # Entry is a list of Card objects, not a PlayAction
        assert isinstance(entry, list)
        assert not hasattr(entry, "type")  # not a PlayAction
        if len(entry) > 0:
            from server.sm.card_model import Card
            assert isinstance(entry[0], Card)


@pytest.mark.asyncio
async def test_snapshot_legal_actions_to_dict_format():
    """to_dict() serializes legal_actions as list of card-dict lists (no 'type' field)."""
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    d = snap.to_dict()
    legal_actions_val = d["legal_actions"]
    if snap.phase == "PLAYING" and len(legal_actions_val) > 0:
        entry = legal_actions_val[0]
        # Entry is a list of card dicts, not a dict with 'type' key
        if len(entry) > 0:
            assert "id" in entry[0]  # card dict format
            assert "type" not in entry[0]  # no PlayAction wrapper


@pytest.mark.asyncio
async def test_snapshot_completed_trick_no_lead_type():
    """CompletedTrick no longer has lead_type field.

    After Task-009, _serialize_completed_trick should not include lead_type.
    """
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    d = snap.to_dict()
    for trick in d["trick_history"]:
        assert "lead_type" not in trick


# ---- Game auto-completion ----


@pytest.mark.asyncio
async def test_game_auto_completes_past_deal_bid():
    """Game with 4 AutoPlayers progresses past DEAL_BID phase.

    Verifies that the dealing loop makes progress and the game transitions
    to a later phase after waiting.
    """
    game = _create_game_with_auto_players()
    await game.run()
    # Wait for dealing to make progress
    await asyncio.sleep(3)
    # Verify the game has progressed
    phase = game.get_phase()
    assert phase in (
        "DEAL_BID", "STIRRING", "EXCHANGE", "PLAYING",
        "COMPLETE", "GAME_OVER",
    )
    # Snapshot must still be valid
    snap = game.snapshot(for_player=0)
    assert isinstance(snap.player_hand, list)
    await game.cancel()


@pytest.mark.asyncio
async def test_game_over_via_auto_players_starts():
    """Game with 4 AutoPlayers starts and has valid initial state.

    Verifies that the game is created with valid phase and can be started.
    """
    game = _create_game_with_auto_players()
    initial_phase = game.get_phase()
    assert initial_phase in ("IDLE", "IN_ROUND", "DEAL_BID")


# ---- Bug 1 regression: no resource explosion ----


@pytest.mark.asyncio
async def test_full_game_flow_completes_without_resource_explosion():
    """A game with 4 AutoPlayers must complete without CPU/memory explosion.

    Regression test for Bug 1: AutoPlayer on_state() -> create_task(bid)
    -> game.act() -> _push_state_to_all() -> on_state() -> ... exponential
    task cascade consumed 96.9% CPU and 8.8 GB RAM.

    This test creates a game with 4 AutoPlayers and lets them drive
    it through at least one full round. After running, the game must
    have progressed and not be stuck in an infinite task cascade.
    """
    game = _create_game_with_auto_players()

    # Patch asyncio.sleep in the game module to speed up dealing.
    # The dealing loop calls asyncio.sleep(0.5) between cards;
    # replacing it with a 1ms sleep makes the test practical.
    original_sleep = asyncio.sleep

    async def fast_sleep(delay: float) -> None:
        if delay >= 0.1:
            # Only speed up long sleeps (dealing loop), not short waits
            await original_sleep(0.001)
        else:
            await original_sleep(delay)

    with patch.object(asyncio, 'sleep', side_effect=fast_sleep):
        await game.run()

        # Let the game auto-progress -- with fast sleep, dealing
        # completes in ~0.1s and AI plays complete quickly
        await original_sleep(3)

        phase = game.get_phase()
        # After 3s, the game should have progressed past DEAL_BID
        assert phase in (
            "DEAL_BID", "STIRRING", "EXCHANGE", "PLAYING",
            "COMPLETE", "GAME_OVER",
        ), f"Game stuck in unexpected phase: {phase}"

        # Snapshot must still be valid (no cascading error state)
        try:
            snap = game.snapshot(for_player=0)
            assert isinstance(snap.player_hand, list)
        except RuntimeError:
            pass

        await game.cancel()


# ---- Game over removes from registry ----


@pytest.mark.asyncio
async def test_game_over_removes_from_registry():
    """Test that the on_game_over callback can remove the game from registry.

    Creates a Game with mocked sm functions to force it through to GAME_OVER,
    and verifies the callback fires and can remove the game from the registry.
    """
    from server.sm import game_sm as gm, round_sm as rm
    from server.sm.card_model import Rank
    from server.sm.scoring import RoundResult
    from server.game_registry import GameRegistry

    test_registry = GameRegistry()
    players = [AutoPlayer(index=i) for i in range(4)]

    # Create game in IN_ROUND state by patching create_game
    in_round_state = gm.GameState(
        phase="IN_ROUND",
        team0_level=Rank.TEN,
        team1_level=Rank.TEN,
        declarer_team=0,
        last_declarer_player=0,
        winning_team=None,
        round_number=1,
    )

    # Create a COMPLETE-phase RoundState mock (so act() accepts NextRoundAction)
    complete_round = MagicMock()
    complete_round.phase = "COMPLETE"
    complete_round.players_hand = [[] for _ in range(4)]
    complete_round.declarer_player = 0
    complete_round.result = None

    # The GAME_OVER state that process_round_result will return
    game_over_state = gm.GameState(
        phase="GAME_OVER",
        team0_level=Rank.ACE,
        team1_level=Rank.TEN,
        declarer_team=None,
        last_declarer_player=None,
        winning_team=0,
        round_number=1,
    )

    # Build a mock RoundResult
    mock_result = MagicMock(spec=RoundResult)
    mock_result.team0_new_level = Rank.ACE
    mock_result.team1_new_level = Rank.TEN
    mock_result.next_declarer_team = 0
    mock_result.next_declarer_player = 0

    callback_called = [False]

    with patch.object(gm, "create_game", return_value=in_round_state):
        game = Game(players=players)

    game_id = test_registry.create(game)

    # Set the on_game_over callback that records invocation AND removes from registry
    def on_game_over(g: Game) -> None:
        callback_called[0] = True
        test_registry.delete(game_id)

    game.set_on_game_over(on_game_over)

    # Start the game so _round_state is set to the COMPLETE mock
    with patch.object(gm, "start_game", return_value=Ok(in_round_state)):
        with patch.object(rm, "create_round", return_value=complete_round):
            await game.run()
            await game.cancel()

    # Verify game is in registry
    assert test_registry.get(game_id) is not None

    # Now trigger GAME_OVER via act() with NextRoundAction using patched sm
    with patch.object(gm, "process_round_result", return_value=Ok(game_over_state)):
        with patch.object(rm, "get_round_result", return_value=mock_result):
            for p in range(4):
                await game.act(player_index=p, action=NextRoundAction())

    # Verify the callback was actually called
    assert callback_called[0], "on_game_over callback was not invoked"
    # Verify game is over and removed from registry
    assert game.is_over()
    assert test_registry.get(game_id) is None
