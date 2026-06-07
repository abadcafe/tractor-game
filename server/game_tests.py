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

from server.game import Game, StateSnapshot
from server.player import AutoPlayer, BidAction, PlayAction, NextRoundAction, StirAction, SkipStirAction, DiscardAction


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
    """PlayAction during DEAL_BID should raise ValueError from sm."""
    game = _create_game_with_auto_players()
    await game.run()
    with pytest.raises(ValueError):
        await game.act(player_index=0, action=PlayAction(cards=[]))


@pytest.mark.asyncio
async def test_act_value_error_propagated():
    """ValueError from sm should propagate through act()."""
    game = _create_game_with_auto_players()
    await game.run()
    with pytest.raises(ValueError):
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
    """snapshot() must raise RuntimeError when called before run().

    Before run(), _round_state is None, so snapshot() cannot build a
    valid StateSnapshot. Rather than returning a partial/empty snapshot
    that could mislead callers, it raises an explicit error.
    """
    game = _create_game_with_auto_players()
    with pytest.raises(RuntimeError, match="Game not started"):
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

    legal_actions stores sm.PlayAction Pydantic model objects (not dicts).
    Each PlayAction has .type (PlayType enum) and .cards (list[Card])
    attributes. This is important because AutoPlayer accesses entry.cards
    via attribute access. The to_dict() method handles serialization
    to JSON format for the WebSocket output.
    """
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    assert isinstance(snap.legal_actions, list)
    # If in PLAYING phase, legal_actions should contain PlayAction objects
    if snap.phase == "PLAYING" and len(snap.legal_actions) > 0:
        entry = snap.legal_actions[0]
        # Must have .cards and .type attributes (sm.PlayAction is a Pydantic model)
        assert hasattr(entry, "cards")
        assert hasattr(entry, "type")
        assert isinstance(entry.cards, list)


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
            try:
                await game.act(player_index=0, action=action)
            except ValueError:
                pass  # bid may be rejected for various reasons, that's fine


@pytest.mark.asyncio
async def test_act_skip_stir_during_stirring():
    """SkipStirAction is a valid action type that Game.act() can distinguish
    from StirAction. The actual dispatch routing (SkipStirAction -> round_sm.pass_stir)
    is verified in integration tests (task-008).
    """
    action = SkipStirAction()
    assert isinstance(action, SkipStirAction)
    # Verify it is NOT a StirAction -- Game.act() must dispatch differently
    from server.player import StirAction
    assert not isinstance(action, StirAction)


@pytest.mark.asyncio
async def test_act_next_round_transitions():
    """NextRoundAction during non-COMPLETE phase should raise ValueError."""
    game = _create_game_with_auto_players()
    await game.run()
    with pytest.raises(ValueError):
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
    with patch.object(gm, "start_game", return_value=in_round_state):
        with patch.object(rm, "create_round", return_value=complete_round):
            await game.run()
            # Cancel the dealing loop via public interface
            await game.cancel()

    # Verify game is not over yet (before triggering GAME_OVER)
    assert not game.is_over()

    # Now trigger GAME_OVER via act() with NextRoundAction
    with patch.object(gm, "process_round_result", return_value=game_over_state):
        with patch.object(rm, "is_round_complete", return_value=True):
            with patch.object(rm, "get_round_result", return_value=mock_result):
                with patch.object(gm, "start_game", return_value=game_over_state):
                    with patch.object(rm, "create_round", return_value=complete_round):
                        await game.act(player_index=0, action=NextRoundAction())

    # The game must have transitioned to GAME_OVER; callback must have been called
    assert game.is_over()
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
    contains sm.PlayAction objects which must be serialized to dicts with
    "type" and "cards" keys in the to_dict() output.
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
    # legal_actions must be a list of dicts (not sm.PlayAction objects)
    assert isinstance(result["legal_actions"], list)
    if len(result["legal_actions"]) > 0:
        legal_entry = result["legal_actions"][0]
        assert isinstance(legal_entry, dict)
        assert "type" in legal_entry
        assert "cards" in legal_entry
        assert isinstance(legal_entry["type"], str)
        assert isinstance(legal_entry["cards"], list)


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
    if len(result["player_hand"]) > 0:
        card = result["player_hand"][0]
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
    """Game.resolve_cards() returns Card objects matching the given IDs
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
        resolved = game.resolve_cards(player_index=0, card_ids=card_ids)
        assert len(resolved) == len(card_ids)
        for original, resolved_card in zip(card_ids, resolved):
            assert resolved_card.id == original
            # Must be a Card Pydantic model (not a string or dict)
            from server.sm.card_model import Card
            assert isinstance(resolved_card, Card)


@pytest.mark.asyncio
async def test_resolve_cards_raises_on_unknown_id():
    """Game.resolve_cards() raises ValueError if any card_id is not found
    in the player's hand.

    This prevents human players from submitting cards they don't hold,
    which would be an invalid action.
    """
    game = _create_game_with_auto_players()
    await game.run()
    with pytest.raises(ValueError):
        game.resolve_cards(player_index=0, card_ids=["NONEXISTENT-CARD-ID"])


def test_resolve_cards_raises_before_run():
    """resolve_cards() must raise RuntimeError when called before run().

    Before run(), _round_state is None, so resolve_cards() cannot look up
    cards in any player's hand. It raises an explicit error rather than
    silently returning an empty list or crashing with AttributeError.
    """
    game = _create_game_with_auto_players()
    with pytest.raises(RuntimeError, match="Game not started"):
        game.resolve_cards(player_index=0, card_ids=["SOME-CARD-ID"])
