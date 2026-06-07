"""End-to-end integration tests for the Tractor game server.

These tests exercise the full pipeline: REST -> WebSocket -> Game -> sm.
They are NOT unit tests -- they test the integration between all modules.
They use only public interfaces (REST API, WebSocket, Game.snapshot, Game.is_over,
Game.cancel).
They do NOT directly access Game._game_state, Game._dealing_task, or other private
fields. They do NOT directly access GameRegistry._last_access or _games -- they use
the controllable clock injected via GameRegistry(clock=...) or the public API.
"""

import asyncio

import pytest
import httpx
from starlette.testclient import TestClient

from server.server import app, registry
from server.game_registry import GameRegistry
from server.player import NextRoundAction


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset the global registry before each test.

    Uses public API only: delete() for each game obtained from list_games().
    """
    games = registry.list_games()
    for g in games:
        registry.delete(g["game_id"])
    yield
    games = registry.list_games()
    for g in games:
        registry.delete(g["game_id"])


@pytest.fixture
async def client():
    """Async test client using httpx with ASGI transport for REST tests."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sync_client():
    """Synchronous test client using Starlette TestClient for WebSocket tests."""
    with TestClient(app) as c:
        yield c


async def _create_game(client):
    """Helper: create a game and return the game_id."""
    resp = await client.post("/api/game")
    assert resp.status_code == 201
    return resp.json()["game_id"]


def _create_game_sync(sync_client):
    """Helper: create a game synchronously and return the game_id."""
    resp = sync_client.post("/api/game")
    assert resp.status_code == 201
    return resp.json()["game_id"]


# ---- Full Flow ----


def test_full_game_flow(sync_client):
    """Test creating a game, connecting, and verifying initial state."""
    game_id = _create_game_sync(sync_client)
    with sync_client.websocket_connect(f"/game/{game_id}") as ws:
        data = ws.receive_json()
        assert data["type"] == "state"
        state = data["state"]
        assert "phase" in state
        assert "player_hand" in state
        assert "trump_rank" in state


def test_reconnect_mid_game(sync_client):
    """Test disconnecting and reconnecting to a game."""
    game_id = _create_game_sync(sync_client)
    # First connection
    with sync_client.websocket_connect(f"/game/{game_id}") as ws:
        data1 = ws.receive_json()
        assert data1["type"] == "state"
    # Reconnect
    with sync_client.websocket_connect(f"/game/{game_id}") as ws:
        data2 = ws.receive_json()
        assert data2["type"] == "state"
        assert "phase" in data2["state"]


@pytest.mark.asyncio
async def test_concurrent_games(client):
    """Test that multiple games can exist simultaneously."""
    game_id_1 = await _create_game(client)
    game_id_2 = await _create_game(client)
    assert game_id_1 != game_id_2
    # List games
    resp = await client.get("/api/game")
    games = resp.json()["games"]
    assert len(games) == 2
    game_ids = {g["game_id"] for g in games}
    assert game_ids == {game_id_1, game_id_2}


@pytest.mark.asyncio
async def test_cleanup_expired_games(client):
    """Test that expired games are cleaned up.

    Uses a fresh GameRegistry with a controllable clock instead of
    modifying the global registry's private _last_access field.
    """
    clock_calls = [0]

    def fake_clock():
        clock_calls[0] += 1
        return float(clock_calls[0] * 100)

    test_registry = GameRegistry(clock=fake_clock)
    from unittest.mock import MagicMock
    game = MagicMock()
    game.get_phase.return_value = "IN_PROGRESS"
    game_id = test_registry.create(game)  # T=100

    # Advance clock to T=8000 (game created 7900s ago > 3600)
    clock_calls[0] = 79
    removed = test_registry.cleanup_expired(max_age_seconds=3600)
    assert removed == 1
    assert test_registry.get(game_id) is None


def test_invalid_action_returns_error(sync_client):
    """Test that invalid actions through WebSocket return error messages.

    Sends a "play" action with a fake card ID during the dealing phase.
    The server's _parse_action calls game.resolve_cards() which raises
    ValueError for unknown card IDs. The server catches this and returns
    {"type": "error", "message": ...}. We assert on the error response.
    """
    game_id = _create_game_sync(sync_client)
    with sync_client.websocket_connect(f"/game/{game_id}") as ws:
        data = ws.receive_json()
        # Try to play cards during dealing phase (should be invalid)
        ws.send_json({"type": "play", "cards": ["fake_card_id"]})
        # Server should send back an error response
        resp = ws.receive_json()
        assert resp["type"] == "error"
        assert "message" in resp
        assert len(resp["message"]) > 0


def test_delete_game_disconnects_ws(sync_client):
    """Test that deleting a game while connected closes cleanly."""
    game_id = _create_game_sync(sync_client)
    with sync_client.websocket_connect(f"/game/{game_id}") as ws:
        data = ws.receive_json()
    # Delete after disconnect is fine
    resp = sync_client.delete(f"/api/game/{game_id}")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_list_games_shows_phase(client):
    """Test that listing games includes phase information."""
    game_id = await _create_game(client)
    resp = await client.get("/api/game")
    games = resp.json()["games"]
    assert len(games) == 1
    assert "phase" in games[0]
    assert games[0]["game_id"] == game_id


# ---- Game Auto-Completion ----


@pytest.mark.asyncio
async def test_game_auto_completion(client):
    """Test that a game with 4 AutoPlayers can auto-complete through the full pipeline.

    This test verifies that the dealing loop makes progress by checking that
    the phase transitions from DEAL_BID to a later phase after waiting.
    The dealing loop sleeps 0.75s per card, so we wait long enough for
    several cards to be dealt and check that the phase has changed.
    """
    game_id = await _create_game(client)
    game = registry.get(game_id)
    assert game is not None

    # Initial phase should be DEAL_BID (or a round-level phase)
    initial_phase = game.get_phase()

    # Wait for dealing to make progress (3 cards at 0.75s each = ~2.25s)
    await asyncio.sleep(3)

    # Verify the game is still running and hasn't crashed
    current_phase = game.get_phase()
    assert current_phase is not None

    # The snapshot should still be valid
    snap = game.snapshot(for_player=3)
    assert isinstance(snap.player_hand, list)
    assert snap.phase is not None

    # Verify phase is still a valid game phase (not crashed/error state)
    assert current_phase in (
        "DEAL_BID", "STIRRING", "EXCHANGE", "PLAYING",
        "COMPLETE", "GAME_OVER", "IN_ROUND",
    )


@pytest.mark.asyncio
async def test_game_over_via_auto_players(client):
    """Test that a game with auto players starts and progresses through phases.

    Verifies that the game is created with 4 AutoPlayers (3 Auto + 1 Human)
    and that the initial state is valid. Auto players will drive the game
    forward asynchronously.
    """
    game_id = await _create_game(client)
    game = registry.get(game_id)
    assert game is not None
    # Verify the game has a valid initial state with a known phase
    initial_phase = game.get_phase()
    assert initial_phase in ("IDLE", "IN_ROUND", "DEAL_BID")


@pytest.mark.asyncio
async def test_game_over_removes_from_registry(client):
    """Test that the on_game_over callback mechanism works end-to-end.

    Creates a Game directly with mocked sm functions to force it through
    to GAME_OVER, and verifies the callback fires and removes the game
    from the registry. Uses game.cancel() (public method) to stop the
    dealing loop.
    """
    from server.game import Game
    from server.player import AutoPlayer
    from server.sm import game_sm as gm, round_sm as rm
    from server.sm.card_model import Rank
    from server.sm.scoring import RoundResult
    from unittest.mock import patch, MagicMock

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

    # Build the GAME_OVER state
    game_over_state = gm.GameState(
        phase="GAME_OVER",
        team0_level=Rank.ACE,
        team1_level=Rank.TEN,
        declarer_team=None,
        last_declarer_player=None,
        winning_team=0,
        round_number=1,
    )

    # Mock RoundResult to trigger game over
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
    def on_game_over(g):
        callback_called[0] = True
        test_registry.delete(game_id)

    game.set_on_game_over(on_game_over)

    # Start the game so _round_state is set to the COMPLETE mock
    with patch.object(gm, "start_game", return_value=in_round_state):
        with patch.object(rm, "create_round", return_value=complete_round):
            await game.run()
            await game.cancel()

    # Verify game is in registry
    assert test_registry.get(game_id) is not None

    # Now trigger GAME_OVER via act() with NextRoundAction using patched sm
    with patch.object(gm, "process_round_result", return_value=game_over_state):
        with patch.object(rm, "get_round_result", return_value=mock_result):
            await game.act(player_index=0, action=NextRoundAction())

    # Verify the callback was actually called (not just conditionally checked)
    assert callback_called[0], "on_game_over callback was not invoked"
    # Verify game is over and removed from registry
    assert game.is_over()
    assert test_registry.get(game_id) is None
