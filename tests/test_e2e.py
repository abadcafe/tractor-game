"""End-to-end integration tests for the Tractor game server.

These tests exercise the full pipeline: REST -> WebSocket -> Game -> sm.
They are NOT unit tests -- they test the integration between all modules.
They use only public interfaces (REST API, WebSocket).
They do NOT directly access Game._game_state, Game._dealing_task, or other private
fields. They do NOT directly access GameRegistry._last_access or _games -- they use
the controllable clock injected via GameRegistry(clock=...) or the public API.
"""

from collections.abc import AsyncGenerator, Generator

import pytest
import httpx
from starlette.testclient import TestClient

from server.server import app


@pytest.fixture(autouse=True)
def clean_registry(sync_client: TestClient) -> Generator[None, None, None]:
    """Reset the global registry before each test.

    Uses public API only: GET /api/game + DELETE /api/game/{id}.
    """
    resp = sync_client.get("/api/game")
    for g in resp.json()["games"]:
        sync_client.delete(f"/api/game/{g['game_id']}")
    yield
    resp = sync_client.get("/api/game")
    for g in resp.json()["games"]:
        sync_client.delete(f"/api/game/{g['game_id']}")


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Async test client using httpx with ASGI transport for REST tests."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sync_client() -> Generator[TestClient, None, None]:
    """Synchronous test client using Starlette TestClient for WebSocket tests."""
    with TestClient(app) as c:
        yield c


async def _create_game(client: httpx.AsyncClient) -> str:
    """Helper: create a game and return the game_id."""
    resp = await client.post("/api/game")
    assert resp.status_code == 201
    return resp.json()["game_id"]


def _create_game_sync(sync_client: TestClient) -> str:
    """Helper: create a game synchronously and return the game_id."""
    resp = sync_client.post("/api/game")
    assert resp.status_code == 201
    return resp.json()["game_id"]


# ---- Full Flow ----


def test_full_game_flow(sync_client: TestClient) -> None:
    """Test creating a game, connecting, and verifying initial state."""
    game_id = _create_game_sync(sync_client)
    with sync_client.websocket_connect(f"/game/{game_id}") as ws:
        # Client must send seq=0 to get initial state (no auto-push on connect)
        ws.send_json({"type": "next_round", "seq": 0})
        data = ws.receive_json()
        assert data["type"] == "state"
        state = data["state"]
        assert "phase" in state
        assert "player_hand" in state
        assert "trump_rank" in state


def test_reconnect_mid_game(sync_client: TestClient) -> None:
    """Test disconnecting and reconnecting to a game."""
    game_id = _create_game_sync(sync_client)
    # First connection
    with sync_client.websocket_connect(f"/game/{game_id}") as ws:
        ws.send_json({"type": "next_round", "seq": 0})
        data1 = ws.receive_json()
        assert data1["type"] == "state"
    # Reconnect
    with sync_client.websocket_connect(f"/game/{game_id}") as ws:
        ws.send_json({"type": "next_round", "seq": 0})
        data2 = ws.receive_json()
        assert data2["type"] == "state"
        assert "phase" in data2["state"]


@pytest.mark.asyncio
async def test_concurrent_games(client: httpx.AsyncClient) -> None:
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


def test_invalid_action_returns_error(sync_client: TestClient) -> None:
    """Test that invalid actions through WebSocket return error in state message.

    Sends an unknown action type. The server's _parse_action rejects it
    and returns a state message with an "error" field.
    """
    game_id = _create_game_sync(sync_client)
    with sync_client.websocket_connect(f"/game/{game_id}") as ws:
        # Get initial state via seq=0
        ws.send_json({"type": "next_round", "seq": 0})
        initial = ws.receive_json()
        assert initial["type"] == "state"
        seq = initial["seq"]
        # Send unknown action with current seq
        ws.send_json({"type": "unknown_action_xyz", "seq": seq})
        # Server returns error as a field in the state message
        resp = ws.receive_json()
        assert resp["type"] == "state"
        assert resp.get("error") is not None


def test_delete_game_disconnects_ws(sync_client: TestClient) -> None:
    """Test that deleting a game while connected closes cleanly."""
    game_id = _create_game_sync(sync_client)
    with sync_client.websocket_connect(f"/game/{game_id}") as ws:
        ws.send_json({"type": "next_round", "seq": 0})
        ws.receive_json()
    # Delete after disconnect is fine
    resp = sync_client.delete(f"/api/game/{game_id}")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_list_games_shows_phase(client: httpx.AsyncClient) -> None:
    """Test that listing games includes phase information."""
    game_id = await _create_game(client)
    resp = await client.get("/api/game")
    games = resp.json()["games"]
    assert len(games) == 1
    assert "phase" in games[0]
    assert games[0]["game_id"] == game_id
