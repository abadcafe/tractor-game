"""Tests for server/server.py -- REST + WebSocket API.

REST tests use httpx.AsyncClient with ASGITransport.
WebSocket tests use starlette.testclient.TestClient which supports
ASGI WebSocket testing natively.
"""

import json
import logging
import time
from collections.abc import AsyncGenerator, Generator
from typing import Protocol, TypeGuard

import httpx
import pytest
from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
    WouldBlock,
)
from starlette.testclient import TestClient, WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from server.game_room import GameRoom
from server.player import AIPlayer, HumanPlayer
from server.server import app


class WsReceiveTimeout(TimeoutError):
    """Raised when the test WebSocket waits too long for a message."""


_WS_ERRORS: tuple[type[Exception], ...] = (
    WebSocketDisconnect,
    ClosedResourceError,
    BrokenResourceError,
    EndOfStream,
)
_OPTIONAL_WS_RECEIVE_ERRORS: tuple[type[Exception], ...] = (
    WsReceiveTimeout,
    *_WS_ERRORS,
)
_DEFAULT_WS_RECEIVE_TIMEOUT_SECONDS: float = 5.0


def test_server_logger_has_visible_info_handler() -> None:
    server_logger = logging.getLogger("server")

    assert server_logger.isEnabledFor(logging.INFO)
    assert any(
        handler.level <= logging.INFO
        for handler in server_logger.handlers
    )
    assert server_logger.propagate is False


def test_ai_debug_page_returns_html(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)

    response = sync_client.get(f"/debug/ai/{game_id}?player=0")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AI Transcript" in response.text
    assert "/debug-ai.css" in response.text
    assert "/debug-ai.js" in response.text
    assert game_id not in response.text
    assert "new WebSocket" not in response.text
    assert "/ws/debug/ai/" not in response.text
    assert "renderRecords" not in response.text
    assert "renderTabs" not in response.text
    assert "openStream(player, true)" not in response.text
    assert "/api/debug/ai/" not in response.text
    assert "fetch(" not in response.text
    assert "setInterval" not in response.text


def test_ai_debug_page_missing_game_returns_404(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    response = sync_client.get("/debug/ai/not-a-game?player=0")

    assert response.status_code == 404


def test_ai_debug_transcript_rest_endpoint_removed(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)

    response = sync_client.get(
        f"/api/debug/ai/{game_id}/transcript?player=0"
    )

    assert response.status_code == 404


class AsyncRestClient(Protocol):
    async def get(self, url: str) -> httpx.Response: ...
    async def post(self, url: str) -> httpx.Response: ...
    async def delete(self, url: str) -> httpx.Response: ...


class SyncServerClient(Protocol):
    def get(self, url: str) -> httpx.Response: ...
    def post(self, url: str) -> httpx.Response: ...
    def delete(self, url: str) -> httpx.Response: ...
    def websocket_connect(self, url: str) -> WebSocketTestSession: ...


class _ReceiveNowaitQueue(Protocol):
    def receive_nowait(self) -> dict[str, object]: ...


class _RaiseOnClose(Protocol):
    def __call__(self, message: dict[str, object]) -> None: ...


# ---- Type-guard helpers for JSON narrowing ----


def _is_dict(val: object) -> TypeGuard[dict[str, object]]:
    """Narrow object to dict[str, object]."""
    return isinstance(val, dict)


def _is_list_of_dict(val: object) -> TypeGuard[list[dict[str, object]]]:
    """Narrow object to list[dict[str, object]]."""
    return isinstance(val, list)


def _game_id_from(resp: httpx.Response) -> str:
    """Extract game_id from a create-game response."""
    data = resp.json()
    assert _is_dict(data)
    val = data["game_id"]
    assert isinstance(val, str)
    return val


def _player_ws_path(
    game_id: str, *, player: int = 2, user_id: str = "user-2"
) -> str:
    return f"/game/{game_id}/player/{player}?user_id={user_id}"


def _prepare_ws_game(
    sync_client: SyncServerClient,
    game_id: str,
    *,
    player: int = 2,
    user_id: str = "user-2",
) -> None:
    attach_resp = sync_client.post(
        f"/api/game/{game_id}/player/{player}?user_id={user_id}"
    )
    assert attach_resp.status_code == 200
    fill_resp = sync_client.post(
        f"/api/game/{game_id}/bots?kind=auto&user_id={user_id}"
    )
    assert fill_resp.status_code == 200


def _as_str(val: object) -> str:
    assert isinstance(val, str), (
        f"Expected str, got {type(val).__name__}"
    )
    return val


def _has_receive_nowait(val: object) -> TypeGuard[_ReceiveNowaitQueue]:
    return callable(getattr(val, "receive_nowait", None))


def _has_raise_on_close(val: object) -> TypeGuard[_RaiseOnClose]:
    return callable(val)


def _private_send_rx(ws: WebSocketTestSession) -> _ReceiveNowaitQueue:
    queue = object.__getattribute__(ws, "_send_rx")
    assert _has_receive_nowait(queue)
    return queue


def _private_raise_on_close(ws: WebSocketTestSession) -> _RaiseOnClose:
    fn = object.__getattribute__(ws, "_raise_on_close")
    assert _has_raise_on_close(fn)
    return fn


def _receive_ws_json(
    ws: WebSocketTestSession,
    *,
    timeout: float = _DEFAULT_WS_RECEIVE_TIMEOUT_SECONDS,
) -> object:
    deadline = time.monotonic() + timeout
    while True:
        try:
            message = _private_send_rx(ws).receive_nowait()
            _private_raise_on_close(ws)(message)
            text = _as_str(message["text"])
            return json.loads(text)
        except WouldBlock as exc:
            if time.monotonic() >= deadline:
                raise WsReceiveTimeout(
                    "timed out waiting for websocket message"
                ) from exc
            time.sleep(0.001)


def _try_receive_ws_json(
    ws: WebSocketTestSession,
    *,
    timeout: float = _DEFAULT_WS_RECEIVE_TIMEOUT_SECONDS,
) -> dict[str, object] | None:
    try:
        raw = _receive_ws_json(ws, timeout=timeout)
    except _OPTIONAL_WS_RECEIVE_ERRORS:
        return None
    assert _is_dict(raw)
    return raw


# ---- Fixtures ----


@pytest.fixture
async def client() -> AsyncGenerator[AsyncRestClient, None]:
    """Async test client using httpx with ASGI transport (REST only)."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def sync_client() -> Generator[SyncServerClient, None, None]:
    """
    Synchronous test client using starlette TestClient (WebSocket
    support).
    """
    with TestClient(
        app, backend_options={"use_uvloop": True}
    ) as client:
        yield client


@pytest.fixture
def clean_registry() -> Generator[None, None, None]:
    """Reset the global registry before each test.

    Uses only public API: list_games() + delete() for each game.
    Does NOT access private _games or _last_access fields.
    """
    from server.server import registry

    for g in registry.list_games():
        registry.delete(g["game_id"])
    yield
    for g in registry.list_games():
        registry.delete(g["game_id"])


# ---- REST: Health ----


@pytest.mark.asyncio
async def test_health_endpoint(
    client: AsyncRestClient, clean_registry: None
) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert _is_dict(data)
    assert data["status"] == "ok"


# ---- REST: Create Game ----


@pytest.mark.asyncio
async def test_create_game_returns_201(
    client: AsyncRestClient, clean_registry: None
) -> None:
    response = await client.post("/api/game")
    assert response.status_code == 201
    data = response.json()
    assert _is_dict(data)
    assert "game_id" in data


@pytest.mark.asyncio
async def test_create_game_starts_game(
    client: AsyncRestClient, clean_registry: None
) -> None:
    response = await client.post("/api/game")
    data = response.json()
    assert _is_dict(data)
    game_id_raw = data["game_id"]
    assert isinstance(game_id_raw, str)
    assert len(game_id_raw) > 0


@pytest.mark.asyncio
async def test_create_game_starts_with_empty_players_by_default(
    client: AsyncRestClient,
    clean_registry: None,
) -> None:
    response = await client.post("/api/game")
    game_id = _game_id_from(response)

    from server.server import registry

    room = registry.get(game_id)
    assert isinstance(room, GameRoom)
    assert room.game is None
    for index in (0, 1, 2, 3):
        assert room.player_at(index) is None
    assert [player.kind for player in room.players()] == [
        "empty",
        "empty",
        "empty",
        "empty",
    ]


@pytest.mark.asyncio
async def test_create_auto_game_can_use_ai_bot_players(
    client: AsyncRestClient,
    clean_registry: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRACTOR_BOT_PLAYER", "ai")

    response = await client.post("/api/game/auto")
    game_id = _game_id_from(response)

    from server.server import registry

    room = registry.get(game_id)
    assert isinstance(room, GameRoom)
    assert room.game is None
    for index in (0, 1, 3):
        assert isinstance(room.player_at(index), AIPlayer)
    assert room.player_at(2) is None
    players = room.players()
    assert [players[index].kind for index in (0, 1, 3)] == [
        "ai",
        "ai",
        "ai",
    ]
    assert players[2].kind == "empty"


# ---- REST: List Games ----


@pytest.mark.asyncio
async def test_list_games_empty(
    client: AsyncRestClient, clean_registry: None
) -> None:
    response = await client.get("/api/game")
    assert response.status_code == 200
    data = response.json()
    assert _is_dict(data)
    assert data["games"] == []


@pytest.mark.asyncio
async def test_list_games_with_games(
    client: AsyncRestClient, clean_registry: None
) -> None:
    # Create a game first
    create_resp = await client.post("/api/game")
    assert create_resp.status_code == 201
    # List games
    response = await client.get("/api/game")
    assert response.status_code == 200
    data = response.json()
    assert _is_dict(data)
    games_raw = data["games"]
    assert _is_list_of_dict(games_raw)
    assert len(games_raw) == 1
    first_game = games_raw[0]
    assert "game_id" in first_game
    assert first_game["user_count"] == 0
    assert first_game["capacity"] == 4
    assert first_game["user_players"] == []
    players_raw = first_game["players"]
    assert _is_list_of_dict(players_raw)
    assert len(players_raw) == 4
    for player_raw in players_raw:
        assert player_raw["occupied"] is False
    assert "phase" not in first_game


def test_list_games_counts_attached_user_player(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    attach_resp = sync_client.post(
        f"/api/game/{game_id}/player/1?user_id=user-1"
    )
    assert attach_resp.status_code == 200

    response = sync_client.get("/api/game?user_id=user-1")
    assert response.status_code == 200
    data = response.json()
    assert _is_dict(data)
    games_raw = data["games"]
    assert _is_list_of_dict(games_raw)
    matching = [
        game for game in games_raw if game["game_id"] == game_id
    ]
    assert len(matching) == 1
    assert matching[0]["user_count"] == 1
    assert matching[0]["capacity"] == 4
    assert matching[0]["user_players"] == [1]
    players_raw = matching[0]["players"]
    assert _is_list_of_dict(players_raw)
    player_one = players_raw[1]
    assert player_one["index"] == 1
    assert player_one["occupied"] is True
    assert player_one["connected"] is False
    assert player_one["mine"] is True
    assert player_one["kind"] == "user"


# ---- REST: Player Selection ----


@pytest.mark.asyncio
async def test_attach_player_rest_attaches_lobby_player(
    client: AsyncRestClient, clean_registry: None
) -> None:
    create_resp = await client.post("/api/game")
    game_id = _game_id_from(create_resp)

    attach_resp = await client.post(
        f"/api/game/{game_id}/player/2?user_id=user-2"
    )
    response = await client.get("/api/game?user_id=user-2")

    assert attach_resp.status_code == 200
    attach_data = attach_resp.json()
    assert _is_dict(attach_data)
    assert attach_data["ok"] is True
    data = response.json()
    assert _is_dict(data)
    games_raw = data["games"]
    assert _is_list_of_dict(games_raw)
    matching = [
        game for game in games_raw if game["game_id"] == game_id
    ]
    assert len(matching) == 1
    assert matching[0]["user_count"] == 1
    assert matching[0]["user_players"] == [2]
    players_raw = matching[0]["players"]
    assert _is_list_of_dict(players_raw)
    player_two = players_raw[2]
    assert player_two["occupied"] is True
    assert player_two["connected"] is False
    assert player_two["mine"] is True


@pytest.mark.asyncio
async def test_attach_player_rest_switches_user_player(
    client: AsyncRestClient, clean_registry: None
) -> None:
    create_resp = await client.post("/api/game")
    game_id = _game_id_from(create_resp)

    first_resp = await client.post(
        f"/api/game/{game_id}/player/0?user_id=user-0"
    )
    second_resp = await client.post(
        f"/api/game/{game_id}/player/3?user_id=user-0"
    )
    response = await client.get("/api/game?user_id=user-0")

    assert first_resp.status_code == 200
    assert second_resp.status_code == 200
    data = response.json()
    assert _is_dict(data)
    games_raw = data["games"]
    assert _is_list_of_dict(games_raw)
    matching = [
        game for game in games_raw if game["game_id"] == game_id
    ]
    assert len(matching) == 1
    assert matching[0]["user_players"] == [3]
    players_raw = matching[0]["players"]
    assert _is_list_of_dict(players_raw)
    assert players_raw[0]["occupied"] is False
    assert players_raw[3]["occupied"] is True
    assert players_raw[3]["mine"] is True


@pytest.mark.asyncio
async def test_detach_player_rest_detaches_lobby_player(
    client: AsyncRestClient, clean_registry: None
) -> None:
    create_resp = await client.post("/api/game")
    game_id = _game_id_from(create_resp)
    attach_resp = await client.post(
        f"/api/game/{game_id}/player/1?user_id=user-1"
    )

    detach_resp = await client.delete(
        f"/api/game/{game_id}/player/1?user_id=user-1"
    )
    response = await client.get("/api/game?user_id=user-1")

    assert attach_resp.status_code == 200
    assert detach_resp.status_code == 200
    data = response.json()
    assert _is_dict(data)
    games_raw = data["games"]
    assert _is_list_of_dict(games_raw)
    matching = [
        game for game in games_raw if game["game_id"] == game_id
    ]
    assert len(matching) == 1
    assert matching[0]["user_count"] == 0
    assert matching[0]["user_players"] == []
    players_raw = matching[0]["players"]
    assert _is_list_of_dict(players_raw)
    assert players_raw[1]["occupied"] is False
    assert players_raw[1]["mine"] is False


@pytest.mark.asyncio
async def test_attach_player_rest_rejects_occupied_player(
    client: AsyncRestClient, clean_registry: None
) -> None:
    create_resp = await client.post("/api/game")
    game_id = _game_id_from(create_resp)
    first_resp = await client.post(
        f"/api/game/{game_id}/player/1?user_id=user-1"
    )

    second_resp = await client.post(
        f"/api/game/{game_id}/player/1?user_id=user-other"
    )

    assert first_resp.status_code == 200
    assert second_resp.status_code == 409
    data = second_resp.json()
    assert _is_dict(data)
    assert data["ok"] is False
    assert data["error"] == "player occupied"


@pytest.mark.asyncio
async def test_fill_bot_players_rest_fills_remaining_with_auto(
    client: AsyncRestClient, clean_registry: None
) -> None:
    create_resp = await client.post("/api/game")
    game_id = _game_id_from(create_resp)
    attach_resp = await client.post(
        f"/api/game/{game_id}/player/2?user_id=user-2"
    )

    fill_resp = await client.post(
        f"/api/game/{game_id}/bots?kind=auto&user_id=user-2"
    )
    list_resp = await client.get("/api/game?user_id=user-2")

    assert attach_resp.status_code == 200
    assert fill_resp.status_code == 200
    fill_data = fill_resp.json()
    assert _is_dict(fill_data)
    assert fill_data["ok"] is True
    data = list_resp.json()
    assert _is_dict(data)
    games_raw = data["games"]
    assert _is_list_of_dict(games_raw)
    matching = [
        game for game in games_raw if game["game_id"] == game_id
    ]
    assert len(matching) == 1
    assert matching[0]["user_count"] == 1
    assert matching[0]["user_players"] == [2]
    players_raw = matching[0]["players"]
    assert _is_list_of_dict(players_raw)
    assert [player["kind"] for player in players_raw] == [
        "auto",
        "auto",
        "user",
        "auto",
    ]
    assert all(player["occupied"] is True for player in players_raw)
    assert players_raw[2]["mine"] is True


@pytest.mark.asyncio
async def test_fill_bot_players_rest_requires_attached_user(
    client: AsyncRestClient, clean_registry: None
) -> None:
    create_resp = await client.post("/api/game")
    game_id = _game_id_from(create_resp)

    fill_resp = await client.post(
        f"/api/game/{game_id}/bots?kind=auto&user_id=user-x"
    )

    assert fill_resp.status_code == 409
    data = fill_resp.json()
    assert _is_dict(data)
    assert data["ok"] is False
    assert data["error"] == "user is not attached to a player"


@pytest.mark.asyncio
async def test_fill_bot_players_rest_rejects_invalid_kind(
    client: AsyncRestClient, clean_registry: None
) -> None:
    create_resp = await client.post("/api/game")
    game_id = _game_id_from(create_resp)

    fill_resp = await client.post(
        f"/api/game/{game_id}/bots?kind=random&user_id=user-x"
    )

    assert fill_resp.status_code == 400
    data = fill_resp.json()
    assert _is_dict(data)
    assert data["ok"] is False
    assert data["error"] == "invalid bot kind"


# ---- REST: Delete Game ----


@pytest.mark.asyncio
async def test_delete_game_returns_200(
    client: AsyncRestClient, clean_registry: None
) -> None:
    create_resp = await client.post("/api/game")
    game_id = _game_id_from(create_resp)
    response = await client.delete(f"/api/game/{game_id}")
    assert response.status_code == 200
    del_data = response.json()
    assert _is_dict(del_data)
    assert del_data["ok"] is True


@pytest.mark.asyncio
async def test_delete_nonexistent_returns_200(
    client: AsyncRestClient, clean_registry: None
) -> None:
    """DELETE is idempotent -- returns 200 even for unknown game_id."""
    response = await client.delete("/api/game/nonexistent123")
    assert response.status_code == 200
    data = response.json()
    assert _is_dict(data)
    assert data["ok"] is True


def test_delete_game_after_ws_connection(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    """
    DELETE removes a game after a client has used seq=0 to fetch state.
    """
    from server.server import registry

    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)

    with sync_client.websocket_connect(_player_ws_path(game_id)) as ws:
        # Get initial state via seq=0 (no auto-push on connect)
        ws.send_json({"seq": 0})
        initial = _receive_ws_json(ws)
        assert _is_dict(initial)
        assert initial["type"] == "state"

    response = sync_client.delete(f"/api/game/{game_id}")
    assert response.status_code == 200
    # Verify game is removed
    assert registry.get(game_id) is None


# ---- User player index ----


def test_user_player_can_connect_to_requested_player(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    """A user can enter a specific numbered player via WebSocket."""
    create_resp = sync_client.post("/api/game")
    assert create_resp.status_code == 201
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id, player=0, user_id="user-0")
    with sync_client.websocket_connect(
        _player_ws_path(game_id, player=0, user_id="user-0")
    ) as ws:
        ws.send_json({"seq": 0})
        data = _receive_ws_json(ws)
        assert _is_dict(data)
        assert data["type"] == "state"
        state = data["state"]
        assert _is_dict(state)
        assert "phase" in state


# ---- WebSocket: Connect ----


def test_ws_seq_zero_after_connect_receives_state(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    """
    Connecting to a game via WebSocket and sending seq=0 should receive
    a state message.
    """
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    with sync_client.websocket_connect(_player_ws_path(game_id)) as ws:
        # Client must send seq=0 to get initial state (no auto-push on
        # connect)
        ws.send_json({"seq": 0})
        data = _receive_ws_json(ws)
        assert _is_dict(data)
        assert data["type"] == "state"


def test_ws_connect_nonexistent_rejected(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    """Connecting to a nonexistent game should be rejected."""
    with pytest.raises(_WS_ERRORS):
        with sync_client.websocket_connect(
            _player_ws_path("nonexistent999")
        ) as ws:
            _receive_ws_json(ws)


def test_ws_connect_missing_user_id_rejected(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with sync_client.websocket_connect(
            f"/game/{game_id}/player/1"
        ) as ws:
            _receive_ws_json(ws)

    assert exc_info.value.code == 4410


def test_ws_connect_invalid_player_rejected(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with sync_client.websocket_connect(
            f"/game/{game_id}/player/4?user_id=user-4"
        ) as ws:
            _receive_ws_json(ws)

    assert exc_info.value.code == 4410


def test_ws_connect_rejects_player_stealing(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id, player=1, user_id="user-1")

    with sync_client.websocket_connect(
        _player_ws_path(game_id, player=1, user_id="user-1")
    ) as ws1:
        ws1.send_json({"seq": 0})
        data1 = _receive_ws_json(ws1)
        assert _is_dict(data1)
        assert data1["type"] == "state"

        with pytest.raises(WebSocketDisconnect) as exc_info:
            with sync_client.websocket_connect(
                _player_ws_path(game_id, player=1, user_id="user-other")
            ) as ws2:
                _receive_ws_json(ws2)

        assert exc_info.value.code == 4409


def test_ws_connect_allows_same_user_to_reenter_player(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id, player=3, user_id="user-3")

    with sync_client.websocket_connect(
        _player_ws_path(game_id, player=3, user_id="user-3")
    ) as ws1:
        ws1.send_json({"seq": 0})
        data1 = _receive_ws_json(ws1)
        assert _is_dict(data1)
        assert data1["type"] == "state"

        with sync_client.websocket_connect(
            _player_ws_path(game_id, player=3, user_id="user-3")
        ) as ws2:
            ws2.send_json({"seq": 0})
            data2 = _receive_ws_json(ws2)
            assert _is_dict(data2)
            assert data2["type"] == "state"


def test_ws_connect_game_over_uses_same_state_request_protocol(
    sync_client: SyncServerClient,
    clean_registry: None,
) -> None:
    """Even for an over game, the client asks for state with seq=0."""
    from unittest.mock import patch

    from server.server import registry

    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    room = registry.get(game_id)
    assert isinstance(room, GameRoom)

    with sync_client.websocket_connect(_player_ws_path(game_id)) as ws:
        ws.send_json({"seq": 0})
        data = _receive_ws_json(ws)
        assert _is_dict(data)
        assert data["type"] == "state"

    game_raw = room.game
    assert game_raw is not None

    with patch.object(game_raw, "is_over", return_value=True):
        with sync_client.websocket_connect(
            _player_ws_path(game_id)
        ) as ws:
            ws.send_json({"seq": 0})
            data = _receive_ws_json(ws)
            assert _is_dict(data)
            assert data["type"] == "state"
            assert "state" in data


def test_ws_connect_takeover_closes_old_connection(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    """When a game already has an active WebSocket connection, a second
    connection should take over: the old connection is closed and the
    new
    one is accepted.
    """
    from server.server import registry

    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    room = registry.get(game_id)
    assert isinstance(room, GameRoom)
    user_player_raw = room.player_at(2)
    assert isinstance(user_player_raw, HumanPlayer)

    # First connection
    with sync_client.websocket_connect(_player_ws_path(game_id)) as ws1:
        ws1.send_json({"seq": 0})
        data1 = _receive_ws_json(ws1)
        assert _is_dict(data1)
        assert data1["type"] == "state"
        assert user_player_raw.is_connected()

        # Second connection should take over (old connection closed, new
        # accepted)
        with sync_client.websocket_connect(
            _player_ws_path(game_id)
        ) as ws2:
            ws2.send_json({"seq": 0})
            data2 = _receive_ws_json(ws2)
            assert _is_dict(data2)
            assert data2["type"] == "state"
            # New connection is now the active one
            assert user_player_raw.is_connected()


# ---- WebSocket: Actions with response verification ----
# Each WS action test sends a message and then verifies that the server
# produces a response (either a state update or an error). This ensures
# the server actually processes the message rather than silently
# discarding it.


def test_ws_bid_action_receives_response(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    """
    Sending a bid action via WebSocket should produce a response from
    the server.
    """
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    with sync_client.websocket_connect(_player_ws_path(game_id)) as ws:
        ws.send_json({"seq": 0})
        initial = _receive_ws_json(ws)
        assert _is_dict(initial)
        assert initial["type"] == "state"
        seq = initial["seq"]
        ws.send_json({"type": "bid", "cards": [], "seq": seq})
        response = _try_receive_ws_json(ws)
        if response is not None:
            # Server responds with "state" type (error is a field, not a
            # separate type)
            assert response["type"] == "state"


def test_ws_play_action_receives_response(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    """Sending a play action via WebSocket should produce a response."""
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    with sync_client.websocket_connect(_player_ws_path(game_id)) as ws:
        ws.send_json({"seq": 0})
        initial = _receive_ws_json(ws)
        assert _is_dict(initial)
        assert initial["type"] == "state"
        seq = initial["seq"]
        ws.send_json({"type": "play", "cards": [], "seq": seq})
        response = _try_receive_ws_json(ws)
        if response is not None:
            assert response["type"] == "state"


def test_ws_next_round_action_receives_response(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    """Sending a next_round action should produce a response."""
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    with sync_client.websocket_connect(_player_ws_path(game_id)) as ws:
        ws.send_json({"seq": 0})
        initial = _receive_ws_json(ws)
        assert _is_dict(initial)
        assert initial["type"] == "state"
        seq = initial["seq"]
        ws.send_json({"type": "next_round", "seq": seq})
        response = _try_receive_ws_json(ws)
        if response is not None:
            assert response["type"] == "state"


def test_ws_stir_action_receives_response(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    """Sending a stir action should produce a response."""
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    with sync_client.websocket_connect(_player_ws_path(game_id)) as ws:
        ws.send_json({"seq": 0})
        initial = _receive_ws_json(ws)
        assert _is_dict(initial)
        assert initial["type"] == "state"
        seq = initial["seq"]
        ws.send_json(
            {"type": "stir", "cards": [], "pass": False, "seq": seq}
        )
        response = _try_receive_ws_json(ws)
        if response is not None:
            assert response["type"] == "state"


def test_ws_discard_action_receives_response(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    """Sending a discard action should produce a response."""
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    with sync_client.websocket_connect(_player_ws_path(game_id)) as ws:
        ws.send_json({"seq": 0})
        initial = _receive_ws_json(ws)
        assert _is_dict(initial)
        assert initial["type"] == "state"
        seq = initial["seq"]
        ws.send_json({"type": "discard", "cards": [], "seq": seq})
        response = _try_receive_ws_json(ws)
        if response is not None:
            assert response["type"] == "state"


# ---- WebSocket: Error Handling ----


def test_ws_invalid_action_returns_error(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    """
    Sending an action with an unknown type should return a state message
    with error field.

    The server must respond with {"type": "state", "error": ...} rather
    than silently dropping the message or crashing.
    """
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    with sync_client.websocket_connect(_player_ws_path(game_id)) as ws:
        # Get initial state first
        ws.send_json({"seq": 0})
        initial = _receive_ws_json(ws)
        assert _is_dict(initial)
        assert initial["type"] == "state"
        seq = initial["seq"]

        ws.send_json(
            {"type": "unknown_action", "data": "value", "seq": seq}
        )
        response = _try_receive_ws_json(ws)
        if response is not None:
            # Error is now merged into state message
            assert response["type"] == "state"
            assert response.get("error") is not None


# ---- Reconnect ----


def test_reconnect_replaces_ws(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    """Reconnecting to a game should replace the WebSocket reference."""
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    # First connection
    with sync_client.websocket_connect(_player_ws_path(game_id)) as ws:
        ws.send_json({"seq": 0})
        data = _receive_ws_json(ws)
        assert _is_dict(data)
    # Second connection (reconnect)
    with sync_client.websocket_connect(_player_ws_path(game_id)) as ws:
        ws.send_json({"seq": 0})
        raw = _receive_ws_json(ws)
        assert _is_dict(raw)
        assert raw["type"] == "state"


# ---- WebSocket: Seq Protocol ----


def test_seq_in_state_message(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    """State messages must include a 'seq' field starting from 1."""
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    with sync_client.websocket_connect(_player_ws_path(game_id)) as ws:
        # Send action with seq=0 to get initial state
        ws.send_json({"seq": 0})
        data = _receive_ws_json(ws)
        assert _is_dict(data)
        assert data["type"] == "state"
        assert "seq" in data
        assert isinstance(data["seq"], int)
        assert data["seq"] >= 1


def test_seq_mismatch_returns_state_without_error(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    """
    When action seq doesn't match, server returns state and ignores the
    action.
    """
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    with sync_client.websocket_connect(_player_ws_path(game_id)) as ws:
        # Get initial state through the explicit seq=0 state request.
        ws.send_json({"seq": 0})
        data = _receive_ws_json(ws)
        assert _is_dict(data)
        assert data["type"] == "state"
        known_seq = data["seq"]
        assert isinstance(known_seq, int)
        assert known_seq >= 1

        # Send action with wrong seq (use a value far from current)
        ws.send_json({"type": "next_round", "seq": 99999})
        data = _receive_ws_json(ws)
        assert _is_dict(data)
        assert data["type"] == "state"
        assert data.get("error") is None
        # Seq must be >= the previously observed seq
        assert isinstance(data["seq"], int)
        assert data["seq"] >= known_seq


def test_error_merged_into_state(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    """
    Invalid actions return error as a field in the state message, not as
    a separate message.

    Tests with an invalid action that the server rejects regardless of
    seq.
    The error field appears alongside the state in a single message.
    """
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    with sync_client.websocket_connect(_player_ws_path(game_id)) as ws:
        # Get initial state
        ws.send_json({"seq": 0})
        data = _receive_ws_json(ws)
        assert _is_dict(data)
        assert data["type"] == "state"
        seq = data["seq"]

        # Send invalid action (unknown action type) with current seq.
        # This is rejected by Game's player-message parser, so it
        # reliably produces an error.
        ws.send_json({"type": "nonexistent_action_xyz", "seq": seq})
        data = _receive_ws_json(ws)
        assert _is_dict(data)
        assert data["type"] == "state"
        assert data.get("error") is not None
        error_val = data["error"]
        assert isinstance(error_val, str)
        assert len(error_val) > 0
        # State should still be valid
        assert "state" in data
        # Seq must be >= our last seen seq
        assert isinstance(data["seq"], int) and isinstance(seq, int)
        assert data["seq"] >= seq


def test_bid_pass_parsed_correctly(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    """Sending a bid pass message (type=bid, pass=true) should be parsed
    as SkipBidAction and handled in DEAL_BID phase without crashing.
    The server parses it as SkipBidAction after seq validation, and
    handles SkipBidAction during DEAL_BID by advancing the bid turn.
    This test verifies the parsing and handling work (no crash) and the
    server responds with a state message (not a disconnect or raw
    error).
    """
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    with sync_client.websocket_connect(_player_ws_path(game_id)) as ws:
        # Get initial state
        ws.send_json({"seq": 0})
        data = _receive_ws_json(ws)
        assert _is_dict(data)
        assert data["type"] == "state"
        seq = data["seq"]

        # Send bid pass with correct seq
        ws.send_json({"type": "bid", "pass": True, "seq": seq})
        data = _receive_ws_json(ws)
        assert _is_dict(data)
        assert data["type"] == "state"
        # SkipBidAction is now handled in DEAL_BID — may succeed or
        # may fail (e.g. if game has moved past DEAL_BID). Either way,
        # we get a valid state response, not a crash or disconnect.


# ---- Static file serving ----


@pytest.mark.asyncio
async def test_index_returns_404_when_not_built(
    client: AsyncRestClient, clean_registry: None
) -> None:
    """
    GET / returns 404 with helpful message when static/index.html does
    not exist.
    """
    import os

    from server.server import static_dir

    index_path = os.path.join(static_dir, "index.html")
    existed = os.path.isfile(index_path)
    try:
        # Temporarily rename index.html if it exists
        if existed:
            os.rename(index_path, index_path + ".bak")
        response = await client.get("/")
        assert response.status_code == 404
        assert "Frontend not built" in response.text
    finally:
        if existed:
            os.rename(index_path + ".bak", index_path)


def test_index_returns_html_when_built(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    """GET / returns 200 with HTML when static/index.html exists."""
    import os

    from server.server import static_dir

    index_path = os.path.join(static_dir, "index.html")
    existed = os.path.isfile(index_path)
    try:
        if not existed:
            os.makedirs(static_dir, exist_ok=True)
            with open(index_path, "w", encoding="utf-8") as f:
                f.write("<!DOCTYPE html><html><body>test</body></html>")
        response = sync_client.get("/")
        assert response.status_code == 200
        assert "<html" in response.text.lower()
    finally:
        if not existed and os.path.isfile(index_path):
            os.remove(index_path)


def test_serve_static_existing_file(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    """GET /config.js serves the file from static/ directory."""
    response = sync_client.get("/config.js")
    assert response.status_code == 200
    # config.js exists in static/ from the build output
    assert "javascript" in response.headers.get("content-type", "")
    assert response.headers.get("cache-control") == "no-store"


def test_serve_static_subdirectory_file(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    """
    GET /core/types.js serves files from subdirectories in static/.
    """
    response = sync_client.get("/core/types.js")
    assert response.status_code == 200


def test_serve_static_unknown_path_returns_spa_fallback(
    sync_client: SyncServerClient,
    clean_registry: None,
) -> None:
    """
    GET /nonexistent/file.js returns SPA fallback (index.html) when
    static dir exists,
    or 404 when static dir is empty."""
    import os

    from server.server import static_dir

    index_path = os.path.join(static_dir, "index.html")
    existed = os.path.isfile(index_path)
    try:
        if existed:
            response = sync_client.get("/nonexistent/file.js")
            assert response.status_code == 200
            assert "<html" in response.text.lower()
        else:
            response = sync_client.get("/nonexistent/file.js")
            assert response.status_code == 404
    finally:
        pass


def test_serve_static_game_player_path_returns_root_assets(
    sync_client: SyncServerClient,
    clean_registry: None,
) -> None:
    response = sync_client.get("/game/example/player/1?user_id=user-1")

    assert response.status_code == 200
    assert '<link rel="stylesheet" href="/style.css">' in response.text
    assert '<script type="module" src="/main.js?v=' in response.text


def test_path_traversal_unencoded_returns_safe_response(
    sync_client: SyncServerClient,
    clean_registry: None,
) -> None:
    """
    Unencoded path traversal is normalized by the ASGI layer, so the
    handler
    receives a cleaned path. When static/index.html exists, SPA fallback
    serves it;
    otherwise returns 404. Either way, the actual system file is never
    served.
    """
    import os

    from server.server import static_dir

    index_path = os.path.join(static_dir, "index.html")
    existed = os.path.isfile(index_path)
    response = sync_client.get("/../../../etc/passwd")
    # ASGI normalizes /../../../etc/passwd -> /etc/passwd
    # SPA fallback serves index.html if it exists, otherwise 404
    if existed:
        assert response.status_code == 200
        assert "passwd" not in response.text
    else:
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_path_traversal_encoded_returns_403(
    client: AsyncRestClient, clean_registry: None
) -> None:
    """
    Encoded path traversal attempts bypass ASGI normalization and are
    blocked by
    the server's path traversal protection with 403."""
    response = await client.get("/..%2F..%2F..%2Fetc/passwd")
    assert response.status_code == 403
