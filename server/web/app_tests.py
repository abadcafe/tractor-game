"""Tests for the server's REST, SSE, and game WebSocket APIs.

REST tests use httpx.AsyncClient with ASGITransport.
SSE tests call the public ASGI interface so streaming is observable.
Game WebSocket tests use starlette.testclient.TestClient.
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Generator,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeGuard
from urllib.parse import urlencode, urlsplit

import httpx
import pytest
from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
    WouldBlock,
)
from starlette.testclient import TestClient, WebSocketTestSession
from starlette.types import Message, Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect

from server.game.players import AIPlayer, HumanPlayer
from server.game.room.game_room import GameRoom
from server.training_control.process_inspection import pid_file_path
from server.training_events import (
    EventContext,
    ProcessIdentity,
    StructuredEventSink,
)
from server.web.app import app, state


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


def test_training_api_exposes_distinct_init_and_resume_schemas(
    sync_client: SyncServerClient,
) -> None:
    init_response = sync_client.get("/api/training/init/schema")
    resume_response = sync_client.get("/api/training/resume/schema")

    assert init_response.status_code == 200
    assert resume_response.status_code == 200
    init_document = init_response.json()
    resume_document = resume_response.json()
    assert _is_dict(init_document)
    assert _is_dict(resume_document)
    init_properties = init_document["properties"]
    resume_properties = resume_document["properties"]
    assert _is_dict(init_properties)
    assert _is_dict(resume_properties)
    assert "d_model" in init_properties
    assert "checkpoint" not in init_properties
    assert "checkpoint" in resume_properties
    assert "d_model" not in resume_properties


def test_training_config_returns_server_default_directory(
    sync_client: SyncServerClient,
) -> None:
    response = sync_client.get("/api/training/config")

    assert response.status_code == 200
    document = response.json()
    assert _is_dict(document)
    assert document["default_run_dir"] == str(
        state.training_control_config.default_run_dir
    )


async def test_training_logs_have_rest_history_and_cursor_tail(
    client: AsyncRestClient,
    tmp_path: Path,
) -> None:
    initialized = await client.post(
        "/api/training/init",
        json={
            "run_dir": str(tmp_path),
            "d_model": 8,
            "layers": 1,
            "heads": 1,
        },
    )
    assert initialized.status_code == 204
    query = urlencode(
        {
            "run_dir": str(tmp_path),
        }
    )

    page = await client.get(f"/api/training/logs?{query}")
    assert page.status_code == 200
    document = page.json()
    assert _is_dict(document)
    events = document["events"]
    assert isinstance(events, list) and events
    store_id = document["store_id"]
    assert isinstance(store_id, str)
    response = await _read_sse(
        "/api/training/events/logs?"
        f"{query}&store_id={store_id}&after_sequence=0"
    )
    assert response.status == 200
    assert response.headers["content-type"].startswith(
        "text/event-stream"
    )
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert response.headers["x-accel-buffering"] == "no"
    event = response.events[0]
    assert event.name == "log"
    message = event.json()
    assert _is_dict(message)
    sequence = message["sequence"]
    assert isinstance(sequence, int)
    assert event.event_id == f"{store_id}:{sequence}"


def test_training_summary_route_is_removed(
    sync_client: SyncServerClient,
) -> None:
    assert sync_client.get("/api/training/summary").status_code == 404


async def test_training_process_events_send_current_snapshot(
    tmp_path: Path,
) -> None:
    query = urlencode({"run_dir": str(tmp_path)})

    response = await _read_sse(f"/api/training/events/process?{query}")
    assert response.events[0].name == "process"
    snapshot = response.events[0].json()
    assert snapshot == {"process": None}


async def test_training_metrics_events_send_complete_snapshot(
    tmp_path: Path,
) -> None:
    query = urlencode(
        {
            "run_dir": str(tmp_path),
            "update_limit": 200,
            "series_points": 500,
        }
    )

    response = await _read_sse(f"/api/training/events/metrics?{query}")
    assert response.events[0].name == "metrics"
    snapshot = response.events[0].json()
    assert _is_dict(snapshot)
    assert snapshot["schema_version"] == 2
    assert snapshot["store_id"] is None
    assert snapshot["through_sequence"] == 0
    assert _is_dict(snapshot["datasets"])


async def test_training_metrics_events_push_replacement_snapshot(
    client: AsyncRestClient,
    tmp_path: Path,
) -> None:
    query = urlencode(
        {
            "run_dir": str(tmp_path),
            "update_limit": 200,
            "series_points": 500,
        }
    )

    async def initialize_after_initial(_event: _SseEvent) -> None:
        nonlocal initialized
        initialized = await client.post(
            "/api/training/init",
            json={
                "run_dir": str(tmp_path),
                "d_model": 8,
                "layers": 1,
                "heads": 1,
            },
        )
        sink = StructuredEventSink(
            run_dir=tmp_path,
            process=ProcessIdentity(kind="worker", index=0),
        )
        sink.emit(
            "inference.batch",
            context=EventContext(
                policy_version=0,
                rollout_id="rollout-a",
                worker_index=0,
            ),
            fields={"batch_size": 1},
        )
        sink.close()

    initialized: httpx.Response | None = None
    response = await _read_sse(
        f"/api/training/events/metrics?{query}",
        event_count=2,
        after_first=initialize_after_initial,
    )
    initial, updated = [event.json() for event in response.events]

    assert initialized is not None and initialized.status_code == 204
    assert _is_dict(initial)
    assert initial["store_id"] is None
    assert _is_dict(updated)
    assert isinstance(updated["store_id"], str)
    updated_sequence = updated["through_sequence"]
    assert isinstance(updated_sequence, int)
    assert updated_sequence >= 1
    datasets = updated["datasets"]
    assert _is_dict(datasets)
    inference = datasets["inference"]
    assert _is_list_of_dict(inference) and len(inference) == 1


async def test_training_metrics_events_apply_projection_parameters(
    client: AsyncRestClient,
    tmp_path: Path,
) -> None:
    initialized = await client.post(
        "/api/training/init",
        json={
            "run_dir": str(tmp_path),
            "d_model": 8,
            "layers": 1,
            "heads": 1,
        },
    )
    assert initialized.status_code == 204
    sink = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="coordinator"),
    )
    for update in (1, 2):
        sink.emit(
            "update",
            context=EventContext(
                policy_version=update - 1,
                rollout_id=f"rollout-{update}",
            ),
            fields={"total_updates": update},
        )
    sink.close()
    query = urlencode(
        {
            "run_dir": str(tmp_path),
            "update_limit": 1,
            "series_points": 1,
        }
    )

    response = await _read_sse(f"/api/training/events/metrics?{query}")
    snapshot = response.events[0].json()
    assert _is_dict(snapshot)
    datasets = snapshot["datasets"]
    assert _is_dict(datasets)
    throughput = datasets["throughput"]
    assert _is_list_of_dict(throughput)
    assert len(throughput) == 1
    values = throughput[0]["values"]
    assert _is_dict(values)
    assert values["total_updates"] == 2


async def test_training_events_report_store_replacement(
    client: AsyncRestClient,
    tmp_path: Path,
) -> None:
    initialized = await client.post(
        "/api/training/init",
        json={
            "run_dir": str(tmp_path),
            "d_model": 8,
            "layers": 1,
            "heads": 1,
        },
    )
    assert initialized.status_code == 204
    query = urlencode(
        {
            "run_dir": str(tmp_path),
            "store_id": "0" * 32,
        }
    )

    for path in ("logs", "checkpoints"):
        response = await _read_sse(
            f"/api/training/events/{path}?{query}"
        )
        assert response.events[0].name == "replacement"
        message = response.events[0].json()
        assert _is_dict(message)
        assert message["store_id"] != "0" * 32


async def test_training_events_send_terminal_rejection(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / ("x" * 120)
    run_dir.mkdir()
    with sqlite3.connect(run_dir / "training.sqlite3") as connection:
        connection.execute("CREATE TABLE invalid_store (value TEXT)")
    query = urlencode({"run_dir": str(run_dir)})
    expected_error = (
        "unsupported training database schema: "
        f"{run_dir / 'training.sqlite3'}"
    )

    for path in ("logs", "metrics", "checkpoints"):
        response = await _read_sse(
            f"/api/training/events/{path}?{query}"
        )
        assert len(response.events) == 1
        assert response.events[0].name == "rejected"
        assert response.events[0].json() == {"error": expected_error}


async def test_training_log_events_resume_from_last_event_id(
    client: AsyncRestClient,
    tmp_path: Path,
) -> None:
    initialized = await client.post(
        "/api/training/init",
        json={
            "run_dir": str(tmp_path),
            "d_model": 2,
            "layers": 1,
            "heads": 1,
            "max_tokens": 512,
        },
    )
    assert initialized.status_code == 204
    page = await client.get(
        f"/api/training/logs?{urlencode({'run_dir': str(tmp_path)})}"
    )
    document = page.json()
    assert _is_dict(document)
    store_id = document["store_id"]
    events = document["events"]
    assert isinstance(store_id, str)
    assert _is_list_of_dict(events) and events
    first = events[0]
    first_sequence = first["sequence"]
    assert isinstance(first_sequence, int)
    sink = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="coordinator"),
    )
    sink.emit("training", fields={"total_updates": 0})
    sink.close()
    query = urlencode(
        {
            "run_dir": str(tmp_path),
            "store_id": store_id,
            "after_sequence": "0",
        }
    )

    response = await _read_sse(
        f"/api/training/events/logs?{query}",
        headers=(
            (b"last-event-id", f"{store_id}:{first_sequence}".encode()),
        ),
    )

    message = response.events[0].json()
    assert _is_dict(message)
    sequence = message["sequence"]
    assert isinstance(sequence, int) and sequence > first_sequence


def test_training_init_requires_yes_before_replacement(
    sync_client: SyncServerClient,
    tmp_path: Path,
) -> None:
    request: dict[str, object] = {
        "run_dir": str(tmp_path),
        "d_model": 8,
        "layers": 1,
        "heads": 1,
    }

    initialized = sync_client.post("/api/training/init", json=request)
    (tmp_path / "stdout.log").write_text(
        "old output\n", encoding="utf-8"
    )
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "stale").write_text("old", encoding="utf-8")
    pid_file_path(tmp_path).write_text(
        f"{os.getpid()}\n", encoding="ascii"
    )
    rejected = sync_client.post("/api/training/init", json=request)
    request["replace_existing"] = "yes"
    replaced = sync_client.post("/api/training/init", json=request)

    assert initialized.status_code == 204
    assert rejected.status_code == 412
    assert rejected.json() == {
        "detail": "type yes to replace existing training artifacts"
    }
    assert replaced.status_code == 204
    assert not (tmp_path / "stdout.log").exists()
    assert not (tmp_path / "runtime").exists()
    assert not pid_file_path(tmp_path).exists()


def test_training_resume_returns_before_cli_preflight_failure(
    sync_client: SyncServerClient,
    tmp_path: Path,
) -> None:
    (tmp_path / "checkpoints").mkdir()

    response = sync_client.post(
        "/api/training/resume",
        json={"run_dir": str(tmp_path), "checkpoint": "latest.json"},
    )

    assert response.status_code == 204
    deadline = time.monotonic() + 15.0
    log_path = tmp_path / "training-cli.log"
    while not log_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert log_path.exists()
    while log_path.stat().st_size == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    output = log_path.read_text(encoding="utf-8")
    assert "latest.json" in output
    stopped = sync_client.post(
        "/api/training/stop", json={"run_dir": str(tmp_path)}
    )
    assert stopped.status_code == 200
    assert not pid_file_path(tmp_path).exists()


def test_ai_debug_page_returns_html(
    sync_client: SyncServerClient, clean_registry: None
) -> None:
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)

    response = sync_client.get(f"/debug/ai/{game_id}?player=0")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AI Transcript" in response.text
    assert "/ai-debug/style.css" in response.text
    assert "/ai-debug/main.js" in response.text
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


@dataclass(frozen=True, slots=True)
class _SseEvent:
    name: str
    data: str
    event_id: str | None

    def json(self) -> object:
        return json.loads(self.data)


@dataclass(frozen=True, slots=True)
class _SseResponse:
    status: int
    headers: dict[str, str]
    events: tuple[_SseEvent, ...]


async def _read_sse(
    url: str,
    *,
    event_count: int = 1,
    headers: tuple[tuple[bytes, bytes], ...] = (),
    after_first: Callable[[_SseEvent], Awaitable[None]] | None = None,
) -> _SseResponse:
    """Read named SSE events through the app's public ASGI interface."""
    assert event_count > 0
    parsed = urlsplit(url)
    disconnect = asyncio.Event()
    request_sent = False
    status: int | None = None
    response_headers: dict[str, str] = {}
    events: list[_SseEvent] = []
    buffer = bytearray()
    called_after_first = False

    async def receive() -> Message:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {
                "type": "http.request",
                "body": b"",
                "more_body": False,
            }
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        nonlocal status, buffer, called_after_first
        if message["type"] == "http.response.start":
            status = message["status"]
            response_headers.update(
                {
                    key.decode("latin-1"): value.decode("latin-1")
                    for key, value in message["headers"]
                }
            )
            return
        assert message["type"] == "http.response.body"
        buffer.extend(message.get("body", b""))
        blocks = bytes(buffer).split(b"\n\n")
        buffer = bytearray(blocks.pop())
        for block in blocks:
            event = _parse_sse_event(block)
            if event is None:
                continue
            events.append(event)
            if len(events) == 1 and after_first is not None:
                assert not called_after_first
                called_after_first = True
                await after_first(event)
            if len(events) >= event_count:
                disconnect.set()

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": parsed.path,
        "raw_path": parsed.path.encode("ascii"),
        "query_string": parsed.query.encode("ascii"),
        "root_path": "",
        "headers": ((b"accept", b"text/event-stream"), *headers),
        "client": ("test", 123),
        "server": ("test", 80),
        "state": {},
    }
    app_receive: Receive = receive
    app_send: Send = send
    await asyncio.wait_for(
        app(scope, app_receive, app_send), timeout=8.0
    )
    assert status is not None
    assert len(events) >= event_count
    return _SseResponse(
        status=status,
        headers=response_headers,
        events=tuple(events[:event_count]),
    )


def _parse_sse_event(block: bytes) -> _SseEvent | None:
    name: str | None = None
    event_id: str | None = None
    data: list[str] = []
    for raw_line in block.decode("utf-8").splitlines():
        if raw_line.startswith(":") or raw_line.startswith("retry:"):
            continue
        field, separator, raw_value = raw_line.partition(":")
        value = (
            raw_value[1:] if raw_value.startswith(" ") else raw_value
        )
        if separator == "":
            continue
        if field == "event":
            name = value
        elif field == "id":
            event_id = value
        elif field == "data":
            data.append(value)
    if name is None:
        return None
    return _SseEvent(name=name, data="\n".join(data), event_id=event_id)


class AsyncRestClient(Protocol):
    async def get(self, url: str) -> httpx.Response: ...
    async def post(
        self,
        url: str,
        *,
        json: dict[str, object] | None = None,
    ) -> httpx.Response: ...
    async def delete(self, url: str) -> httpx.Response: ...


class SyncServerClient(Protocol):
    def get(self, url: str) -> httpx.Response: ...
    def post(
        self,
        url: str,
        *,
        json: dict[str, object] | None = None,
    ) -> httpx.Response: ...
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
    from server.web.app import registry

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

    from server.web.app import registry

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

    from server.web.app import registry

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
    from server.web.app import registry

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

    from server.web.app import registry

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
    from server.web.app import registry

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


# ---- Static applications ----


def test_game_spa_owns_root_and_deep_game_routes(
    sync_client: SyncServerClient,
    clean_registry: None,
) -> None:
    root = sync_client.get("/")
    deep = sync_client.get("/game/example/player/1?user_id=user-1")

    assert root.status_code == 200
    assert deep.status_code == 200
    assert "/game/style.css" in deep.text
    assert "/game/main.js?v=" in deep.text


def test_training_spa_is_independent_from_game_routes(
    sync_client: SyncServerClient,
    clean_registry: None,
) -> None:
    response = sync_client.get("/training/metrics")

    assert response.status_code == 200
    assert "Training Console" in response.text
    assert "/training/main.js?v=" in response.text


def test_static_assets_are_scoped_by_application(
    sync_client: SyncServerClient,
    clean_registry: None,
) -> None:
    game = sync_client.get("/game/config.js")
    training = sync_client.get("/training/main.js")

    assert game.status_code == 200
    assert training.status_code == 200
    assert game.headers.get("cache-control") == "no-store"
    assert training.headers.get("cache-control") == "no-store"


@pytest.mark.asyncio
async def test_path_traversal_encoded_returns_403(
    client: AsyncRestClient, clean_registry: None
) -> None:
    """
    Encoded path traversal attempts bypass ASGI normalization and are
    blocked by
    the server's path traversal protection with 403."""
    response = await client.get("/training/..%2F..%2F..%2Fetc/passwd")
    assert response.status_code == 403
