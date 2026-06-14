"""Full-game integration tests via server external interfaces (REST + WebSocket).

These tests only use the server's public interfaces: REST API and WebSocket protocol.
They do NOT import any server internal modules (Game, Player, sm, etc.).

Module: test_full_game_ws
Responsibilities: Play complete games through REST+WS external interfaces,
    verifying state transitions, boundary conditions, and error handling.
"""

import time

from collections.abc import Generator
from typing import TypeGuard

import pytest
from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
from starlette.testclient import TestClient, WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from server.server import app

# Connection-related exceptions from third-party libraries (normal flow control).
# Programming errors (AssertionError, TypeError, etc.) must NOT be caught here.
_WS_ERRORS: tuple[type[Exception], ...] = (
    WebSocketDisconnect,
    ClosedResourceError,
    BrokenResourceError,
    EndOfStream,
)


# ---- Type-guard helpers for JSON narrowing ----


def _is_dict(val: object) -> TypeGuard[dict[str, object]]:
    """Narrow object to dict[str, object]."""
    return isinstance(val, dict)


def _as_dict(val: object) -> dict[str, object]:
    """Assert val is dict[str, object] and return it narrowed.

    For use when the caller knows the value is a dict (e.g. from
    ws.receive_json() which returns object). Raises AssertionError
    if the value is not a dict.
    """
    assert _is_dict(val), f"Expected dict, got {type(val).__name__}"
    return val


def _as_int(val: object) -> int:
    """Assert val is int and return it narrowed."""
    assert isinstance(val, int), f"Expected int, got {type(val).__name__}: {val!r}"
    return val


def _as_str(val: object) -> str:
    """Assert val is str and return it narrowed."""
    assert isinstance(val, str), f"Expected str, got {type(val).__name__}: {val!r}"
    return val


def _as_dict_or_none(val: object) -> dict[str, object] | None:
    """Narrow object to dict[str, object] | None."""
    if val is None:
        return None
    assert _is_dict(val), f"Expected dict or None, got {type(val).__name__}"
    return val


# ---- Fixtures ----


@pytest.fixture(autouse=True)
def clean_registry(sync_client: TestClient) -> Generator[None, None, None]:
    """Reset the global registry before each test."""
    resp = sync_client.get("/api/game")
    for g in resp.json()["games"]:
        sync_client.delete(f"/api/game/{g['game_id']}")
    yield
    resp = sync_client.get("/api/game")
    for g in resp.json()["games"]:
        sync_client.delete(f"/api/game/{g['game_id']}")


@pytest.fixture
def sync_client() -> Generator[TestClient, None, None]:
    """Synchronous test client using Starlette TestClient."""
    with TestClient(app) as c:
        yield c


class WsGameDriver:
    """WebSocket game driver with send-and-wait semantics.

    Tracks state sequence numbers, handles auto-reconnection,
    and provides high-level do_* methods for game actions.

    Reconnection strategy: do_action() handles reconnection via its retry
    loop. receive_msg() does NOT auto-reconnect — it raises on disconnect
    so do_action() can reconnect and resend the original action.
    receive_msg_safe() DOES auto-reconnect and is used by wait_for_* and
    the test main loop.
    """

    def __init__(self, sync_client: TestClient) -> None:
        self._client = sync_client
        self._ws: WebSocketTestSession | None = None
        self._current_seq: int = 0
        self.last_error: str | None = None
        self._game_id: str | None = None

    @property
    def current_seq(self) -> int:
        """Current state sequence number (read-only public accessor)."""
        return self._current_seq

    def connect(self, game_id: str) -> None:
        """Connect to a game via WebSocket.

        Does NOT automatically receive initial state.
        Client must send an action with seq=0 to get the initial state.
        """
        self._game_id = game_id
        self._ws = self._client.websocket_connect(f"/game/{game_id}")
        self._current_seq = 0
        self.last_error = None

    def close(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            try:
                self._ws.close()
            except _WS_ERRORS:
                pass
            self._ws = None

    def force_disconnect(self) -> None:
        """Public method to force-close the WebSocket connection.

        Used by the test's async disconnector thread to simulate network
        interruptions. This is the only supported way for external code
        to break the connection — do NOT access _ws directly.
        """
        if self._ws is not None:
            try:
                self._ws.close()
            except _WS_ERRORS:
                pass
            self._ws = None

    def send_action(self, action: dict[str, object]) -> None:
        """Send an action with the current seq number."""
        if self._ws is None:
            raise RuntimeError("Not connected")
        action_with_seq = {**action, "seq": self._current_seq}
        self._ws.send_json(action_with_seq)

    def receive_msg(self) -> dict[str, object]:
        """Receive a message from the server.

        Does NOT auto-reconnect on disconnect. Raises the exception
        so that do_action() can handle reconnection in its retry loop.
        """
        if self._ws is None:
            raise RuntimeError("Not connected")
        raw = self._ws.receive_json()
        return _as_dict(raw)

    def receive_msg_safe(self) -> dict[str, object]:
        """Receive a message with auto-reconnect on disconnect.

        If the WS connection is broken, reconnects and sends seq=0 to get
        the current state. Used by wait_for_* methods and the test main loop.
        do_action() uses raw receive_msg() for its own retry logic.

        Note: On reconnect, this method sends {"type": "next_round", "seq": 0}
        to trigger a seq mismatch and get the current state. This will set
        last_error to a seq-mismatch or phase-mismatch error message. Callers
        should not rely on last_error being clean after reconnection.
        """
        for attempt in range(2):
            try:
                if self._ws is None:
                    if self._game_id is None:
                        raise RuntimeError("Not connected and no game_id")
                    self.connect(self._game_id)
                    ws = self._ws
                    assert ws is not None
                    ws.send_json({"type": "next_round", "seq": 0})
                msg = self.receive_msg()
                if msg.get("type") == "state":
                    self._current_seq = _as_int(msg["seq"])
                return msg
            except _WS_ERRORS:
                if attempt == 0:
                    self._ws = None
                    continue
                raise
        raise RuntimeError("Failed to receive message after reconnect")

    def do_action(self, action: dict[str, object]) -> bool:
        """Send action and wait for response. Returns True if successful, False if error.

        Updates _current_seq on success. Sets last_error on failure.
        Auto-reconnects on WS disconnect (one retry): reconnects, then
        resends the original action with the old seq. If seq doesn't
        match after reconnect, returns False (same as any other error).
        """
        self.last_error = None
        for attempt in range(2):
            try:
                if self._ws is None:
                    if self._game_id is not None:
                        self.connect(self._game_id)
                    else:
                        return False
                self.send_action(action)
                msg = self.receive_msg()
                if msg.get("type") == "state":
                    self._current_seq = _as_int(msg["seq"])
                    error = msg.get("error")
                    if error is not None:
                        self.last_error = _as_str(error)
                        return False
                    return True
                return False
            except _WS_ERRORS:
                if attempt == 0:
                    self._ws = None
                    continue  # Retry once after reconnect
                return False
        return False

    def wait_for_phase(self, phase: str, timeout: float = 30) -> dict[str, object]:
        """Wait until the game reaches the specified phase.

        Returns the state message. Raises TimeoutError if not reached within timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = self.receive_msg_safe()
            if msg.get("type") == "state":
                state = _as_dict_or_none(msg.get("state"))
                if state is not None and state.get("phase") == phase:
                    return msg
        raise TimeoutError(f"Phase '{phase}' not reached within {timeout}s")

    def wait_for_awaiting(self, value: str, timeout: float = 30) -> dict[str, object]:
        """Wait until awaiting_action equals the specified value.

        Returns the state message. Raises TimeoutError if not reached within timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = self.receive_msg_safe()
            if msg.get("type") == "state":
                if msg.get("awaiting") == value:
                    return msg
        raise TimeoutError(f"awaiting='{value}' not reached within {timeout}s")

    def do_bid(self, card_ids: list[str]) -> bool:
        """Bid with the given cards. Returns True if successful."""
        return self.do_action({"type": "bid", "cards": card_ids})

    def do_bid_pass(self) -> bool:
        """Pass during DEAL_BID. Returns True if successful."""
        return self.do_action({"type": "bid", "pass": True})

    def do_stir_pass(self) -> bool:
        """Pass during STIRRING. Returns True if successful."""
        return self.do_action({"type": "stir", "pass": True})

    def do_stir(self, card_ids: list[str]) -> bool:
        """Stir with the given cards. Returns True if successful."""
        return self.do_action({"type": "stir", "cards": card_ids})

    def do_discard(self, card_ids: list[str]) -> bool:
        """Discard the given cards. Returns True if successful."""
        return self.do_action({"type": "discard", "cards": card_ids})

    def do_play(self, card_ids: list[str]) -> bool:
        """Play the given cards. Returns True if successful."""
        return self.do_action({"type": "play", "cards": card_ids})

    def do_next_round(self) -> bool:
        """Confirm next round. Returns True if successful."""
        return self.do_action({"type": "next_round"})


# ---- Infrastructure Tests ----


def test_create_game_returns_201(sync_client: TestClient) -> None:
    """POST /api/game returns 201 with game_id in response body."""
    resp = sync_client.post("/api/game")
    assert resp.status_code == 201
    data = resp.json()
    assert _is_dict(data)
    assert "game_id" in data
    assert isinstance(data["game_id"], str)
    assert len(data["game_id"]) > 0


def test_health_check(sync_client: TestClient) -> None:
    """GET /health returns 200 with status ok."""
    resp = sync_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_games(sync_client: TestClient) -> None:
    """GET /api/game returns a list of games."""
    # Create a game first
    resp = sync_client.post("/api/game")
    data = resp.json()
    assert _is_dict(data)
    game_id = _as_str(data["game_id"])

    resp = sync_client.get("/api/game")
    assert resp.status_code == 200
    resp_data = resp.json()
    # Don't narrow resp_data with _is_dict — it would make resp_data["games"]
    # return `object` (non-iterable in pyright strict). Keep resp_data as Any
    # so list iteration works with proper type narrowing on each element.
    games = resp_data["games"]
    assert len(games) >= 1
    game_ids = [_as_str(g["game_id"]) for g in games]
    assert game_id in game_ids
    # Each game should have a phase field
    for g in games:
        assert "phase" in g


def test_delete_game_closes_ws(sync_client: TestClient) -> None:
    """DELETE a game while WS connected: receive final state push, then WS closes."""
    resp = sync_client.post("/api/game")
    data = resp.json()
    assert _is_dict(data)
    game_id = _as_str(data["game_id"])

    # Step 1: Connect, get initial state, delete, verify final push.
    # Step 2: Exit the context manager — its __exit__ may raise RuntimeError
    # because the server already closed the WS during delete.
    # We separate these so the test assertions are NOT swallowed by the
    # RuntimeError from __exit__.
    final_state_pushed = False
    try:
        with sync_client.websocket_connect(f"/game/{game_id}") as ws:
            # Get initial state
            ws.send_json({"type": "next_round", "seq": 0})
            raw = ws.receive_json()
            data_msg = _as_dict(raw)
            assert data_msg["type"] == "state"

            # Delete the game (while WS is still connected)
            resp = sync_client.delete(f"/api/game/{game_id}")
            assert resp.status_code == 200

            # Should receive a final state push before close
            final_raw = ws.receive_json()
            final = _as_dict(final_raw)
            assert final["type"] == "state"
            final_state = _as_dict_or_none(final.get("state"))
            assert final_state is not None
            assert "phase" in final_state
            final_state_pushed = True

            # Verify WS actually closes after final state push
            with pytest.raises((WebSocketDisconnect, Exception)):
                # Try to receive another message -- should fail because WS is closed
                ws.send_json({"type": "next_round", "seq": 0})
                ws.receive_json()
    except RuntimeError:
        # Server-initiated close during delete causes __exit__ to fail
        # when it tries to close the already-closed WS. This is expected.
        # The assertion for final_state_pushed above ensures the test
        # actually verified the final state push before this point.
        pass

    assert final_state_pushed, "Final state push was not received before WS closed"


def test_connect_nonexistent_game(sync_client: TestClient) -> None:
    """Connecting to a nonexistent game closes with code 4404."""
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with sync_client.websocket_connect("/game/nonexistent_id") as ws:
            ws.receive_json()
    assert exc_info.value.code == 4404


def test_connection_takeover(sync_client: TestClient) -> None:
    """New WS connection kicks the old one; old connection gets closed."""
    resp = sync_client.post("/api/game")
    data = resp.json()
    assert _is_dict(data)
    game_id = _as_str(data["game_id"])

    # First connection: get state
    with sync_client.websocket_connect(f"/game/{game_id}") as ws1:
        ws1.send_json({"type": "next_round", "seq": 0})
        raw1 = ws1.receive_json()
        data1 = _as_dict(raw1)
        assert data1["type"] == "state"

        # Second connection (should kick first)
        with sync_client.websocket_connect(f"/game/{game_id}") as ws2:
            ws2.send_json({"type": "next_round", "seq": 0})
            raw2 = ws2.receive_json()
            data2 = _as_dict(raw2)
            assert data2["type"] == "state"

        # After ws2 context exits, ws1 should be unusable (kicked by server)
        # The server closed ws1 when ws2 connected, so receive should fail.
        # Note: send_json() raises RuntimeError (not in _WS_ERRORS) because
        # the server already sent a close frame, so we only try receive.
        with pytest.raises(_WS_ERRORS):
            ws1.receive_json()


def test_reconnect_resumes_game(sync_client: TestClient) -> None:
    """After disconnect, reconnect with seq=0 gets current state; then actions work."""
    resp = sync_client.post("/api/game")
    data = resp.json()
    assert _is_dict(data)
    game_id = _as_str(data["game_id"])

    # First connection: get state
    with sync_client.websocket_connect(f"/game/{game_id}") as ws1:
        ws1.send_json({"type": "next_round", "seq": 0})
        raw1 = ws1.receive_json()
        data1 = _as_dict(raw1)
        assert data1["type"] == "state"
        seq1 = _as_int(data1["seq"])

    # Reconnect: send seq=0 -> mismatch -> get current state
    with sync_client.websocket_connect(f"/game/{game_id}") as ws2:
        ws2.send_json({"type": "next_round", "seq": 0})
        raw2 = ws2.receive_json()
        data2 = _as_dict(raw2)
        assert data2["type"] == "state"
        # State should be valid and seq should be >= seq1
        assert _as_int(data2["seq"]) >= seq1
        state2 = _as_dict(data2["state"])
        assert "phase" in state2

        # Verify subsequent action works with the correct seq from received state
        current_seq = _as_int(data2["seq"])
        ws2.send_json({"type": "next_round", "seq": current_seq})
        raw3 = ws2.receive_json()
        data3 = _as_dict(raw3)
        assert data3["type"] == "state"
        # Action with correct seq should either succeed (state changes) or
        # be accepted (no seq mismatch error). Either way, seq should advance
        # or stay the same (if no state change occurred).
        assert _as_int(data3["seq"]) >= current_seq
