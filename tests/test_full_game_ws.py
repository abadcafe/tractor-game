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
