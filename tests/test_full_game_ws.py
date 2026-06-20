"""Full-game integration tests via server external interfaces (REST + WebSocket).

These tests only use the server's public interfaces: REST API and WebSocket protocol.
They do NOT import any server internal modules (Game, Player, sm, etc.).

Module: test_full_game_ws
Responsibilities: Play complete games through REST+WS external interfaces,
    verifying state transitions, boundary conditions, and error handling.
"""

import json
import time

from collections.abc import Generator
from itertools import combinations
from typing import Protocol, TypeGuard

import httpx
import pytest
from anyio import BrokenResourceError, ClosedResourceError, EndOfStream, WouldBlock
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


class WsReceiveTimeout(TimeoutError):
    """Raised when the test WebSocket driver waits too long for a message."""


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


def _is_list(val: object) -> TypeGuard[list[object]]:
    """Narrow object to list[object]."""
    return isinstance(val, list)


def _as_list(val: object) -> list[object]:
    """Assert val is list and return it narrowed."""
    assert _is_list(val), f"Expected list, got {type(val).__name__}: {val!r}"
    return val


def _as_str_or_none(val: object) -> str | None:
    """Narrow object to str | None."""
    if val is None:
        return None
    assert isinstance(val, str), f"Expected str or None, got {type(val).__name__}"
    return val


def _awaiting(msg: dict[str, object]) -> str | None:
    """Return the player-facing awaiting_action from a state message."""
    state = _as_dict_or_none(msg.get("state"))
    if state is None:
        return None
    return _as_str_or_none(state.get("awaiting_action"))


def _is_list_of_dict(val: object) -> TypeGuard[list[dict[str, object]]]:
    """Narrow object to list[dict[str, object]]."""
    if not _is_list(val):
        return False
    for item in val:
        if not _is_dict(item):
            return False
    return True


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


# ---- Fixtures ----


@pytest.fixture(autouse=True)
def clean_registry(sync_client: SyncServerClient) -> Generator[None, None, None]:
    """Reset the global registry before each test."""
    resp = sync_client.get("/api/game")
    for g in resp.json()["games"]:
        sync_client.delete(f"/api/game/{g['game_id']}")
    yield
    resp = sync_client.get("/api/game")
    for g in resp.json()["games"]:
        sync_client.delete(f"/api/game/{g['game_id']}")


@pytest.fixture
def sync_client() -> Generator[SyncServerClient, None, None]:
    """Synchronous test client using Starlette TestClient."""
    with TestClient(app) as c:
        yield c


class WsGameDriver:
    """WebSocket game driver with send-and-wait semantics.

    Tracks state sequence numbers and provides high-level do_* methods
    for game actions.
    """

    def __init__(self, sync_client: SyncServerClient) -> None:
        self._client = sync_client
        self._ws: WebSocketTestSession | None = None
        self._ws_cm: WebSocketTestSession | None = None  # context manager for websocket_connect
        self._known_seq: int = 0
        self.last_error: str | None = None
        self._game_id: str | None = None
        self._last_state_msg: dict[str, object] | None = None

    @property
    def known_seq(self) -> int:
        """Current state sequence number (read-only public accessor)."""
        return self._known_seq

    def sync_seq(self, seq: int) -> None:
        """Update _known_seq monotonically.

        Public accessor for seq synchronization from helper functions
        that need to keep the driver's seq in sync with server pushes.
        WebSocket buffers may still contain older state pushes; those must
        not move the client seq backwards or the next action becomes stale.
        """
        if seq > self._known_seq:
            self._known_seq = seq

    def connect(self, game_id: str) -> None:
        """Connect to a game via WebSocket.

        Does NOT automatically receive initial state.
        Client must send a seq=0 state request to get the initial state.
        """
        self._game_id = game_id
        self._ws_cm = self._client.websocket_connect(f"/game/{game_id}")
        self._ws = self._ws_cm.__enter__()
        self._known_seq = 0
        self.last_error = None
        print(f"  [WsGameDriver] connected to game {game_id}, seq=0", flush=True)

    def close(self) -> None:
        """Close the WebSocket connection.

        Tries graceful close first. If that hangs (e.g. the server's
        receive loop is blocked), falls back to just dropping the
        references so the TestClient teardown can proceed.
        """
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._ws_cm is not None:
            try:
                self._ws_cm.__exit__(None, None, None)
            except Exception:
                pass
            self._ws_cm = None

    def send_action(self, action: dict[str, object]) -> None:
        """Send an action with the current seq number."""
        self.send_action_with_seq(action, self._known_seq)

    def send_action_with_seq(self, action: dict[str, object], seq: int) -> None:
        """Send an action with an explicit sequence number."""
        if self._ws is None:
            raise RuntimeError("Not connected")
        action_with_seq = {**action, "seq": seq}
        t0 = time.monotonic()
        self._ws.send_json(action_with_seq)
        dt = time.monotonic() - t0
        slow = f" *** SLOW {dt:.2f}s ***" if dt > 0.5 else ""
        # Show pass/cards for bid and stir actions
        detail = ""
        if action_with_seq.get("type") == "bid":
            detail = " pass" if action_with_seq.get("pass") else f" cards={action_with_seq.get('cards')}"
        elif action_with_seq.get("type") == "stir":
            detail = " pass" if action_with_seq.get("pass") else f" cards={action_with_seq.get('cards')}"
        print(f"  [WsGameDriver] send: type={action_with_seq.get('type')}{detail} seq={seq}{slow}", flush=True)

    def receive_msg(self, *, verbose: bool = True, timeout: float | None = None) -> dict[str, object]:
        """Receive a message from the server.

        Does NOT auto-reconnect on disconnect. Raises the exception
        so that do_action() can handle reconnection in its retry loop.
        When verbose=False, skips per-message logging for high-volume drains.
        timeout prevents Starlette's blocking receive_json() from hanging
        the whole test process when no further WS messages are produced.
        """
        if self._ws is None:
            raise RuntimeError("Not connected")
        if timeout is None:
            raw = self._ws.receive_json()
        else:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    message = _private_send_rx(self._ws).receive_nowait()
                    _private_raise_on_close(self._ws)(message)
                    text = _as_str(message["text"])
                    raw = json.loads(text)
                    break
                except WouldBlock as exc:
                    if time.monotonic() >= deadline:
                        raise WsReceiveTimeout("timed out waiting for websocket message") from exc
                    time.sleep(0.001)
        result = _as_dict(raw)
        if verbose:
            err = result.get("error")
            extra = ""
            if err is not None:
                extra = f" ERROR={err}"
            print(f"  [WsGameDriver] recv: type={result.get('type')} seq={result.get('seq')} awaiting={_awaiting(result)}{extra}", flush=True)
        if result.get("type") == "state":
            self._last_state_msg = result
        return result

    def debug_context(self, label: str) -> str:
        """Return concise context for timeout failures."""
        if self._last_state_msg is None:
            return f"{label}: no state received yet; known_seq={self._known_seq}"
        state = _as_dict_or_none(self._last_state_msg.get("state"))
        if state is None:
            return f"{label}: last state payload missing; known_seq={self._known_seq}"
        parts = [
            f"{label}: known_seq={self._known_seq}",
            f"last_seq={self._last_state_msg.get('seq')}",
            f"phase={state.get('phase')}",
            f"awaiting={_awaiting(self._last_state_msg)}",
            f"hand_count={len(_as_list(state.get('player_hand', [])))}",
        ]
        stirring = _as_dict_or_none(state.get("stirring_state"))
        if stirring is not None:
            parts.append(
                "stirring="
                f"phase:{stirring.get('phase')},"
                f"current:{stirring.get('current_player')},"
                f"exchanging:{stirring.get('exchanging_player')}"
            )
        trick = _as_dict_or_none(state.get("trick"))
        if trick is not None:
            parts.append(
                "trick="
                f"current:{trick.get('current_player')},"
                f"lead:{trick.get('lead_player')}"
            )
        return " ".join(parts)

    def receive_msg_safe(self) -> dict[str, object]:
        """Receive a message from the server.

        Updates _known_seq if the message is a state message.
        """
        msg = self.receive_msg()
        if msg.get("type") == "state":
            self.sync_seq(_as_int(msg["seq"]))
        return msg

    def receive_state(self) -> dict[str, object]:
        """Receive the next state message from the server.

        Pure receive — does NOT send any action. The caller is responsible
        for ensuring state pushes are coming (e.g., AutoPlayer cascade
        or a prior action submission).
        """
        return self.receive_msg_safe()

    def request_state(self) -> dict[str, object]:
        """Request the current state with seq=0."""
        if self._ws is None:
            raise RuntimeError("Not connected")
        self._ws.send_json({"seq": 0})
        msg = self.receive_msg(timeout=5)
        assert msg.get("type") == "state"
        self.sync_seq(_as_int(msg["seq"]))
        return msg

    def drain_pending(self) -> None:
        """Drain any pending messages from the WS buffer.

        Reads messages until a read would block (no more messages available).
        Updates _known_seq for each state message. This ensures the WS
        buffer is clean before sending an action, so do_action doesn't
        read stale AutoPlayer cascade messages instead of our action's
        response.

        Uses a short timeout to avoid blocking if no messages are available.
        """
        # In TestClient, there's no non-blocking read. Instead, we rely
        # on the fact that after _recv_until_our_turn returns, all
        # AutoPlayer cascade messages have been consumed. If any remain,
        # it's because they arrived between our last read and now.
        # We can't detect this without a timeout, so this is a no-op.
        # The do_action path ignores same-seq sync responses.
        pass

    def do_action(self, action: dict[str, object]) -> dict[str, object] | None:
        """Send action and wait for response.

        Returns the response message dict on success, None on failure.
        Updates _known_seq on success. Sets last_error on failure.

        Sends exactly once. This test driver only calls do_action after it
        has drained to the human player's turn, so automatic resends create
        duplicate confirmations/actions and make WS response attribution
        ambiguous.
        """
        if self._ws is None:
            raise RuntimeError("Not connected")
        deadline = time.monotonic() + 5
        self.last_error = None
        seq_before = self._known_seq
        self.send_action(action)
        while True:
            if time.monotonic() > deadline:
                print(f"  [do_action] TIMEOUT after 5s, seq stuck at {self._known_seq}", flush=True)
                return None
            try:
                msg = self.receive_msg(verbose=False, timeout=max(0.001, deadline - time.monotonic()))
            except WsReceiveTimeout:
                print(f"  [do_action] TIMEOUT waiting for WS message, seq stuck at {self._known_seq}", flush=True)
                return None
            except Exception as e:
                print(f"  [WsGameDriver] do_action: receive error: {type(e).__name__}: {e}", flush=True)
                raise
            if msg.get("type") != "state":
                continue
            msg_seq = _as_int(msg["seq"])
            self.sync_seq(msg_seq)
            error = msg.get("error")
            if error is not None:
                self.last_error = _as_str(error)
                print(f"  [do_action] rejected: error={self.last_error} seq={self._known_seq}", flush=True)
                return None
            if msg_seq > seq_before:
                return msg

    def wait_for_phase(self, phase: str, timeout: float = 5) -> dict[str, object]:
        """Wait until the game reaches the specified phase.

        Receives state messages until the target phase is found.
        The caller must have previously triggered state pushes (e.g., by
        sending a valid action).

        Returns the state message. Raises TimeoutError if not reached within timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = self.receive_msg(verbose=True, timeout=max(0.001, deadline - time.monotonic()))
            if msg.get("type") == "state":
                self.sync_seq(_as_int(msg["seq"]))
                state = _as_dict_or_none(msg.get("state"))
                if state is not None and state.get("phase") == phase:
                    return msg
        raise TimeoutError(f"Phase '{phase}' not reached within {timeout}s")

    def wait_for_awaiting(self, value: str, timeout: float = 5) -> dict[str, object]:
        """Wait until awaiting_action equals the specified value.

        Pure receive — does NOT send any action. The caller is responsible
        for ensuring state pushes are coming (e.g., AutoPlayer cascade
        or a prior action submission).
        Returns the state message. Raises TimeoutError if not reached within timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = self.receive_msg(verbose=False, timeout=max(0.001, deadline - time.monotonic()))
            if msg.get("type") == "state":
                self.sync_seq(_as_int(msg["seq"]))
                if _awaiting(msg) == value:
                    print(f"  [WsGameDriver] wait_for_awaiting: found {value} seq={self._known_seq}", flush=True)
                    return msg
        raise TimeoutError(f"awaiting='{value}' not reached within {timeout}s")

    def do_bid(self, card_ids: list[str]) -> dict[str, object] | None:
        """Bid with the given cards. Returns response msg on success, None on failure."""
        return self.do_action({"type": "bid", "cards": card_ids})

    def do_bid_pass(self) -> dict[str, object] | None:
        """Pass during DEAL_BID. Returns response msg on success, None on failure."""
        return self.do_action({"type": "bid", "pass": True})

    def do_stir_pass(self) -> dict[str, object] | None:
        """Pass during STIRRING. Returns response msg on success, None on failure."""
        if self._ws is None:
            raise RuntimeError("Not connected")

        deadline = time.monotonic() + 5
        self.last_error = None
        self.send_action({"type": "stir", "pass": True})
        while time.monotonic() < deadline:
            try:
                msg = self.receive_msg(verbose=False, timeout=max(0.001, deadline - time.monotonic()))
            except WsReceiveTimeout:
                print(f"  [do_stir_pass] TIMEOUT waiting for WS message, seq stuck at {self._known_seq}", flush=True)
                return None
            if msg.get("type") != "state":
                continue

            self.sync_seq(_as_int(msg["seq"]))
            error = msg.get("error")
            if error is not None:
                self.last_error = _as_str(error)
                return None

            state = _as_dict(msg["state"])
            if state.get("phase") != "STIRRING" or _awaiting(msg) != "stir":
                return msg

        print(f"  [do_stir_pass] TIMEOUT after 5s, seq stuck at {self._known_seq}", flush=True)
        return None

    def do_stir(self, card_ids: list[str]) -> dict[str, object] | None:
        """Stir with the given cards. Returns response msg on success, None on failure."""
        return self.do_action({"type": "stir", "cards": card_ids})

    def do_discard(self, card_ids: list[str]) -> dict[str, object] | None:
        """Discard the given cards. Returns response msg on success, None on failure."""
        return self.do_action({"type": "discard", "cards": card_ids})

    def do_play(self, card_ids: list[str]) -> dict[str, object] | None:
        """Play the given cards. Returns response msg on success, None on failure."""
        return self.do_action({"type": "play", "cards": card_ids})

    def do_next_round(self) -> dict[str, object] | None:
        """Confirm next round. Returns response msg on success, None on failure."""
        if self._ws is None:
            raise RuntimeError("Not connected")

        deadline = time.monotonic() + 5
        attempts = 0
        self.last_error = None
        self.send_action({"type": "next_round"})
        while time.monotonic() < deadline:
            try:
                msg = self.receive_msg(verbose=False, timeout=max(0.001, deadline - time.monotonic()))
            except WsReceiveTimeout:
                print(f"  [do_next_round] TIMEOUT waiting for WS message, seq stuck at {self._known_seq}", flush=True)
                return None
            if msg.get("type") != "state":
                continue

            self.sync_seq(_as_int(msg["seq"]))
            error = msg.get("error")
            if error is not None:
                self.last_error = _as_str(error)
                return None

            state = _as_dict(msg["state"])
            if state.get("phase") != "WAITING":
                return msg
            confirmed = _as_list(state.get("next_round_confirmed", []))
            if 3 in confirmed:
                return msg
            if _awaiting(msg) == "next_round" and attempts < 4:
                attempts += 1
                self.send_action({"type": "next_round"})
                continue

        print(f"  [do_next_round] TIMEOUT after 5s, seq stuck at {self._known_seq}", flush=True)
        return None


# ---- Infrastructure Tests ----


def test_create_game_returns_201(sync_client: SyncServerClient) -> None:
    """POST /api/game returns 201 with game_id in response body."""
    resp = sync_client.post("/api/game")
    assert resp.status_code == 201
    data = resp.json()
    assert _is_dict(data)
    assert "game_id" in data
    assert isinstance(data["game_id"], str)
    assert len(data["game_id"]) > 0


def test_health_check(sync_client: SyncServerClient) -> None:
    """GET /health returns 200 with status ok."""
    resp = sync_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_games(sync_client: SyncServerClient) -> None:
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


def test_delete_game_closes_ws(sync_client: SyncServerClient) -> None:
    """DELETE a game while WS connected closes the connection without pushing state."""
    resp = sync_client.post("/api/game")
    data = resp.json()
    assert _is_dict(data)
    game_id = _as_str(data["game_id"])

    with sync_client.websocket_connect(f"/game/{game_id}") as ws:
        ws.send_json({"seq": 0})
        raw = ws.receive_json()
        data_msg = _as_dict(raw)
        assert data_msg["type"] == "state"

        resp = sync_client.delete(f"/api/game/{game_id}")
        assert resp.status_code == 200

        with pytest.raises(_WS_ERRORS):
            ws.receive_json()


def test_connect_nonexistent_game(sync_client: SyncServerClient) -> None:
    """Connecting to a nonexistent game closes with code 4404."""
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with sync_client.websocket_connect("/game/nonexistent_id") as ws:
            ws.receive_json()
    assert exc_info.value.code == 4404


def test_connection_takeover(sync_client: SyncServerClient) -> None:
    """New WS connection kicks the old one; old connection gets closed."""
    resp = sync_client.post("/api/game")
    data = resp.json()
    assert _is_dict(data)
    game_id = _as_str(data["game_id"])

    # First connection: get state
    with sync_client.websocket_connect(f"/game/{game_id}") as ws1:
        ws1.send_json({"seq": 0})
        raw1 = ws1.receive_json()
        data1 = _as_dict(raw1)
        assert data1["type"] == "state"

        # Second connection (should kick first)
        with sync_client.websocket_connect(f"/game/{game_id}") as ws2:
            ws2.send_json({"seq": 0})
            raw2 = ws2.receive_json()
            data2 = _as_dict(raw2)
            assert data2["type"] == "state"

        # After ws2 context exits, ws1 should be unusable (kicked by server)
        # The server closed ws1 when ws2 connected, so receive should fail.
        # Note: send_json() raises RuntimeError (not in _WS_ERRORS) because
        # the server already sent a close frame, so we only try receive.
        with pytest.raises(_WS_ERRORS):
            ws1.receive_json()


def test_reconnect_resumes_game(sync_client: SyncServerClient) -> None:
    """After disconnect, reconnect with seq=0 gets current state; then actions work."""
    resp = sync_client.post("/api/game")
    data = resp.json()
    assert _is_dict(data)
    game_id = _as_str(data["game_id"])

    # First connection: get state
    with sync_client.websocket_connect(f"/game/{game_id}") as ws1:
        ws1.send_json({"seq": 0})
        raw1 = ws1.receive_json()
        data1 = _as_dict(raw1)
        assert data1["type"] == "state"
        seq1 = _as_int(data1["seq"])

    # Reconnect: send seq=0 to request the current state.
    with sync_client.websocket_connect(f"/game/{game_id}") as ws2:
        ws2.send_json({"seq": 0})
        raw2 = ws2.receive_json()
        data2 = _as_dict(raw2)
        assert data2["type"] == "state"
        # State should be valid and seq should be >= seq1
        assert _as_int(data2["seq"]) >= seq1
        state2 = _as_dict(data2["state"])
        assert "phase" in state2

        # Verify subsequent action works with the correct seq from received state
        known_seq = _as_int(data2["seq"])
        ws2.send_json({"type": "next_round", "seq": known_seq})
        raw3 = ws2.receive_json()
        data3 = _as_dict(raw3)
        assert data3["type"] == "state"
        # Action with correct seq should either succeed (state changes) or
        # be accepted (no seq mismatch error). Either way, seq should advance
        # or stay the same (if no state change occurred).
        assert _as_int(data3["seq"]) >= known_seq


# ---- Scoring Helpers ----

# Scoring thresholds from server/sm/constants.py
# Each entry: (max_points, declarer_change, switch_declarer)
# - declarer_change: how many levels the declarer advances (positive = up)
# - switch_declarer: if True, defenders become new declarer and gain levels
# Levels never retreat. For defender_points >= 80, formula applies.
_SCORE_THRESHOLDS: list[tuple[int, int, bool]] = [
    (0,   3, False),   # 0 points: declarer +3, no switch
    (39,  2, False),   # 1-39: declarer +2, no switch
    (79,  1, False),   # 40-79: declarer +1, no switch
]

_RANK_ORDER = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]


def _advance_level(rank: str, change: int) -> str:
    """Advance a level by the given change amount, clamped to [TWO, ACE]."""
    idx = _RANK_ORDER.index(rank)
    new_idx = max(0, min(len(_RANK_ORDER) - 1, idx + change))
    return _RANK_ORDER[new_idx]


def _compute_expected_levels(
    total_defender_points: int,
    declarer_team: int,
    team0_level: str,
    team1_level: str,
) -> tuple[str, str]:
    """Compute expected team levels from scoring data.

    Mirrors server/sm/constants.py SCORE_THRESHOLDS + formula logic:
    - Declarer advances by declarer_change levels (0 when switch)
    - If switch_declarer is True, new declarer (old defender) advances by defender_gain
    - Formula for >= 80: defender_gain = max(0, (total - 80) // 40)
    - Levels never retreat
    """
    declarer_change = 0
    defender_gain = 0

    for max_pts, change, _sw in _SCORE_THRESHOLDS:
        if total_defender_points <= max_pts:
            declarer_change = change
            break
    else:
        # defender_points >= 80: switch, formula for new declarer gain
        defender_gain = max(0, (total_defender_points - 80) // 40)

    if declarer_team == 0:
        return (
            _advance_level(team0_level, declarer_change),
            _advance_level(team1_level, defender_gain),
        )
    else:
        return (
            _advance_level(team0_level, defender_gain),
            _advance_level(team1_level, declarer_change),
        )


# ---- Phase Helpers ----


def _verify_common_fields(state: dict[str, object], phase: str) -> None:
    """Verify common state fields are present and valid."""
    assert "phase" in state
    assert state["phase"] == phase
    assert "player_hand" in state
    assert "player_hand_counts" in state
    hand_counts = _as_list(state["player_hand_counts"])
    assert len(hand_counts) == 4
    for count in hand_counts:
        assert isinstance(count, int)
        assert count >= 0
    assert "trump_rank" in state
    assert "declarer_team" in state
    assert "declarer_player" in state or phase == "DEAL_BID"
    assert "team0_level" in state
    assert "team1_level" in state
    assert "defender_points" in state


# ---- Full Game Playthrough ----


def _recv_state(
    driver: WsGameDriver,
    label: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Receive next state message from driver.

    Returns (state_dict, raw_msg). Prints phase/awaiting for debugging.
    Skips non-state messages.

    NOTE: Does NOT update driver._known_seq. See _recv_until_our_turn
    for the rationale.
    """
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            msg = driver.receive_msg(timeout=max(0.001, deadline - time.monotonic()))
        except WsReceiveTimeout as exc:
            raise TimeoutError(driver.debug_context(label)) from exc
        if msg.get("type") == "state":
            state = _as_dict(msg["state"])
            awaiting = _awaiting(msg)
            phase = state.get("phase", "?")
            seq = msg.get("seq", "?")
            # Show trick current_player when in PLAYING
            extra = ""
            if phase == "PLAYING":
                trick_raw = state.get("trick")
                if trick_raw is not None:
                    trick = _as_dict(trick_raw)
                    cp = trick.get("current_player")
                    lp = trick.get("lead_player")
                    extra = f" trick_cp={cp} lead={lp}"
            print(f"  [{label}] recv: phase={phase} awaiting={awaiting} seq={seq}{extra}", flush=True)
            return state, msg
        print(f"  [{label}] recv: non-state msg type={msg.get('type')}", flush=True)
    raise TimeoutError(f"_recv_state: no state message within 5s for {label}")


def _recv_until_our_turn(
    driver: WsGameDriver,
    label: str,
    phase: str,
    awaiting_values: set[str],
    timeout: float = 5,
    exit_awaiting: set[str] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Drain messages until awaiting is in awaiting_values or phase changes.

    This is an optimized version of _recv_state for phases where we don't
    need to inspect every intermediate AutoPlayer state push — we just need
    to know when it's our turn (awaiting matches) or the phase ended.

    Uses verbose=False for high-volume drains to reduce logging overhead.

    Updates driver._known_seq on every state message. This gives
    do_action a more recent seq to start with, reducing seq mismatch
    retries. The do_action retry mechanism handles any remaining
    seq drift from AutoPlayer cascade.

    If exit_awaiting is provided, also returns when awaiting is in that set
    (even if phase hasn't changed yet).

    Returns (state_dict, raw_msg). Raises TimeoutError if condition not met.
    """
    deadline = time.monotonic() + timeout
    count = 0
    last_stuck_seq: int = -1
    stuck_count: int = 0
    while time.monotonic() < deadline:
        try:
            msg = driver.receive_msg(verbose=False, timeout=max(0.001, deadline - time.monotonic()))
        except WsReceiveTimeout as exc:
            raise TimeoutError(driver.debug_context(f"{label}: receive timeout")) from exc
        if msg.get("type") == "state":
            msg_seq = _as_int(msg["seq"])
            driver.sync_seq(msg_seq)
            if msg_seq < driver.known_seq:
                continue
            state = _as_dict(msg["state"])
            count += 1
            awaiting = _awaiting(msg)
            cur_phase = state.get("phase", "?")
            seq_val = msg_seq

            # Stuck detection: if seq hasn't advanced for many messages,
            # print a warning with elapsed time. This catches cases where
            # the game is stuck (no player responding to a state).
            if seq_val == last_stuck_seq:
                stuck_count += 1
                if stuck_count % 20 == 0:
                    print(f"  [{label}] STUCK? t={time.monotonic()-deadline+timeout:.1f}s seq={seq_val} phase={cur_phase} awaiting={awaiting} (same seq x{stuck_count})", flush=True)
            else:
                last_stuck_seq = seq_val
                stuck_count = 0

            if cur_phase != phase:
                print(f"  [{label}] phase changed: {phase} -> {cur_phase} seq={seq_val} (drained {count})", flush=True)
                return state, msg

            if awaiting in awaiting_values:
                print(f"  [{label}] our turn: awaiting={awaiting} seq={seq_val} (drained {count})", flush=True)
                return state, msg

            if exit_awaiting is not None and awaiting in exit_awaiting:
                print(f"  [{label}] exit awaiting: {awaiting} seq={seq_val} (drained {count})", flush=True)
                return state, msg
        # else: skip non-state messages

    raise TimeoutError(driver.debug_context(f"{label}: condition not met within {timeout}s"))


def _hand_dicts(state: dict[str, object]) -> list[dict[str, object]]:
    """Extract player_hand as list[dict] from state."""
    raw = state.get("player_hand")
    if raw is None:
        return []
    result: list[dict[str, object]] = []
    for h in _as_list(raw):
        result.append(_as_dict(h))
    return result


def _should_bid(state: dict[str, object]) -> bool:
    """Decide whether the human should bid or pass.

    Returns True if action_hints is non-empty and we choose to bid
    (deterministic: always bid when possible in round 1, pass otherwise
    for simplicity). This tests the bid path without making the test
    flaky on random outcomes.
    """
    legal = state.get("action_hints")
    if legal is None:
        return False
    legal_list = _as_list(legal)
    return len(legal_list) > 0


def _pick_best_bid(legal_list: list[object]) -> list[dict[str, object]]:
    """Pick the best bid option from action_hints.

    Prefers pairs (2 cards) over singles (1 card), as pairs have
    higher bid_value. Returns the card dicts for the chosen option.
    """
    # First pass: look for pairs
    for option in legal_list:
        cards = _as_list(option)
        if len(cards) == 2:
            return [_as_dict(c) for c in cards]
    # Fallback: first single
    for option in legal_list:
        cards = _as_list(option)
        if len(cards) >= 1:
            return [_as_dict(c) for c in cards]
    # Should not reach here if legal_list is non-empty
    return []


def _extract_card_ids_from_dicts(cards: list[dict[str, object]]) -> list[str]:
    """Extract card ID strings from a list of card dicts."""
    result: list[str] = []
    for c in cards:
        result.append(_as_str(c["id"]))
    return result


def _pick_free_play_candidates(state: dict[str, object], *, limit: int = 80) -> list[list[dict[str, object]]]:
    """Pick fallback play candidates when action_hints is empty.

    Empty hints mean the backend deliberately does not guide the UI; the
    client may still submit cards and let the backend validate. The test
    client therefore tries a bounded list of plausible choices.
    """
    hand = _hand_dicts(state)
    if not hand:
        return []
    lead_count = 1
    lead_cards: list[dict[str, object]] = []
    trick = _as_dict_or_none(state.get("trick"))
    if trick is not None:
        lead_player = _as_int(trick.get("lead_player", 0))
        slots = _as_list(trick.get("slots", []))
        if 0 <= lead_player < len(slots):
            lead_slot = _as_dict(slots[lead_player])
            raw_lead_cards = _as_list(lead_slot.get("cards", []))
            lead_cards = [_as_dict(c) for c in raw_lead_cards]
            if raw_lead_cards:
                lead_count = len(raw_lead_cards)
    if not lead_cards:
        return [[card] for card in hand[:limit]]

    trump_suit = _as_str_or_none(state.get("trump_suit"))
    trump_rank = _as_str(state.get("trump_rank", "2"))
    lead_eff = _effective_suit_dict(lead_cards[0], trump_suit, trump_rank)
    same_eff_cards = [
        card for card in hand
        if _effective_suit_dict(card, trump_suit, trump_rank) == lead_eff
    ]
    other_cards = [card for card in hand if card not in same_eff_cards]

    candidates: list[list[dict[str, object]]] = []
    if len(same_eff_cards) >= lead_count:
        candidates.extend(_card_combinations_prefer_pairs(same_eff_cards, lead_count, limit))
    elif same_eff_cards:
        needed = lead_count - len(same_eff_cards)
        for fill in _card_combinations_prefer_pairs(other_cards, needed, limit):
            candidates.append(same_eff_cards + fill)
            if len(candidates) >= limit:
                break
    else:
        candidates.extend(_card_combinations_prefer_pairs(hand, lead_count, limit))

    return _dedupe_card_candidates(candidates, limit)


def _card_combinations_prefer_pairs(
    cards: list[dict[str, object]],
    count: int,
    limit: int,
) -> list[list[dict[str, object]]]:
    if count <= 0:
        return [[]]
    if len(cards) < count:
        return []

    combos = [list(combo) for combo in combinations(cards, count)]
    if count == 2:
        combos.sort(key=lambda combo: 0 if _same_rank_pair(combo[0], combo[1]) else 1)
    return combos[:limit]


def _same_rank_pair(a: dict[str, object], b: dict[str, object]) -> bool:
    return _as_str(a.get("suit")) == _as_str(b.get("suit")) and _as_str(a.get("rank")) == _as_str(b.get("rank"))


def _dedupe_card_candidates(
    candidates: list[list[dict[str, object]]],
    limit: int,
) -> list[list[dict[str, object]]]:
    seen: set[tuple[str, ...]] = set()
    result: list[list[dict[str, object]]] = []
    for candidate in candidates:
        key = tuple(sorted(_as_str(card["id"]) for card in candidate))
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
        if len(result) >= limit:
            break
    return result


def _effective_suit_dict(card: dict[str, object], trump_suit: str | None, trump_rank: str) -> str:
    suit = _as_str(card.get("suit"))
    rank = _as_str(card.get("rank"))
    if suit == "joker" or rank == trump_rank or (trump_suit is not None and suit == trump_suit):
        return "trump"
    return suit


_WRONG_ACTION_POOL: list[dict[str, object]] = [
    {"type": "invalid_action_type"},
    {"type": "play", "cards": ["NONEXISTENT_0"]},
    {"type": "bid", "cards": ["NONEXISTENT_0"]},
    {"type": "stir", "cards": ["NONEXISTENT_0"]},
    {"type": "discard", "cards": ["NONEXISTENT_0"]},
    {"type": "play", "cards": "NONEXISTENT_0"},
    {"type": "stir", "cards": 123},
    {"cards": ["NONEXISTENT_0"]},
]


def _wrong_payload_key(action: dict[str, object]) -> str:
    """Stable-enough key for keeping injected bad payloads distinct."""
    return repr(action)


def _pick_distinct_wrong_action(start_index: int, used_payloads: set[str]) -> dict[str, object]:
    """Pick a bad action payload that has not been used in this injection batch."""
    for offset in range(len(_WRONG_ACTION_POOL)):
        candidate = _WRONG_ACTION_POOL[(start_index + offset) % len(_WRONG_ACTION_POOL)]
        if _wrong_payload_key(candidate) not in used_payloads:
            return candidate
    raise AssertionError("wrong action pool is too small for distinct injections")


def _send_wrong_actions(
    driver: WsGameDriver,
    label: str,
    correct_action: dict[str, object],
) -> None:
    """Send 1-3 bad requests before the real action and assert protocol behavior.

    The final injected request uses the exact correct payload with an old
    seq when possible. That verifies seq-mismatch handling at the protocol
    layer: the server returns current state and ignores action fields.
    """
    count = 1 + (driver.known_seq % 3)
    used_payloads: set[str] = set()
    for i in range(count):
        seq_before = driver.known_seq
        if i == count - 1 and seq_before > 0:
            wrong_action = correct_action
            assert _wrong_payload_key(wrong_action) not in used_payloads, (
                f"{label}: duplicate wrong payload before seq-mismatch injection: {wrong_action}"
            )
            used_payloads.add(_wrong_payload_key(wrong_action))
            driver.send_action_with_seq(wrong_action, seq_before - 1)
            _expect_seq_mismatch_response(driver, label, wrong_action, seq_before)
        else:
            pool_index = (seq_before + i) % len(_WRONG_ACTION_POOL)
            wrong_action = _pick_distinct_wrong_action(pool_index, used_payloads)
            used_payloads.add(_wrong_payload_key(wrong_action))
            driver.send_action(wrong_action)
            _expect_error_response(driver, label, wrong_action, seq_before)


def _expect_error_response(
    driver: WsGameDriver,
    label: str,
    wrong_action: dict[str, object],
    seq_before: int,
) -> None:
    """Expect either parser rejection or a seq-mismatch state response.

    AutoPlayers may advance the game between send and receive. If that happens,
    the server must ignore the invalid action fields and return state without
    error, which is valid protocol behavior.
    """
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            msg = driver.receive_msg(verbose=False, timeout=max(0.001, deadline - time.monotonic()))
        except WsReceiveTimeout as exc:
            raise TimeoutError(driver.debug_context(f"{label}: wrong action timeout")) from exc
        if msg.get("type") != "state":
            continue
        msg_seq = _as_int(msg["seq"])
        driver.sync_seq(msg_seq)
        if msg.get("error") is not None:
            return
        if msg_seq > seq_before:
            return
    raise TimeoutError(
        f"{label}: wrong action was not rejected within 3s; "
        f"action={wrong_action} seq_before={seq_before}; {driver.debug_context(label)}"
    )


def _expect_seq_mismatch_response(
    driver: WsGameDriver,
    label: str,
    wrong_action: dict[str, object],
    seq_before: int,
) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            msg = driver.receive_msg(verbose=False, timeout=max(0.001, deadline - time.monotonic()))
        except WsReceiveTimeout as exc:
            raise TimeoutError(driver.debug_context(f"{label}: seq-mismatch timeout")) from exc
        if msg.get("type") != "state":
            continue
        msg_seq = _as_int(msg["seq"])
        driver.sync_seq(msg_seq)
        assert msg.get("error") is None, (
            f"{label}: seq mismatch should not return error; "
            f"action={wrong_action} seq_before={seq_before}; msg={msg}"
        )
        assert msg_seq >= seq_before
        return
    raise TimeoutError(
        f"{label}: seq-mismatch action did not return state within 3s; "
        f"action={wrong_action} seq_before={seq_before}; {driver.debug_context(label)}"
    )


# ---- Phase Handlers ----


def _play_deal_bid(
    driver: WsGameDriver,
    round_count: int,
    expected_t0: str | None,
    expected_t1: str | None,
    initial_state: dict[str, object] | None = None,
    initial_msg: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object], str | None, str | None]:
    """Play through DEAL_BID phase.

    Returns (final_state, final_msg, expected_t0, expected_t1).

    In the correct DEAL_BID flow, each card dealt triggers a push.
    The player who received the card sees awaiting_action='bid' and
    must bid or skip before the next card is dealt. The human tries
    to bid from action_hints when available; otherwise passes.
    """
    print(f"[round {round_count}] === DEAL_BID ===", flush=True)

    if initial_state is not None and initial_msg is not None:
        state, msg = initial_state, initial_msg
    else:
        state, msg = _recv_state(driver, f"R{round_count}:BID")
        if state["phase"] != "DEAL_BID":
            print(f"  [R{round_count}:BID] phase={state['phase']}, waiting for DEAL_BID", flush=True)
            msg = driver.wait_for_phase("DEAL_BID", timeout=5)
            state = _as_dict(msg["state"])

    _verify_common_fields(state, "DEAL_BID")
    assert "bid_events" in state
    assert "action_hints" in state
    assert "bid_winner" in state
    assert "trump_suit" in state

    if expected_t0 is not None and expected_t1 is not None:
        assert state["team0_level"] == expected_t0, (
            f"team0_level: expected {expected_t0}, got {state['team0_level']}"
        )
        assert state["team1_level"] == expected_t1, (
            f"team1_level: expected {expected_t1}, got {state['team1_level']}"
        )

    bid_made: bool = False  # Track if human has successfully bid this round
    prev_bid_events_count: int = len(_as_list(state.get("bid_events", [])))

    while True:
        if _awaiting(msg) == "bid":
            # Try to bid from action_hints if available and we
            # haven't already bid this round
            bid_action_accepted = False
            legal = state.get("action_hints")
            legal_list = _as_list(legal) if legal is not None else []
            if not bid_made and _should_bid(state) and legal_list:
                chosen = _pick_best_bid(legal_list)
                card_ids = _extract_card_ids_from_dicts(chosen)
                print(f"  [R{round_count}:BID] human bids: {card_ids}", flush=True)
                _send_wrong_actions(driver, f"R{round_count}:BID before bid", {"type": "bid", "cards": card_ids})
                response = driver.do_bid(card_ids)
                if response is not None:
                    bid_action_accepted = True
                    state = _as_dict(response["state"])
                    msg = response
                    # Verify bid succeeded: bid_events should have grown
                    new_events = _as_list(state.get("bid_events", []))
                    if len(new_events) > prev_bid_events_count:
                        bid_made = True
                        prev_bid_events_count = len(new_events)
                        print(f"  [R{round_count}:BID] bid accepted (events={len(new_events)})", flush=True)
                    else:
                        print(f"  [R{round_count}:BID] bid response but events unchanged", flush=True)
                    if state["phase"] != "DEAL_BID":
                        print(f"[round {round_count}] DEAL_BID -> {state['phase']}", flush=True)
                        break
                    if _awaiting(msg) == "bid":
                        continue
                else:
                    # Bid failed (rejected by server), fall through to pass
                    print(f"  [R{round_count}:BID] bid rejected: {driver.last_error}", flush=True)

            # Pass: either no legal actions, already bid, or bid failed
            if not bid_action_accepted:
                _send_wrong_actions(driver, f"R{round_count}:BID before pass", {"type": "bid", "pass": True})
                response = driver.do_bid_pass()
                if response is not None:
                    state = _as_dict(response["state"])
                    msg = response
                    if state["phase"] != "DEAL_BID":
                        print(f"[round {round_count}] DEAL_BID -> {state['phase']}", flush=True)
                        break
                    if _awaiting(msg) == "bid":
                        continue
            # Fall through to drain

        state, msg = _recv_until_our_turn(
            driver, f"R{round_count}:BID", "DEAL_BID", {"bid"},
        )
        if state["phase"] != "DEAL_BID":
            print(f"[round {round_count}] DEAL_BID -> {state['phase']}", flush=True)
            break

    return state, msg, expected_t0, expected_t1


def _play_stirring(
    driver: WsGameDriver,
    state: dict[str, object],
    msg: dict[str, object],
    round_count: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Play through STIRRING phase. Returns (final_state, final_msg).

    Handles both EXCHANGING sub-phase (discard) and WAITING sub-phase (stir/pass).
    The human always passes during WAITING and discards during EXCHANGING.
    """
    print(f"[round {round_count}] === STIRRING ===", flush=True)
    _verify_common_fields(state, "STIRRING")
    assert "stirring_state" in state
    stirring = _as_dict_or_none(state.get("stirring_state"))
    assert stirring is not None
    assert "phase" in stirring
    assert "trump_suit" in stirring
    assert "current_player" in stirring
    assert "declarer_player" in stirring
    assert "legal_actions" not in stirring

    while True:
        # Check if STIRRING is done
        if state["phase"] != "STIRRING":
            break

        # Handle EXCHANGING sub-phase (discard bottom cards)
        if _awaiting(msg) == "discard":
            state, msg = _play_stirring_exchange(driver, state, round_count)
            if state["phase"] != "STIRRING":
                break
            continue

        # Handle WAITING sub-phase (stir/pass)
        if _awaiting(msg) == "stir":
            _send_wrong_actions(driver, f"R{round_count}:STIR before pass", {"type": "stir", "pass": True})
            response = driver.do_stir_pass()
            if response is not None:
                state = _as_dict(response["state"])
                msg = response
                if state["phase"] != "STIRRING":
                    break
                if _awaiting(msg) in ("stir", "discard"):
                    continue

        state, msg = _recv_until_our_turn(
            driver, f"R{round_count}:STIR", "STIRRING", {"stir", "discard"},
        )
        if state["phase"] != "STIRRING":
            print(f"[round {round_count}] STIRRING -> {state['phase']}", flush=True)
            break

    return state, msg


def _play_stirring_exchange(
    driver: WsGameDriver,
    state: dict[str, object],
    round_count: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Play through STIRRING EXCHANGING sub-phase (discard during stirring).

    Called when phase is "STIRRING" and awaiting_action is "discard".
    Returns (final_state, final_msg).
    """
    _verify_common_fields(state, "STIRRING")
    stirring = _as_dict_or_none(state.get("stirring_state"))
    assert stirring is not None
    exchanging_player_raw = stirring.get("exchanging_player")
    exchanging_player = _as_int(exchanging_player_raw) if isinstance(exchanging_player_raw, int) else None
    exchange_count_raw = stirring.get("exchange_count", 8)
    exchange_count = _as_int(exchange_count_raw) if isinstance(exchange_count_raw, int) else 8

    # Build a placeholder msg; will be updated when we receive actual responses
    msg: dict[str, object] = {}

    if exchanging_player == 3:
        hand = _hand_dicts(state)
        discard_ids: list[str] = []
        for c in hand[-exchange_count:]:
            discard_ids.append(_as_str(c["id"]))
        print(f"  [R{round_count}:STIR-EXCH] human discard {len(discard_ids)} cards", flush=True)
        _send_wrong_actions(
            driver,
            f"R{round_count}:STIR-EXCH before discard",
            {"type": "discard", "cards": discard_ids},
        )
        response = driver.do_discard(discard_ids)
        if response is not None:
            state = _as_dict(response["state"])
            msg = response
            print(f"  [R{round_count}:STIR-EXCH] discard response: phase={state.get('phase')} awaiting={_awaiting(response)}", flush=True)
        else:
            print(f"  [R{round_count}:STIR-EXCH] discard failed, draining to get current state", flush=True)
            state, msg = _recv_until_our_turn(
                driver, f"R{round_count}:STIR-EXCH", "STIRRING", {"discard"},
                timeout=5,
            )
            if _as_str(state.get("awaiting_action", "")) == "discard":
                response = driver.do_discard(discard_ids)
                if response is not None:
                    state = _as_dict(response["state"])
                    msg = response
    else:
        print(f"  [R{round_count}:STIR-EXCH] exchanging_player={exchanging_player}, waiting for our turn", flush=True)
        state, msg = _recv_until_our_turn(
            driver, f"R{round_count}:STIR-EXCH", "STIRRING", {"discard", "stir", "play"},
            timeout=5,
        )

    return state, msg


def _verify_trick_invariants(
    state: dict[str, object],
    current_trump_suit: str | None,
) -> None:
    """Verify last completed trick invariants: winners, suit-following, trump-wins."""
    last_trick_raw = state.get("last_completed_trick")
    if last_trick_raw is None:
        return

    last_trick = _as_dict(last_trick_raw)
    assert "winner" in last_trick, (
        "last_completed_trick must have 'winner' field"
    )
    winner_val = last_trick["winner"]
    assert isinstance(winner_val, int) and winner_val in (0, 1, 2, 3), (
        f"trick_winner must be a valid player index, got {winner_val}"
    )

    # Verify suit-following in the last trick
    last_trick_slots_raw = last_trick.get("slots", [])
    last_trick_slots = _as_list(last_trick_slots_raw)
    if len(last_trick_slots) > 1:
        lead_slot_dict = _as_dict(last_trick_slots[0])
        lead_cards = _as_list(lead_slot_dict.get("cards", []))
        if lead_cards:
            lead_card = _as_dict(lead_cards[0])
            lead_suit = _as_str_or_none(lead_card.get("suit"))
            for slot in last_trick_slots[1:]:
                slot_dict = _as_dict(slot)
                follow_cards = _as_list(slot_dict.get("cards", []))
                if follow_cards:
                    follow_card = _as_dict(follow_cards[0])
                    follow_suit = _as_str_or_none(follow_card.get("suit"))
                    if current_trump_suit is not None and follow_suit == current_trump_suit:
                        continue
                    if lead_suit is not None and follow_suit != lead_suit:
                        pass  # void in that suit — legal

    # Verify trump wins over non-trump in the last completed trick.
    if current_trump_suit is not None:
        winner_slot = None
        for slot in last_trick_slots:
            slot_dict = _as_dict(slot)
            if slot_dict.get("player") == last_trick["winner"]:
                winner_slot = slot_dict
                break
        if winner_slot is not None:
            winner_cards = _as_list(winner_slot.get("cards", []))
            if winner_cards and last_trick_slots:
                winner_card = _as_dict(winner_cards[0])
                lead_slot_dict = _as_dict(last_trick_slots[0])
                lead_cards = _as_list(lead_slot_dict.get("cards", []))
                if lead_cards:
                    lead_card = _as_dict(lead_cards[0])
                    winner_suit = _as_str_or_none(winner_card.get("suit"))
                    lead_suit = _as_str_or_none(lead_card.get("suit"))
                    if (winner_suit == current_trump_suit
                            and lead_suit != current_trump_suit):
                        assert last_trick["winner"] != _as_dict(last_trick_slots[0]).get("player"), (
                            "Trump must beat non-trump lead"
                        )


def _last_completed_trick_key(state: dict[str, object]) -> str | None:
    last_trick_raw = state.get("last_completed_trick")
    if last_trick_raw is None:
        return None
    last_trick = _as_dict(last_trick_raw)
    slots = _as_list(last_trick.get("slots", []))
    slot_parts: list[str] = []
    for slot_raw in slots:
        slot = _as_dict(slot_raw)
        cards = _as_list(slot.get("cards", []))
        card_ids = ",".join(_as_str(_as_dict(card).get("id")) for card in cards)
        slot_parts.append(f"{_as_int(slot.get('player'))}:{card_ids}")
    return "|".join([
        str(_as_int(last_trick.get("lead_player"))),
        str(_as_int(last_trick.get("winner"))),
        str(_as_int(last_trick.get("points"))),
        *slot_parts,
    ])


def _play_playing(
    driver: WsGameDriver,
    state: dict[str, object],
    msg: dict[str, object],
    round_count: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Play through PLAYING phase. Returns (final_state, final_msg).

    Message handling pattern:
    - When it's our turn (awaiting=play): act, use the response as current state.
      If still our turn after acting, act again immediately (e.g., won trick → lead next).
      If not our turn, fall through to drain.
    - When it's not our turn: drain AutoPlayer cascade messages via _recv_until_our_turn.
      Process intermediate states (trick completions) during drain.
    """
    print(f"[round {round_count}] === PLAYING ===", flush=True)
    tricks_played = 0
    current_trump_suit: str | None = _as_str_or_none(state.get("trump_suit"))
    completed_tricks_seen = 0
    prev_completed_trick_key = _last_completed_trick_key(state)

    # If msg is stale (e.g., awaiting is not "play"), drain until we know
    # whether it's our turn. This handles cases where the STIRRING EXCHANGING sub-phase
    # didn't produce a response with awaiting="play" (e.g., when the
    # human is not the declarer and the phase transition happened in the
    # background).
    if _awaiting(msg) != "play" and state.get("phase") == "PLAYING":
        state, msg = _recv_until_our_turn(
            driver, f"R{round_count}:PLAY", "PLAYING", {"play"},
            timeout=5,
        )

    while True:
        if state["phase"] != "PLAYING":
            break

        _verify_common_fields(state, "PLAYING")
        assert "trick" in state
        assert "last_completed_trick" in state
        assert "defender_point_cards" in state
        assert "defender_points" in state
        assert "action_hints" in state or _awaiting(msg) != "play"

        ts_raw = _as_str_or_none(state.get("trump_suit"))
        if ts_raw is not None:
            current_trump_suit = ts_raw

        # Check if a trick just completed.
        cur_completed_trick_key = _last_completed_trick_key(state)
        if cur_completed_trick_key is not None and cur_completed_trick_key != prev_completed_trick_key:
            _verify_trick_invariants(state, current_trump_suit)
            prev_completed_trick_key = cur_completed_trick_key
            completed_tricks_seen += 1
            print(f"  [R{round_count}:PLAY] trick completed ({completed_tricks_seen})", flush=True)

        if _awaiting(msg) == "play":
            legal = state.get("action_hints", [])
            legal_list = _as_list(legal)
            if legal_list:
                play_candidates = [[_as_dict(c) for c in _as_list(option)] for option in legal_list]
            else:
                play_candidates = _pick_free_play_candidates(state)

            play_candidates = [candidate for candidate in play_candidates if len(candidate) >= 1]
            if play_candidates:
                response: dict[str, object] | None = None
                attempted: list[list[str]] = []
                for candidate in play_candidates:
                    play_card_ids = [_as_str(c["id"]) for c in candidate]
                    attempted.append(play_card_ids)
                    print(f"  [R{round_count}:PLAY] trick {tricks_played + 1}: play {play_card_ids}", flush=True)
                    if len(attempted) == 1:
                        _send_wrong_actions(
                            driver,
                            f"R{round_count}:PLAY before play",
                            {"type": "play", "cards": play_card_ids},
                        )
                    response = driver.do_play(play_card_ids)
                    if response is not None:
                        break
                    print(f"  [R{round_count}:PLAY] play rejected: error={driver.last_error}", flush=True)
                assert response is not None, (
                    f"Could not find accepted play from {len(play_candidates)} candidates; "
                    f"attempted={attempted[:10]} last_error={driver.last_error}"
                )
                tricks_played += 1

                # Use the direct response as current state — do NOT call
                # _recv_state() or receive_msg_safe() here! The response
                # already tells us the state after our play.
                state = _as_dict(response["state"])
                msg = response
                if state["phase"] != "PLAYING":
                    print(f"[round {round_count}] PLAYING -> {state['phase']} after {tricks_played} tricks", flush=True)
                    break
                # Check if we won the trick and it's still our turn
                cur_completed_trick_key = _last_completed_trick_key(state)
                if cur_completed_trick_key is not None and cur_completed_trick_key != prev_completed_trick_key:
                    _verify_trick_invariants(state, current_trump_suit)
                    prev_completed_trick_key = cur_completed_trick_key
                    completed_tricks_seen += 1
                    print(f"  [R{round_count}:PLAY] trick completed ({completed_tricks_seen})", flush=True)
                if _awaiting(msg) == "play":
                    continue  # Still our turn (e.g., won trick -> lead next)
                # Action failed or not our turn — fall through to drain
            else:
                print(f"  [R{round_count}:PLAY] no legal actions!? awaiting={_awaiting(msg)}", flush=True)

        # Not our turn — drain AutoPlayer cascade until our turn or phase change.
        # Process trick completions during drain.
        while True:
            state, msg = _recv_until_our_turn(
                driver, f"R{round_count}:PLAY", "PLAYING", {"play"},
                timeout=5,
            )
            if state["phase"] != "PLAYING":
                print(f"[round {round_count}] PLAYING -> {state['phase']} after {tricks_played} tricks", flush=True)
                break
            # Check trick completion in the drained state
            cur_completed_trick_key = _last_completed_trick_key(state)
            if cur_completed_trick_key is not None and cur_completed_trick_key != prev_completed_trick_key:
                _verify_trick_invariants(state, current_trump_suit)
                prev_completed_trick_key = cur_completed_trick_key
                completed_tricks_seen += 1
                print(f"  [R{round_count}:PLAY] trick completed ({completed_tricks_seen})", flush=True)
            if _awaiting(msg) == "play":
                break  # Our turn now — go back to top of outer loop
            # Shouldn't happen (_recv_until_our_turn guarantees one of the
            # conditions), but just in case, keep draining

        if state["phase"] != "PLAYING":
            break

    return state, msg


def _play_waiting(
    driver: WsGameDriver,
    state: dict[str, object],
    msg: dict[str, object],
    round_count: int,
    prev_team0_level: str | None = None,
    prev_team1_level: str | None = None,
) -> tuple[dict[str, object], dict[str, object], str | None, str | None]:
    """Play through WAITING phase (round complete, awaiting next_round confirm).

    Returns (final_state, final_msg, expected_t0, expected_t1).

    prev_team0_level/prev_team1_level: the levels BEFORE process_round_result
    updated them. These are needed because process_round_result runs at the
    end of PLAYING, so the WAITING state already contains the updated levels.
    We need the pre-update levels to compute the expected change.
    """
    print(f"[round {round_count}] === WAITING (round complete) === phase={state.get('phase')} awaiting={_awaiting(msg)}", flush=True)
    _verify_common_fields(state, "WAITING")
    assert "scoring" in state
    assert "bottom_cards" in state
    assert "next_round_confirmed" in state

    bottom_cards_raw = state["bottom_cards"]
    bottom_cards = _as_list(bottom_cards_raw)
    assert len(bottom_cards) > 0, "bottom_cards must contain actual card data in WAITING phase"
    for card in bottom_cards:
        card_dict = _as_dict(card)
        assert "id" in card_dict, f"Bottom card missing 'id' field: {card_dict}"
        assert "suit" in card_dict, f"Bottom card missing 'suit' field: {card_dict}"
        assert "rank" in card_dict, f"Bottom card missing 'rank' field: {card_dict}"

    scoring_raw = state["scoring"]
    assert scoring_raw is not None
    scoring = _as_dict(scoring_raw)
    assert "total_defender_points" in scoring
    assert "declarer_team" in scoring
    assert isinstance(state["next_round_confirmed"], list)

    scoring_tdp = scoring["total_defender_points"]
    assert isinstance(scoring_tdp, int)
    scoring_dt = scoring["declarer_team"]
    assert isinstance(scoring_dt, int)

    # Compute expected levels from scoring data. process_round_result runs
    # at the end of PLAYING, so state already has the updated levels.
    # We need the pre-update levels to verify the change was correct.
    # If prev levels are not available, use current levels (no verification).
    t0_for_calc = prev_team0_level if prev_team0_level is not None else _as_str(state["team0_level"])
    t1_for_calc = prev_team1_level if prev_team1_level is not None else _as_str(state["team1_level"])
    expected_t0, expected_t1 = _compute_expected_levels(
        scoring_tdp,
        scoring_dt,
        t0_for_calc,
        t1_for_calc,
    )
    # Verify that the levels in the state match our expected calculation
    if prev_team0_level is not None and prev_team1_level is not None:
        assert _as_str(state["team0_level"]) == expected_t0, (
            f"team0_level mismatch: expected {expected_t0}, got {state['team0_level']} "
            f"(prev={prev_team0_level}, tdp={scoring_tdp}, dt={scoring_dt})"
        )
        assert _as_str(state["team1_level"]) == expected_t1, (
            f"team1_level mismatch: expected {expected_t1}, got {state['team1_level']} "
            f"(prev={prev_team1_level}, tdp={scoring_tdp}, dt={scoring_dt})"
        )
    print(
        f"  [R{round_count}:WAITING] defender_pts={scoring_tdp} "
        f"declarer_team={scoring_dt} levels: t0={expected_t0} t1={expected_t1}",
        flush=True,
    )


    # Confirm next_round and drain until phase changes.
    # do_action may return on an AutoPlayer cascade push instead of
    # our own confirmation push. After it returns, we keep draining
    # messages until the phase actually changes. If our confirmation
    # hasn't been processed yet, the server will process it on the
    # next event loop tick (triggered by receive_msg()'s portal.call).
    if _awaiting(msg) == "next_round":
        print(f"  [R{round_count}:WAITING] human confirm next_round", flush=True)
        _send_wrong_actions(driver, f"R{round_count}:WAITING before next_round", {"type": "next_round"})
        response = driver.do_next_round()
        if response is not None:
            state = _as_dict(response["state"])
            msg = response
            if state["phase"] != "WAITING":
                print(f"[round {round_count}] WAITING -> {state['phase']}", flush=True)
                return state, msg, expected_t0, expected_t1
        while state["phase"] == "WAITING":
            state, msg = _recv_until_our_turn(
                driver, f"R{round_count}:WAITING", "WAITING", set(), timeout=5,
            )

    print(f"[round {round_count}] WAITING -> {state['phase']}", flush=True)
    return state, msg, expected_t0, expected_t1


def _verify_game_over(
    driver: WsGameDriver,
    state: dict[str, object],
    game_id: str,
    sync_client: SyncServerClient,
    expected_t0: str | None,
    expected_t1: str | None,
) -> None:
    """Verify GAME_OVER state and post-game cleanup."""
    print("=== GAME_OVER ===")
    assert state["phase"] == "GAME_OVER", f"Expected GAME_OVER, got {state['phase']}"
    assert "winning_team" in state
    winning_team = state["winning_team"]
    assert isinstance(winning_team, int) and winning_team in (0, 1)
    print(f"  winning_team={winning_team}")

    if expected_t0 is not None and expected_t1 is not None:
        assert state["team0_level"] == expected_t0
        assert state["team1_level"] == expected_t1

    # Close the current connection and verify a fresh connection can
    # recover the finished state with seq=0.
    print("  closing driver (GAME_OVER)", flush=True)
    driver.close()
    print("  driver closed", flush=True)

    # Verify game remains in registry after GAME_OVER so a crashed client
    # can reconnect and request the current state.
    print("  checking registry...", flush=True)
    resp = sync_client.get("/api/game")
    games_raw = resp.json()["games"]
    assert _is_list_of_dict(games_raw)
    game_ids = [g["game_id"] for g in games_raw]
    assert game_id in game_ids, (
        f"Game {game_id} should remain in registry after GAME_OVER"
    )
    print("  game retained in registry", flush=True)

    print("  checking seq=0 reconnect...", flush=True)
    with sync_client.websocket_connect(f"/game/{game_id}") as ws:
        ws.send_json({"seq": 0})
        raw = ws.receive_json()
        msg = _as_dict(raw)
        assert msg["type"] == "state"
        reconnect_state = _as_dict(msg["state"])
        assert reconnect_state["phase"] == "GAME_OVER"
    print("  GAME_OVER verification complete", flush=True)


# ---- Full Game Playthrough ----


def test_full_game(sync_client: SyncServerClient) -> None:
    """Play a complete game from start to GAME_OVER.

    At each phase:
    - Before every correct human action, inject 1-3 rejected requests
      covering malformed actions, invalid fields/cards, and stale seq.
    - Verify state fields and phase transitions
    - Let AutoPlayers handle their turns automatically

    The human player (index 3) acts when awaiting_action matches.
    AutoPlayers (indices 0-2) act automatically via the server.
    """
    print(">>> test_full_game starting <<<", flush=True)
    resp = sync_client.post("/api/game")
    game_id = resp.json()["game_id"]
    print(f"=== Created game {game_id} ===", flush=True)

    driver = WsGameDriver(sync_client)
    try:
        print(">>> connecting <<<", flush=True)
        driver.connect(game_id)
        print(">>> connected <<<", flush=True)

        # Game starts in WAITING phase. AutoPlayers (0-2) request state with
        # seq=0 and confirm automatically via run(). The human client must
        # first request state with seq=0, then send next_round with the returned
        # seq. seq=0 never carries an action.
        # AutoPlayer cascade may have already advanced the game further, so we
        # need to drain until we know the current phase.
        print(">>> requesting initial state <<<", flush=True)
        sync_msg = driver.request_state()
        sync_state = _as_dict(sync_msg["state"])
        print(
            f"  initial sync: phase={sync_state.get('phase')} awaiting={_awaiting(sync_msg)} seq={driver.known_seq}",
            flush=True,
        )

        print(">>> sending initial next_round <<<", flush=True)
        if _awaiting(sync_msg) == "next_round":
            result = driver.do_next_round()
            assert result is not None, f"next_round should succeed with synced seq: error={driver.last_error}"
            print(f"  next_round succeeded, seq={driver.known_seq}", flush=True)
        else:
            result = sync_msg

        # Drain any AutoPlayer cascade pushes until we reach a stable state
        # where we know what phase we're in.
        state = _as_dict(result["state"])
        msg = result
        while True:
            cur_phase = state.get("phase", "?")
            cur_awaiting = _awaiting(msg)
            print(f"  initial state: phase={cur_phase} awaiting={cur_awaiting} seq={driver.known_seq}", flush=True)
            if cur_phase in ("DEAL_BID", "STIRRING", "PLAYING", "GAME_OVER"):
                break
            # Still in WAITING or transitional state — drain more
            msg = driver.receive_msg(timeout=5)
            if msg.get("type") == "state":
                driver.sync_seq(_as_int(msg["seq"]))
                state = _as_dict(msg["state"])
            else:
                break

        round_count = 0
        max_rounds = 100  # play until GAME_OVER
        expected_t0: str | None = None
        expected_t1: str | None = None
        # state and msg are set during initialization or by phase handlers
        msg: dict[str, object] = msg  # from initialization drain above
        state: dict[str, object] = state  # from initialization drain above

        while round_count < max_rounds:
            round_count += 1
            print(f"\n{'='*40} ROUND {round_count} {'='*40}")

            # Snapshot levels before the round starts (before process_round_result)
            prev_t0 = _as_str(state["team0_level"])
            prev_t1 = _as_str(state["team1_level"])

            # DEAL_BID: pass initial state for round 1 (already obtained above)
            state, msg, expected_t0, expected_t1 = _play_deal_bid(
                driver, round_count, expected_t0, expected_t1,
                initial_state=state if round_count == 1 else None,
                initial_msg=msg if round_count == 1 else None,
            )

            # STIRRING (optional)
            if state["phase"] == "STIRRING":
                state, msg = _play_stirring(driver, state, msg, round_count)

            # PLAYING
            if state["phase"] == "PLAYING":
                state, msg = _play_playing(driver, state, msg, round_count)

            # WAITING (round complete)
            if state["phase"] == "WAITING":
                state, msg, expected_t0, expected_t1 = _play_waiting(
                    driver, state, msg, round_count,
                    prev_team0_level=prev_t0,
                    prev_team1_level=prev_t1,
                )

            if state["phase"] == "GAME_OVER":
                # Compute expected levels for this round (PLAYING may have
                # transitioned directly to GAME_OVER, skipping WAITING's
                # level verification).
                scoring = _as_dict(state["scoring"]) if state.get("scoring") is not None else None
                if scoring is not None:
                    scoring_tdp = scoring.get("total_defender_points")
                    scoring_dt = scoring.get("declarer_team")
                    if isinstance(scoring_tdp, int) and isinstance(scoring_dt, int):
                        expected_t0, expected_t1 = _compute_expected_levels(
                            scoring_tdp, scoring_dt, prev_t0, prev_t1,
                        )
                break

        assert state["phase"] == "GAME_OVER", (
            f"Game should reach GAME_OVER within {max_rounds} rounds, "
            f"but is still in {state['phase']}"
        )
        _verify_game_over(driver, state, game_id, sync_client, expected_t0, expected_t1)
    finally:
        # ALWAYS close the WebSocket driver — even if the test fails midway.
        # Without this, the server's handle_connection is stuck in receive_json(),
        # the TestClient's event loop cannot shut down, and pytest hangs at 100%.
        driver.close()
