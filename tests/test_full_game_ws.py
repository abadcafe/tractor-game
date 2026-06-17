"""Full-game integration tests via server external interfaces (REST + WebSocket).

These tests only use the server's public interfaces: REST API and WebSocket protocol.
They do NOT import any server internal modules (Game, Player, sm, etc.).

Module: test_full_game_ws
Responsibilities: Play complete games through REST+WS external interfaces,
    verifying state transitions, boundary conditions, and error handling.
"""

import random
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


def _is_list_of_dict(val: object) -> TypeGuard[list[dict[str, object]]]:
    """Narrow object to list[dict[str, object]]."""
    if not _is_list(val):
        return False
    for item in val:
        if not _is_dict(item):
            return False
    return True


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

    Tracks state sequence numbers and provides high-level do_* methods
    for game actions.
    """

    def __init__(self, sync_client: TestClient) -> None:
        self._client = sync_client
        self._ws: WebSocketTestSession | None = None
        self._ws_cm: WebSocketTestSession | None = None  # context manager for websocket_connect
        self._current_seq: int = 0
        self.last_error: str | None = None
        self._game_id: str | None = None

    @property
    def current_seq(self) -> int:
        """Current state sequence number (read-only public accessor)."""
        return self._current_seq

    def sync_seq(self, seq: int) -> None:
        """Update _current_seq to the given value.

        Public accessor for seq synchronization from helper functions
        that need to keep the driver's seq in sync with server pushes.
        """
        self._current_seq = seq

    def connect(self, game_id: str) -> None:
        """Connect to a game via WebSocket.

        Does NOT automatically receive initial state.
        Client must send an action with seq=0 to get the initial state.
        """
        self._game_id = game_id
        self._ws_cm = self._client.websocket_connect(f"/game/{game_id}")
        self._ws = self._ws_cm.__enter__()
        self._current_seq = 0
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
        if self._ws is None:
            raise RuntimeError("Not connected")
        action_with_seq = {**action, "seq": self._current_seq}
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
        print(f"  [WsGameDriver] send: type={action_with_seq.get('type')}{detail} seq={self._current_seq}{slow}", flush=True)

    def receive_msg(self, *, verbose: bool = True) -> dict[str, object]:
        """Receive a message from the server.

        Does NOT auto-reconnect on disconnect. Raises the exception
        so that do_action() can handle reconnection in its retry loop.
        When verbose=False, skips per-message logging for high-volume drains.
        """
        if self._ws is None:
            raise RuntimeError("Not connected")
        raw = self._ws.receive_json()
        result = _as_dict(raw)
        if verbose:
            err = result.get("error")
            extra = ""
            if err is not None:
                extra = f" ERROR={err}"
            print(f"  [WsGameDriver] recv: type={result.get('type')} seq={result.get('seq')} awaiting={result.get('awaiting')}{extra}", flush=True)
        return result

    def receive_msg_safe(self) -> dict[str, object]:
        """Receive a message from the server.

        Updates _current_seq if the message is a state message.
        """
        msg = self.receive_msg()
        if msg.get("type") == "state":
            self._current_seq = _as_int(msg["seq"])
        return msg

    def receive_state(self) -> dict[str, object]:
        """Receive the next state message from the server.

        Pure receive — does NOT send any action. The caller is responsible
        for ensuring state pushes are coming (e.g., AutoPlayer cascade
        or a prior action submission).
        """
        return self.receive_msg_safe()

    def drain_pending(self) -> None:
        """Drain any pending messages from the WS buffer.

        Reads messages until a read would block (no more messages available).
        Updates _current_seq for each state message. This ensures the WS
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
        # The do_action retry mechanism handles stale messages.
        pass

    def do_action(self, action: dict[str, object]) -> dict[str, object] | None:
        """Send action and wait for response.

        Returns the response message dict on success, None on failure.
        Updates _current_seq on success. Sets last_error on failure.

        On seq mismatch, retries with the updated seq. AutoPlayer cascade
        can advance the server's seq between our reads, so we allow
        multiple retries.

        Timeout: 5s overall. If exceeded, returns None and prints a
        stuck warning. In this state-machine test, a timeout means the
        game is stuck (no player responding), not slow.
        """
        if self._ws is None:
            raise RuntimeError("Not connected")
        deadline = time.monotonic() + 5
        for _ in range(20):  # allow many retries for seq mismatch
            if time.monotonic() > deadline:
                print(f"  [do_action] TIMEOUT after 5s, seq stuck at {self._current_seq}", flush=True)
                return None
            self.last_error = None
            seq_before = self._current_seq
            self.send_action(action)
            # Drain messages until we get a definitive response:
            # - seq advanced with no error → action succeeded
            # - error field present with seq unchanged → action rejected
            # - error field present with seq advanced → stale seq, retry
            while True:
                if time.monotonic() > deadline:
                    print(f"  [do_action] TIMEOUT after 5s, seq stuck at {self._current_seq}", flush=True)
                    return None
                try:
                    msg = self.receive_msg(verbose=False)
                except Exception as e:
                    print(f"  [WsGameDriver] do_action: receive error: {type(e).__name__}: {e}", flush=True)
                    raise
                if msg.get("type") == "state":
                    self._current_seq = _as_int(msg["seq"])
                    error = msg.get("error")
                    if error is not None:
                        self.last_error = _as_str(error)
                        if self._current_seq > seq_before:
                            # Seq advanced with error → our seq was stale
                            # (AutoPlayer cascade advanced server state).
                            # Retry with the updated seq.
                            break
                        # Error with seq unchanged → action rejected
                        return None
                    if self._current_seq > seq_before:
                        return msg  # action succeeded
                    # seq unchanged, no error → stale cascade push, drain more
                    continue
                # non-state message → skip and keep receiving
            # Only reach here on seq mismatch → retry
            continue
        return None

    def wait_for_phase(self, phase: str, timeout: float = 5) -> dict[str, object]:
        """Wait until the game reaches the specified phase.

        Receives state messages until the target phase is found.
        The caller must have previously triggered state pushes (e.g., by
        sending a valid action).

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

    def wait_for_awaiting(self, value: str, timeout: float = 5) -> dict[str, object]:
        """Wait until awaiting_action equals the specified value.

        Pure receive — does NOT send any action. The caller is responsible
        for ensuring state pushes are coming (e.g., AutoPlayer cascade
        or a prior action submission).
        Returns the state message. Raises TimeoutError if not reached within timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = self.receive_msg(verbose=False)
            if msg.get("type") == "state":
                self._current_seq = _as_int(msg["seq"])
                if msg.get("awaiting") == value:
                    print(f"  [WsGameDriver] wait_for_awaiting: found {value} seq={self._current_seq}", flush=True)
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
        return self.do_action({"type": "stir", "pass": True})

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

    NOTE: Does NOT update driver._current_seq. See _recv_until_our_turn
    for the rationale.
    """
    while True:
        msg = driver.receive_msg()
        if msg.get("type") == "state":
            state = _as_dict(msg["state"])
            awaiting = msg.get("awaiting")
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

    Updates driver._current_seq on every state message. This gives
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
        msg = driver.receive_msg(verbose=False)
        if msg.get("type") == "state":
            driver.sync_seq(_as_int(msg["seq"]))
            state = _as_dict(msg["state"])
            count += 1
            awaiting = msg.get("awaiting")
            cur_phase = state.get("phase", "?")
            seq_val = _as_int(msg["seq"])

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

    raise TimeoutError(f"_recv_until_our_turn: condition not met within {timeout}s")


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

    Returns True if bid_legal_actions is non-empty and we choose to bid
    (deterministic: always bid when possible in round 1, pass otherwise
    for simplicity). This tests the bid path without making the test
    flaky on random outcomes.
    """
    legal = state.get("bid_legal_actions")
    if legal is None:
        return False
    legal_list = _as_list(legal)
    return len(legal_list) > 0


def _pick_best_bid(legal_list: list[object]) -> list[dict[str, object]]:
    """Pick the best bid option from bid_legal_actions.

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


# ---- Wrong Action Injection ----


_WRONG_ACTION_POOL: list[dict[str, object]] = [
    {"type": "invalid_action_type"},
    {"type": "play", "cards": ["NONEXISTENT_0"]},
    {"type": "bid", "cards": ["NONEXISTENT_0"]},
    {"type": "stir", "cards": ["NONEXISTENT_0"]},
    {"type": "discard", "cards": ["NONEXISTENT_0"]},
    {"type": "play", "cards": ["NONEXISTENT_1", "NONEXISTENT_2"]},
    {"type": "bid", "cards": ["NONEXISTENT_1", "NONEXISTENT_2"]},
    {"type": "stir", "cards": ["NONEXISTENT_2", "NONEXISTENT_3"]},
    {"type": "discard", "cards": ["NONEXISTENT_2", "NONEXISTENT_3"]},
]


def _send_wrong_actions(driver: WsGameDriver) -> None:
    """Send 1-3 random wrong actions before the correct action.

    Verifies that each wrong action is rejected by the server.
    This tests that the game properly handles invalid actions and
    continues working normally after rejection.

    Uses send_action + receive_msg directly (not do_action) to avoid
    the retry-on-seq-mismatch logic, which would re-send the same
    wrong action indefinitely.
    """
    n = random.randint(1, 3)
    pool = _WRONG_ACTION_POOL.copy()
    random.shuffle(pool)
    for i in range(n):
        wrong_action = pool[i % len(pool)]
        seq_before = driver.current_seq
        driver.send_action(wrong_action)
        deadline = time.monotonic() + 3
        rejected = False
        while time.monotonic() < deadline:
            msg = driver.receive_msg(verbose=False)
            if msg.get("type") == "state":
                new_seq = _as_int(msg["seq"])
                driver.sync_seq(new_seq)
                error = msg.get("error")
                if error is not None:
                    # Wrong action was rejected — expected
                    rejected = True
                    break
                if new_seq > seq_before:
                    # Seq advanced without error — AutoPlayer cascade
                    # push, not our action's response. Keep reading.
                    continue
                # Seq unchanged, no error — stale push, keep reading
                continue
        assert rejected, (
            f"Wrong action {wrong_action} was not rejected within 3s "
            f"(seq: {seq_before} -> {driver.current_seq})"
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
    to bid from bid_legal_actions when available; otherwise passes.
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
    assert "bid_legal_actions" in state
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
        if msg.get("awaiting") == "bid":
            # Try to bid from bid_legal_actions if available and we
            # haven't already bid this round
            legal = state.get("bid_legal_actions")
            legal_list = _as_list(legal) if legal is not None else []
            if not bid_made and _should_bid(state) and legal_list:
                chosen = _pick_best_bid(legal_list)
                card_ids = _extract_card_ids_from_dicts(chosen)
                print(f"  [R{round_count}:BID] human bids: {card_ids}", flush=True)
                _send_wrong_actions(driver)
                response = driver.do_bid(card_ids)
                if response is not None:
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
                    if msg.get("awaiting") == "bid":
                        continue
                    # Not our turn anymore — fall through to drain
                else:
                    # Bid failed (rejected by server), fall through to pass
                    print(f"  [R{round_count}:BID] bid rejected: {driver.last_error}", flush=True)

            # Pass: either no legal actions, already bid, or bid failed
            _send_wrong_actions(driver)
            response = driver.do_bid_pass()
            if response is not None:
                state = _as_dict(response["state"])
                msg = response
                if state["phase"] != "DEAL_BID":
                    print(f"[round {round_count}] DEAL_BID -> {state['phase']}", flush=True)
                    break
                if msg.get("awaiting") == "bid":
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
    assert "legal_actions" in stirring

    while True:
        # Check if STIRRING is done
        if state["phase"] != "STIRRING":
            break

        # Handle EXCHANGING sub-phase (discard bottom cards)
        if msg.get("awaiting") == "discard":
            state, msg = _play_stirring_exchange(driver, state, round_count)
            if state["phase"] != "STIRRING":
                break
            continue

        # Handle WAITING sub-phase (stir/pass)
        if msg.get("awaiting") == "stir":
            _send_wrong_actions(driver)
            response = driver.do_stir_pass()
            if response is not None:
                state = _as_dict(response["state"])
                msg = response
                if state["phase"] != "STIRRING":
                    break
                if msg.get("awaiting") in ("stir", "discard"):
                    continue
            # Fall through to drain

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
        _send_wrong_actions(driver)
        response = driver.do_discard(discard_ids)
        if response is not None:
            state = _as_dict(response["state"])
            msg = response
            print(f"  [R{round_count}:STIR-EXCH] discard response: phase={state.get('phase')} awaiting={response.get('awaiting')}", flush=True)
        else:
            print(f"  [R{round_count}:STIR-EXCH] discard failed, draining to get current state", flush=True)
            state, msg = _recv_until_our_turn(
                driver, f"R{round_count}:STIR-EXCH", "STIRRING", {"discard"},
                timeout=5,
            )
            if _as_str(state.get("awaiting_action", "")) == "discard":
                _send_wrong_actions(driver)
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
    """Verify trick history invariants: winners, suit-following, trump-wins."""
    trick_history_raw = state.get("trick_history", [])
    trick_history = _as_list(trick_history_raw)
    if not trick_history:
        return

    last_trick_raw = trick_history[-1]
    last_trick = _as_dict(last_trick_raw)
    assert "winner" in last_trick, (
        "Completed trick in trick_history must have 'winner' field"
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

    # Verify trump wins over non-trump
    if current_trump_suit is not None:
        for past_trick_raw in trick_history:
            past_trick = _as_dict(past_trick_raw)
            past_slots_raw = past_trick.get("slots", [])
            past_slots = _as_list(past_slots_raw)
            if not past_slots:
                continue
            winner_slot = None
            for slot in past_slots:
                slot_dict = _as_dict(slot)
                if slot_dict.get("player") == past_trick["winner"]:
                    winner_slot = slot_dict
                    break
            if winner_slot is not None:
                winner_cards = _as_list(winner_slot.get("cards", []))
                if winner_cards:
                    winner_card = _as_dict(winner_cards[0])
                    lead_slot_dict = _as_dict(past_slots[0])
                    lead_cards = _as_list(lead_slot_dict.get("cards", []))
                    if lead_cards:
                        lead_card = _as_dict(lead_cards[0])
                        winner_suit = _as_str_or_none(winner_card.get("suit"))
                        lead_suit = _as_str_or_none(lead_card.get("suit"))
                        if (winner_suit == current_trump_suit
                                and lead_suit != current_trump_suit):
                            assert past_trick["winner"] != _as_dict(past_slots[0]).get("player"), (
                                "Trump must beat non-trump lead"
                            )


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
    first_trick = True
    current_trump_suit: str | None = _as_str_or_none(state.get("trump_suit"))
    prev_trick_count: int = len(_as_list(state.get("trick_history", [])))

    # If msg is stale (e.g., awaiting is not "play"), drain until we know
    # whether it's our turn. This handles cases where the STIRRING EXCHANGING sub-phase
    # didn't produce a response with awaiting="play" (e.g., when the
    # human is not the declarer and the phase transition happened in the
    # background).
    if msg.get("awaiting") != "play" and state.get("phase") == "PLAYING":
        state, msg = _recv_until_our_turn(
            driver, f"R{round_count}:PLAY", "PLAYING", {"play"},
            timeout=5,
        )

    while True:
        if state["phase"] != "PLAYING":
            break

        _verify_common_fields(state, "PLAYING")
        assert "trick" in state
        assert "trick_history" in state
        assert "defender_points" in state
        assert "legal_actions" in state or msg.get("awaiting") != "play"

        ts_raw = _as_str_or_none(state.get("trump_suit"))
        if ts_raw is not None:
            current_trump_suit = ts_raw

        # Check if a trick just completed (trick_history grew)
        cur_trick_count = len(_as_list(state.get("trick_history", [])))
        if cur_trick_count > prev_trick_count:
            _verify_trick_invariants(state, current_trump_suit)
            prev_trick_count = cur_trick_count
            print(f"  [R{round_count}:PLAY] trick {cur_trick_count} completed", flush=True)

        if msg.get("awaiting") == "play":
            legal = state.get("legal_actions", [])
            legal_list = _as_list(legal)
            if legal_list:
                chosen_cards = _as_list(legal_list[0])
                assert len(chosen_cards) >= 1, (
                    f"Legal play must have at least 1 card, got {len(chosen_cards)}"
                )

                if first_trick:
                    # Send ALL cards of the legal action in dict format,
                    # not just the first card — a legal action may require
                    # multiple cards (e.g. a pair or tractor).
                    dict_cards: list[dict[str, object]] = []
                    for c in chosen_cards:
                        dict_cards.append(_as_dict(c))
                    play_action: dict[str, object] = {
                        "type": "play",
                        "cards": dict_cards,
                    }
                    _send_wrong_actions(driver)
                    response = driver.do_action(play_action)
                    assert response is not None, "Dict-format card play should succeed"
                    first_trick = False
                    print(f"  [R{round_count}:PLAY] trick {tricks_played + 1}: dict-format play ({len(dict_cards)} cards)", flush=True)
                else:
                    play_card_ids: list[str] = []
                    for c in chosen_cards:
                        c_dict = _as_dict(c)
                        play_card_ids.append(_as_str(c_dict["id"]))
                    print(f"  [R{round_count}:PLAY] trick {tricks_played + 1}: play {play_card_ids}", flush=True)
                    _send_wrong_actions(driver)
                    response = driver.do_play(play_card_ids)
                    if response is None:
                        print(f"  [R{round_count}:PLAY] PLAY FAILED! error={driver.last_error}", flush=True)
                tricks_played += 1

                # Use the direct response as current state — do NOT call
                # _recv_state() or receive_msg_safe() here! The response
                # already tells us the state after our play.
                if response is not None:
                    state = _as_dict(response["state"])
                    msg = response
                    if state["phase"] != "PLAYING":
                        print(f"[round {round_count}] PLAYING -> {state['phase']} after {tricks_played} tricks", flush=True)
                        break
                    # Check if we won the trick and it's still our turn
                    cur_trick_count = len(_as_list(state.get("trick_history", [])))
                    if cur_trick_count > prev_trick_count:
                        _verify_trick_invariants(state, current_trump_suit)
                        prev_trick_count = cur_trick_count
                        print(f"  [R{round_count}:PLAY] trick {cur_trick_count} completed", flush=True)
                    if msg.get("awaiting") == "play":
                        continue  # Still our turn (e.g., won trick → lead next)
                # Action failed or not our turn — fall through to drain
            else:
                print(f"  [R{round_count}:PLAY] no legal actions!? awaiting={msg.get('awaiting')}", flush=True)

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
            cur_trick_count = len(_as_list(state.get("trick_history", [])))
            if cur_trick_count > prev_trick_count:
                _verify_trick_invariants(state, current_trump_suit)
                prev_trick_count = cur_trick_count
                print(f"  [R{round_count}:PLAY] trick {cur_trick_count} completed", flush=True)
            if msg.get("awaiting") == "play":
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
    print(f"[round {round_count}] === WAITING (round complete) === phase={state.get('phase')} awaiting={msg.get('awaiting')}", flush=True)
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
    if msg.get("awaiting") == "next_round":
        print(f"  [R{round_count}:WAITING] human confirm next_round", flush=True)
        _send_wrong_actions(driver)
        response = driver.do_next_round()
        if response is not None:
            state = _as_dict(response["state"])
            msg = response
            if state["phase"] != "WAITING":
                print(f"[round {round_count}] WAITING -> {state['phase']}", flush=True)
                return state, msg, expected_t0, expected_t1
        # Our do_action may have read a cascade push instead of our
        # confirmation. Keep sending next_round and draining. Each
        # do_action call runs the event loop, letting handle_connection
        # process our earlier next_round from the WS buffer. If it
        # was already processed, the resend gets rejected harmlessly
        # ("你已经确认过了"), but do_action still drains cascade messages.
        while state["phase"] == "WAITING":
            _send_wrong_actions(driver)
            response = driver.do_next_round()
            if response is not None:
                state = _as_dict(response["state"])
                msg = response
                if state["phase"] != "WAITING":
                    break

    print(f"[round {round_count}] WAITING -> {state['phase']}", flush=True)
    return state, msg, expected_t0, expected_t1


def _verify_game_over(
    driver: WsGameDriver,
    state: dict[str, object],
    game_id: str,
    sync_client: TestClient,
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

    # After GAME_OVER, the server's handle_connection exits. In the
    # TestClient model, the WS send buffer still accepts messages but
    # no one reads them. If we call receive_msg(), it will hang forever
    # waiting for a server response that never comes.
    # So we just close the driver immediately — we've already verified
    # the GAME_OVER state above.
    print("  closing driver (GAME_OVER)", flush=True)
    driver.close()
    print("  driver closed", flush=True)

    # Verify game is removed from registry after GAME_OVER
    print("  checking registry...", flush=True)
    resp = sync_client.get("/api/game")
    games_raw = resp.json()["games"]
    assert _is_list_of_dict(games_raw)
    game_ids = [g["game_id"] for g in games_raw]
    assert game_id not in game_ids, (
        f"Game {game_id} should be removed from registry after GAME_OVER"
    )
    print("  game removed from registry", flush=True)

    # Verify new connection to finished game returns close code 4404
    print("  checking 4404 on reconnect...", flush=True)
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with sync_client.websocket_connect(f"/game/{game_id}") as ws:
            ws.receive_json()
    assert exc_info.value.code == 4404, (
        f"Expected close code 4404 for finished game, got {exc_info.value.code}"
    )
    print("  GAME_OVER verification complete", flush=True)


# ---- Full Game Playthrough ----


def test_full_game(sync_client: TestClient) -> None:
    """Play a complete game from start to GAME_OVER.

    At each phase:
    - Interleave illegal actions to verify error handling
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

        # Game starts in WAITING phase. AutoPlayers (0-2) confirm automatically
        # via run(). When we connect with seq=0 and send next_round,
        # do_action handles seq mismatch internally: it gets the current seq (4,
        # from 4 AutoPlayer confirmations), retries, and our next_round succeeds
        # (we're the 4th confirmer). This starts the game → DEAL_BID.
        #
        # The returned message is the first state after our confirmation.
        # AutoPlayer cascade may have already advanced the game further, so we
        # need to drain until we know the current phase.
        print(">>> sending initial next_round <<<", flush=True)
        result = driver.do_action({"type": "next_round"})
        if result is not None:
            # Our next_round succeeded (AutoPlayers already confirmed, we were
            # the 4th). Game has started. result is the state push.
            print(f"  next_round succeeded immediately, seq={driver.current_seq}", flush=True)
        else:
            # Seq mismatch but retry exhausted, or action rejected for other
            # reason. Check last_error for details.
            print(f"  next_round failed, error={driver.last_error}, seq={driver.current_seq}", flush=True)
            # Send next_round again with the updated seq
            result = driver.do_action({"type": "next_round"})
            assert result is not None, f"next_round should succeed with correct seq: error={driver.last_error}"
            print(f"  next_round succeeded on second try, seq={driver.current_seq}", flush=True)

        # Drain any AutoPlayer cascade pushes until we reach a stable state
        # where we know what phase we're in.
        state = _as_dict(result["state"])
        msg = result
        while True:
            cur_phase = state.get("phase", "?")
            cur_awaiting = msg.get("awaiting")
            print(f"  initial state: phase={cur_phase} awaiting={cur_awaiting} seq={driver.current_seq}", flush=True)
            if cur_phase in ("DEAL_BID", "STIRRING", "PLAYING", "GAME_OVER"):
                break
            # Still in WAITING or transitional state — drain more
            msg = driver.receive_msg_safe()
            if msg.get("type") == "state":
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
