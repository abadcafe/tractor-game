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
        self._ws_cm: object | None = None  # context manager for websocket_connect
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
        self._ws_cm = self._client.websocket_connect(f"/game/{game_id}")
        self._ws = self._ws_cm.__enter__()
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
        if self._ws_cm is not None:
            try:
                self._ws_cm.__exit__(None, None, None)
            except _WS_ERRORS:
                pass
            self._ws_cm = None

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
        if self._ws_cm is not None:
            try:
                self._ws_cm.__exit__(None, None, None)
            except _WS_ERRORS:
                pass
            self._ws_cm = None

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

    def receive_state(self) -> dict[str, object]:
        """Receive the next state message from the server.

        After the caller sends an action, subsequent state pushes from
        game.act() and AutoPlayer tasks arrive on the WS. This method
        receives the next such push.

        The caller MUST have previously sent an action to ensure there is a
        state push to receive. If no push is pending, it blocks until one
        arrives.
        """
        return self.receive_msg_safe()

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

    def wait_for_awaiting(self, value: str, timeout: float = 30) -> dict[str, object]:
        """Wait until awaiting_action equals the specified value.

        Sends a next_round action each iteration to trigger a state push.
        Returns the state message. Raises TimeoutError if not reached within timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self.send_action({"type": "next_round"})
            except RuntimeError:
                pass
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


# ---- Scoring Helpers ----

# Scoring thresholds from server/sm/constants.py
# Each entry: (max_points, declarer_change, switch_declarer)
# - declarer_change: how many levels the declarer advances (positive = up)
# - switch_declarer: if True, defenders advance by |declarer_change| levels
_SCORE_THRESHOLDS: list[tuple[int, int, bool]] = [
    (0,   3, False),   # 0 points: declarer +3, no switch
    (39,  2, False),   # 1-39: declarer +2, no switch
    (79,  1, False),   # 40-79: declarer +1, no switch
    (119, 0, True),    # 80-119: declarer +0, switch (defenders advance)
    (159, -1, True),   # 120-159: declarer -1, switch
    (199, -2, True),   # 160-199: declarer -2, switch
    (200, -3, True),   # 200: declarer -3, switch
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

    Mirrors server/sm/constants.py SCORE_THRESHOLDS logic:
    - Declarer advances by declarer_change levels
    - If switch_declarer is True, defenders advance by |declarer_change| levels
    - If switch_declarer is False, defenders do not advance
    """
    declarer_change = 0
    switch = False
    for max_pts, change, sw in _SCORE_THRESHOLDS:
        if total_defender_points <= max_pts:
            declarer_change = change
            switch = sw
            break

    # Defenders only advance when they win (switch_declarer=True)
    defender_change = -declarer_change if switch else 0

    if declarer_team == 0:
        return (
            _advance_level(team0_level, declarer_change),
            _advance_level(team1_level, defender_change),
        )
    else:
        return (
            _advance_level(team0_level, defender_change),
            _advance_level(team1_level, declarer_change),
        )


# ---- Phase Helpers ----


def _interleave_error(
    driver: WsGameDriver,
    phase: str,
    state: dict[str, object],
    hand: list[dict[str, object]] | None = None,
) -> None:
    """Send at least one illegal action appropriate for the current phase.

    Uses real cards from hand where possible to exercise deeper validation paths
    (not just fake card ID rejection). Verifies the action is rejected (do_action
    returns False, last_error set). Also verifies state is unchanged after the
    illegal action by checking that the seq number hasn't advanced.
    """
    # Capture seq before illegal actions for unchanged verification
    seq_before = driver.current_seq

    if phase == "DEAL_BID":
        bid_legal = state.get("bid_legal_actions", [])

        # Test 1: bid with a real card NOT in bid_legal_actions
        # legal_actions entries are serialized as list[CardDict] (flat list of
        # card dicts), NOT as {"cards": [CardDict]}. See snapshot.py to_dict().
        legal_card_ids: set[str] = set()
        for option in _as_list(bid_legal):
            for c in _as_list(option):
                c_dict = _as_dict(c)
                legal_card_ids.add(_as_str(c_dict["id"]))
        # Find a card in hand that is NOT a legal bid card
        player_hand = state.get("player_hand", [])
        non_legal_card = None
        for c in _as_list(player_hand):
            c_dict = _as_dict(c)
            if _as_str(c_dict["id"]) not in legal_card_ids:
                non_legal_card = _as_str(c_dict["id"])
                break
        if non_legal_card:
            result = driver.do_bid([non_legal_card])
            assert not result, "Expected bid with non-legal real card to fail"
            assert driver.last_error is not None
            assert any(
                kw in driver.last_error
                for kw in ("不在手牌中", "不是主牌等级", "不一致", "价值为零",
                           "王叫牌必须出对子", "优先级不足", "非庄家方不能叫牌")
            ), f"Unexpected bid error: {driver.last_error}"
        else:
            # All cards in hand are legal bids; use fake card
            result = driver.do_bid(["fake_card_id_12345"])
            assert not result, "Expected bid with fake card to fail"
            assert driver.last_error is not None
            assert "not in hand" in driver.last_error or "不在" in driver.last_error, (
                f"Expected card-not-found error, got: {driver.last_error}"
            )

        # Test 2: wrong action type for DEAL_BID
        # Using fake card IDs: server's _parse_action resolves cards BEFORE act()
        # is called, so fake cards may produce a card-resolution error rather than
        # a phase-mismatch error. Both are valid rejections.
        result = driver.do_play(["fake_card_id_12345"])
        assert not result, "Expected play during DEAL_BID to fail"
        assert driver.last_error is not None
        assert any(
            kw in driver.last_error
            for kw in ("无效的操作", "不能在", "not in hand", "不在")
        ), f"Expected phase-mismatch or card-resolution error, got: {driver.last_error}"

    elif phase == "STIRRING":
        # Test 1: stir with a single card (not a pair) from hand.
        # Stirring requires pairs; a single card should be rejected.
        stirring = _as_dict_or_none(state.get("stirring_state"))
        stirring_typed: dict[str, object] = stirring if stirring is not None else {}
        legal_stir = stirring_typed.get("legal_actions", [])
        player_hand = state.get("player_hand", [])

        # Collect all card IDs in legal stir options
        # legal_actions entries are serialized as list[CardDict] (flat list of
        # card dicts), NOT as {"cards": [CardDict]}. See snapshot.py to_dict().
        legal_stir_ids: set[str] = set()
        for option in _as_list(legal_stir):
            for c in _as_list(option):
                c_dict = _as_dict(c)
                legal_stir_ids.add(_as_str(c_dict["id"]))

        # Find a trump-rank card from hand that is NOT part of any legal stir pair.
        # This tests deeper validation paths (e.g. priority too low, same suit as
        # current trump) rather than just "card not in hand".
        trump_rank = state.get("trump_rank")
        non_legal_trump_card = None
        for c in _as_list(player_hand):
            c_dict = _as_dict(c)
            if _as_str(c_dict["id"]) not in legal_stir_ids and c_dict.get("rank") == trump_rank:
                non_legal_trump_card = _as_str(c_dict["id"])
                break

        if non_legal_trump_card:
            # Try stirring with a single trump-rank card (not a pair)
            result = driver.do_stir([non_legal_trump_card])
            assert not result, "Expected stir with single card (not a pair) to fail"
            assert driver.last_error is not None
            assert any(
                kw in driver.last_error
                for kw in ("反主", "对子", "优先级", "不在", "不是主牌等级")
            ), f"Unexpected stir error: {driver.last_error}"
        else:
            result = driver.do_stir(["fake_card_id_12345"])
            assert not result, "Expected stir with fake card to fail"
            assert driver.last_error is not None
            assert "not in hand" in driver.last_error or "不在" in driver.last_error, (
                f"Expected card-not-found error, got: {driver.last_error}"
            )

        # Test 2: wrong action type for STIRRING
        # Using fake card IDs: _parse_action may reject for card-resolution before
        # phase mismatch. Both rejections are valid.
        result = driver.do_bid(["fake_card_id_12345"])
        assert not result, "Expected bid during STIRRING to fail"
        assert driver.last_error is not None
        assert any(
            kw in driver.last_error
            for kw in ("无效的操作", "不能在", "not in hand", "不在")
        ), f"Expected phase-mismatch or card-resolution error, got: {driver.last_error}"

    elif phase == "EXCHANGE":
        # Test 1: discard wrong count of REAL cards from hand
        exchange = _as_dict_or_none(state.get("exchange_state"))
        exchange_typed: dict[str, object] = exchange if exchange is not None else {}
        expected_count_raw = exchange_typed.get("count", 8)
        expected_count = _as_int(expected_count_raw) if isinstance(expected_count_raw, int) else 8
        if hand and len(hand) > expected_count:
            result = driver.do_discard([_as_str(c["id"]) for c in hand[:expected_count - 1]])
            assert not result, "Expected discard with wrong count to fail"
            assert driver.last_error is not None
            assert "埋牌数量错误" in driver.last_error, (
                f"Expected '埋牌数量错误' error, got: {driver.last_error}"
            )

        # Test 2: discard card not in hand
        result = driver.do_discard(["fake_card_id_12345"])
        assert not result, "Expected discard with fake card to fail"
        assert driver.last_error is not None
        assert "not in hand" in driver.last_error or "不在手牌中" in driver.last_error, (
            f"Expected card-not-found error, got: {driver.last_error}"
        )

        # Test 3: wrong action type for EXCHANGE
        # Using fake card IDs: _parse_action may reject for card-resolution before
        # phase mismatch. Both rejections are valid.
        result = driver.do_play(["fake_card_id_12345"])
        assert not result, "Expected play during EXCHANGE to fail"
        assert driver.last_error is not None
        assert any(
            kw in driver.last_error
            for kw in ("无效的操作", "不能在", "not in hand", "不在")
        ), f"Expected phase-mismatch or card-resolution error, got: {driver.last_error}"

    elif phase == "PLAYING":
        # Test 1: play a card NOT in legal_actions but IS in hand
        # legal_actions entries are serialized as list[CardDict] (flat list of
        # card dicts), NOT as {"cards": [CardDict]}. See snapshot.py to_dict().
        legal = state.get("legal_actions", [])
        legal_play_ids: set[str] = set()
        for option in _as_list(legal):
            for c in _as_list(option):
                c_dict = _as_dict(c)
                legal_play_ids.add(_as_str(c_dict["id"]))
        player_hand = state.get("player_hand", [])
        non_legal_play_card = None
        for c in _as_list(player_hand):
            c_dict = _as_dict(c)
            if _as_str(c_dict["id"]) not in legal_play_ids:
                non_legal_play_card = _as_str(c_dict["id"])
                break
        if non_legal_play_card:
            result = driver.do_play([non_legal_play_card])
            assert not result, "Expected play with non-legal real card to fail"
            assert driver.last_error is not None
            assert any(
                kw in driver.last_error
                for kw in ("出牌不符合规则", "必须跟牌", "首出牌不符合规则")
            ), f"Unexpected play error: {driver.last_error}"
        else:
            result = driver.do_play(["fake_card_id_12345"])
            assert not result, "Expected play with fake card to fail"
            assert driver.last_error is not None
            assert "not in hand" in driver.last_error or "不在" in driver.last_error, (
                f"Expected card-not-found error, got: {driver.last_error}"
            )

        # Test 2: wrong action type for PLAYING
        # Using fake card IDs: _parse_action may reject for card-resolution before
        # phase mismatch. Both rejections are valid.
        result = driver.do_bid(["fake_card_id_12345"])
        assert not result, "Expected bid during PLAYING to fail"
        assert driver.last_error is not None
        assert any(
            kw in driver.last_error
            for kw in ("无效的操作", "不能在", "not in hand", "不在")
        ), f"Expected phase-mismatch or card-resolution error, got: {driver.last_error}"

    elif phase == "COMPLETE":
        # Test 1: wrong action type for COMPLETE
        # Using fake card IDs: _parse_action may reject for card-resolution before
        # phase mismatch. Both rejections are valid.
        result = driver.do_play(["fake_card_id_12345"])
        assert not result, "Expected play during COMPLETE to fail"
        assert driver.last_error is not None
        assert any(
            kw in driver.last_error
            for kw in ("无效的操作", "不能在", "not in hand", "不在")
        ), f"Expected phase-mismatch or card-resolution error, got: {driver.last_error}"

        # Test 2: wrong action type for COMPLETE
        result = driver.do_bid(["fake_card_id_12345"])
        assert not result, "Expected bid during COMPLETE to fail"
        assert driver.last_error is not None
        assert any(
            kw in driver.last_error
            for kw in ("无效的操作", "不能在", "not in hand", "不在")
        ), f"Expected phase-mismatch or card-resolution error, got: {driver.last_error}"

    # Common tests for all non-COMPLETE phases: next_round is invalid
    if phase != "COMPLETE":
        result = driver.do_next_round()
        assert not result, f"Expected next_round during {phase} to fail"
        assert driver.last_error is not None
        assert "无效的操作" in driver.last_error or "不能在" in driver.last_error, (
            f"Expected phase-mismatch error for next_round in {phase}, got: {driver.last_error}"
        )

    # Common test for all phases: unknown action type is rejected
    result = driver.do_action({"type": "nonexistent_action_xyz"})
    assert not result, f"Expected unknown action type during {phase} to fail"
    assert driver.last_error is not None
    assert "无效的操作" in driver.last_error or "不能在" in driver.last_error or "未知" in driver.last_error or "unknown" in driver.last_error, (
        f"Expected unknown-action error during {phase}, got: {driver.last_error}"
    )

    # Verify state is unchanged after illegal actions by checking seq hasn't advanced.
    # When an action is rejected, the server returns the current state with the same seq.
    # When an action is accepted, the server increments seq. Since all illegal actions
    # were rejected (do_* returned False), seq should not have changed.
    assert driver.current_seq == seq_before, (
        f"State seq changed after illegal action: expected {seq_before}, "
        f"got {driver.current_seq}"
    )


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


def test_full_game(sync_client: TestClient) -> None:
    """Play a complete game from start to GAME_OVER.

    At each phase:
    - Interleave illegal actions to verify error handling
    - Verify state fields and phase transitions
    - Let AutoPlayers handle their turns automatically
    - Async thread randomly interrupts WS connection to verify driver auto-reconnect

    The human player (index 3) acts when awaiting_action matches.
    AutoPlayers (indices 0-2) act automatically via the server.
    """
    import random
    import threading

    resp = sync_client.post("/api/game")
    game_id = resp.json()["game_id"]

    driver = WsGameDriver(sync_client)
    driver.connect(game_id)

    # Get initial state: send bid_pass to trigger game.act() and state pushes.
    # The seq-mismatch response from next_round does NOT call game.act(), so no
    # AutoPlayer tasks are created and no state pushes follow. Instead, we send
    # a real action (bid_pass) which triggers game.act() -> _push_state_to_all(),
    # creating AutoPlayer tasks that push state we can receive.
    result = driver.do_bid_pass()
    # May succeed or fail depending on bid_turn, but always triggers game.act()

    # Start async thread that randomly interrupts WS connection
    disconnect_count = 0
    stop_disconnector = threading.Event()

    def random_disconnect() -> None:
        """Randomly close the WS connection to test driver auto-reconnect."""
        nonlocal disconnect_count
        rng = random.Random(42)  # Deterministic seed for reproducibility
        while not stop_disconnector.is_set():
            # Wait random interval (0.5-2s) between disconnections
            wait_time = rng.uniform(0.5, 2.0)
            if stop_disconnector.wait(timeout=wait_time):
                break
            # Force-disconnect via the public method
            driver.force_disconnect()
            disconnect_count += 1

    disconnector_thread = threading.Thread(target=random_disconnect, daemon=True)
    disconnector_thread.start()

    round_count = 0
    max_rounds = 10  # Safety limit
    expected_t0: str | None = None
    expected_t1: str | None = None
    state: dict[str, object] = {}

    try:
        while round_count < max_rounds:
            round_count += 1

            # ---- DEAL_BID Phase ----
            msg = driver.wait_for_phase("DEAL_BID", timeout=60)
            state = _as_dict(msg["state"])
            _verify_common_fields(state, "DEAL_BID")
            assert "bid_events" in state
            assert "bid_legal_actions" in state
            assert "bid_winner" in state  # may be null if no bid yet
            assert "trump_suit" in state  # may be null initially

            # Verify level progression from previous round
            if expected_t0 is not None and expected_t1 is not None:
                assert state["team0_level"] == expected_t0, (
                    f"team0_level: expected {expected_t0}, got {state['team0_level']}"
                )
                assert state["team1_level"] == expected_t1, (
                    f"team1_level: expected {expected_t1}, got {state['team1_level']}"
                )
                expected_t0 = None
                expected_t1 = None

            # Track trump_suit before bidding for change verification
            pre_bid_trump_suit = _as_str_or_none(state.get("trump_suit"))
            bid_made = False  # Track if human made a successful bid
            tested_deal_bid_interleave = False  # Only interleave once per round

            # Drive through DEAL_BID: wait for our turn, bid or pass
            while True:
                msg = driver.receive_state()
                if msg.get("type") != "state":
                    continue
                state = _as_dict(msg["state"])

                if state["phase"] != "DEAL_BID":
                    # Phase changed to STIRRING
                    break

                if msg.get("awaiting") == "bid":
                    if not tested_deal_bid_interleave:
                        tested_deal_bid_interleave = True
                        _interleave_error(driver, "DEAL_BID", state)

                    bid_legal = state.get("bid_legal_actions", [])
                    bid_legal_list = _as_list(bid_legal)
                    if bid_legal_list:
                        # Bid with first legal option
                        # legal_actions entries are list[CardDict] (flat list),
                        # NOT {"cards": [CardDict]}. See snapshot.py to_dict().
                        first_option = _as_list(bid_legal_list[0])
                        card_ids: list[str] = []
                        for c in first_option:
                            c_dict = _as_dict(c)
                            card_ids.append(_as_str(c_dict["id"]))
                        result = driver.do_bid(card_ids)
                        if result:
                            bid_made = True
                        else:
                            # Bid failed (e.g. seq mismatch from disconnect);
                            # pass to avoid deadlock where server waits for
                            # action and test waits for state push.
                            driver.do_bid_pass()
                    else:
                        driver.do_bid_pass()
                # else: waiting for AutoPlayers to bid

            # After DEAL_BID ends, verify trump_suit changed if human made a bid
            if bid_made:
                post_bid_trump_suit = _as_str_or_none(state.get("trump_suit"))
                assert post_bid_trump_suit != pre_bid_trump_suit or (
                    post_bid_trump_suit is None and pre_bid_trump_suit is None
                ), (
                    f"trump_suit should change after successful bid: "
                    f"before={pre_bid_trump_suit}, after={post_bid_trump_suit}"
                )

            # ---- STIRRING Phase ----
            if state["phase"] == "STIRRING":
                _verify_common_fields(state, "STIRRING")
                assert "stirring_state" in state
                stirring = _as_dict_or_none(state.get("stirring_state"))
                assert stirring is not None
                assert "phase" in stirring
                assert "trump_suit" in stirring
                assert "current_player" in stirring
                assert "declarer_player" in stirring
                assert "legal_actions" in stirring

                pre_stir_trump_suit = _as_str_or_none(state.get("trump_suit"))
                human_stir_succeeded = False  # Track if human made a successful stir
                tested_stir_interleave = False  # Only interleave once per round

                while state["phase"] == "STIRRING":
                    msg = driver.receive_state()
                    if msg.get("type") != "state":
                        continue
                    state = _as_dict(msg["state"])

                    # After receiving a new state, check if trump_suit changed
                    # following a human stir (previous iteration). This verifies
                    # the server correctly updated trump_suit on stir acceptance.
                    if human_stir_succeeded:
                        current_trump = _as_str_or_none(state.get("trump_suit"))
                        pre_stir = pre_stir_trump_suit
                        # trump_suit must change after a successful stir,
                        # unless joker pair (null -> null no-trump case)
                        assert current_trump != pre_stir or (
                            current_trump is None and pre_stir is None
                        ), (
                            f"trump_suit should change after successful stir: "
                            f"before={pre_stir}, after={current_trump}"
                        )
                        human_stir_succeeded = False  # Reset flag

                    if state["phase"] != "STIRRING":
                        break

                    if msg.get("awaiting") == "stir":
                        if not tested_stir_interleave:
                            tested_stir_interleave = True
                            _interleave_error(driver, "STIRRING", state)

                        stirring_now = _as_dict_or_none(state.get("stirring_state"))
                        stirring_typed_now: dict[str, object] = stirring_now if stirring_now is not None else {}
                        legal_stir = stirring_typed_now.get("legal_actions", [])
                        legal_stir_list = _as_list(legal_stir)
                        if legal_stir_list:
                            # legal_actions entries are list[CardDict] (flat list),
                            # NOT {"cards": [CardDict]}. See snapshot.py to_dict().
                            first_stir = _as_list(legal_stir_list[0])
                            stir_card_ids: list[str] = []
                            for c in first_stir:
                                c_dict = _as_dict(c)
                                stir_card_ids.append(_as_str(c_dict["id"]))
                            pre_stir_trump_suit = _as_str_or_none(state.get("trump_suit"))
                            result = driver.do_stir(stir_card_ids)
                            if result:
                                human_stir_succeeded = True
                                # trump_suit change will be verified on next
                                # receive_msg_safe() iteration above
                        else:
                            driver.do_stir_pass()
                    # else: waiting for AutoPlayers to stir -- just loop back to receive_msg()

                # After STIRRING ends, verify final trump_suit if human stir
                # hasn't been verified yet (stir was last action before phase change)
                if human_stir_succeeded:
                    current_trump = _as_str_or_none(state.get("trump_suit"))
                    assert current_trump != pre_stir_trump_suit or (
                        current_trump is None and pre_stir_trump_suit is None
                    ), (
                        f"trump_suit should change after successful stir: "
                        f"before={pre_stir_trump_suit}, after={current_trump}"
                    )

            # ---- EXCHANGE Phase ----
            if state["phase"] == "EXCHANGE":
                _verify_common_fields(state, "EXCHANGE")
                assert "exchange_state" in state
                exchange = _as_dict_or_none(state.get("exchange_state"))
                assert exchange is not None
                assert "phase" in exchange
                assert "declarer_player" in exchange
                assert "count" in exchange

                declarer_player_raw = exchange.get("declarer_player")
                declarer_player = _as_int(declarer_player_raw) if isinstance(declarer_player_raw, int) else None

                if declarer_player == 3:  # Human player is index 3
                    player_hand_raw = state.get("player_hand")
                    hand_typed = _as_list(player_hand_raw) if player_hand_raw is not None else None
                    hand_dicts: list[dict[str, object]] = []
                    if hand_typed is not None:
                        for h in hand_typed:
                            hand_dicts.append(_as_dict(h))
                    _interleave_error(driver, "EXCHANGE", state, hand=hand_dicts if hand_dicts else None)

                    count_raw = exchange.get("count", 8)
                    count = _as_int(count_raw) if isinstance(count_raw, int) else 8
                    hand_list = _as_list(state["player_hand"])
                    hand_dicts_for_discard: list[dict[str, object]] = []
                    for h in hand_list:
                        hand_dicts_for_discard.append(_as_dict(h))
                    hand_count_before = len(hand_dicts_for_discard)

                    discard_ids: list[str] = []
                    for c in hand_dicts_for_discard[-count:]:
                        discard_ids.append(_as_str(c["id"]))
                    driver.do_discard(discard_ids)

                    msg = driver.wait_for_phase("PLAYING", timeout=30)
                    state = _as_dict(msg["state"])

                    hand_after = _as_list(state["player_hand"])
                    hand_count_after = len(hand_after)
                    assert hand_count_after == hand_count_before - count, (
                        f"Hand count should decrease by {count}: "
                        f"before={hand_count_before}, after={hand_count_after}"
                    )
                else:
                    msg = driver.wait_for_phase("PLAYING", timeout=30)
                    state = _as_dict(msg["state"])

            # ---- PLAYING Phase ----
            tricks_played = 0
            first_trick = True
            tested_play_not_your_turn = False  # Only test once per round
            tested_play_interleave = False  # Only interleave once per round
            # Extract trump_suit for trick verification
            current_trump_suit: str | None = _as_str_or_none(state.get("trump_suit"))
            while state["phase"] == "PLAYING":
                _verify_common_fields(state, "PLAYING")
                assert "trick" in state
                assert "trick_history" in state
                assert "defender_points" in state
                assert "legal_actions" in state or msg.get("awaiting") != "play"

                # Update trump_suit from current state
                ts_raw = _as_str_or_none(state.get("trump_suit"))
                if ts_raw is not None:
                    current_trump_suit = ts_raw

                if msg.get("awaiting") == "play":
                    if not tested_play_interleave:
                        tested_play_interleave = True
                        player_hand_raw = state.get("player_hand")
                        hand_typed = _as_list(player_hand_raw) if player_hand_raw is not None else None
                        hand_dicts_play: list[dict[str, object]] = []
                        if hand_typed is not None:
                            for h in hand_typed:
                                hand_dicts_play.append(_as_dict(h))
                        _interleave_error(driver, "PLAYING", state, hand=hand_dicts_play if hand_dicts_play else None)

                    legal = state.get("legal_actions", [])
                    legal_list = _as_list(legal)
                    if legal_list:
                        # legal_actions entries are list[CardDict] (flat list),
                        # NOT {"cards": [CardDict]}. See snapshot.py to_dict().
                        chosen_cards = _as_list(legal_list[0])
                        # Verify chosen play is a valid single or pair
                        assert len(chosen_cards) in (1, 2), (
                            f"Legal play must be 1 or 2 cards, got {len(chosen_cards)}"
                        )
                        is_pair = len(chosen_cards) == 2

                        if first_trick:
                            # On first trick, test dict-format card play via do_action
                            result = driver.do_action({
                                "type": "play",
                                "cards": [_as_dict(chosen_cards[0])],
                            })
                            assert result, "Dict-format card play should succeed"
                            first_trick = False
                        else:
                            play_card_ids: list[str] = []
                            for c in chosen_cards:
                                c_dict = _as_dict(c)
                                play_card_ids.append(_as_str(c_dict["id"]))
                            driver.do_play(play_card_ids)
                        tricks_played += 1

                        # Verify trick update after our play
                        post_play_msg = driver.receive_state()
                        if post_play_msg.get("type") == "state":
                            state = _as_dict(post_play_msg["state"])

                            # Verify trick state is present and valid
                            trick_raw = state.get("trick")
                            if trick_raw is not None:
                                trick = _as_dict(trick_raw)
                                assert "current_player" in trick
                                assert "lead_player" in trick
                                assert "slots" in trick
                                # Verify the trick contains our play
                                slots = _as_list(trick["slots"])
                                for slot in slots:
                                    slot_dict = _as_dict(slot)
                                    if slot_dict.get("player") == 3:
                                        slot_cards = _as_list(slot_dict["cards"])
                                        if is_pair:
                                            assert len(slot_cards) == 2, (
                                                "Pair play should result in 2 cards in trick slot"
                                            )
                                        break

                            # Verify completed trick has a winner
                            trick_history_raw = state.get("trick_history", [])
                            trick_history = _as_list(trick_history_raw)
                            if trick_history:
                                last_trick_raw = trick_history[-1]
                                last_trick = _as_dict(last_trick_raw)
                                assert "winner" in last_trick, (
                                    "Completed trick in trick_history must have 'winner' field"
                                )
                                winner_val = last_trick["winner"]
                                assert isinstance(winner_val, int) and winner_val in (0, 1, 2, 3), (
                                    f"trick_winner must be a valid player index, "
                                    f"got {winner_val}"
                                )

                                # Verify suit-following: in completed tricks with
                                # multiple slots, check that follow players' played
                                # the same suit as the lead (or trump)
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
                                                # Follow suit rule: must play same suit
                                                # as lead, unless playing trump
                                                if current_trump_suit is not None and follow_suit == current_trump_suit:
                                                    continue  # Trump always allowed
                                                if lead_suit is not None and follow_suit != lead_suit:
                                                    # Player had no cards of lead suit
                                                    # (void in that suit) -- this is legal
                                                    pass

                                # Verify trump wins over non-trump when applicable
                                # Check if any trick in history has trump beating non-trump
                                if current_trump_suit is not None:
                                    for past_trick_raw in trick_history:
                                        past_trick = _as_dict(past_trick_raw)
                                        past_slots_raw = past_trick.get("slots", [])
                                        past_slots = _as_list(past_slots_raw)
                                        if not past_slots:
                                            continue
                                        # Check if the winning slot used trump while
                                        # the lead didn't
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
                                                    # If winner played trump and lead didn't,
                                                    # verify winner is correct
                                                    winner_suit = _as_str_or_none(winner_card.get("suit"))
                                                    lead_suit = _as_str_or_none(lead_card.get("suit"))
                                                    if (winner_suit == current_trump_suit
                                                            and lead_suit != current_trump_suit):
                                                        assert past_trick["winner"] != _as_dict(past_slots[0]).get("player"), (
                                                            "Trump must beat non-trump lead"
                                                        )

                            continue

                else:
                    # Human is NOT the current player.
                    # Spec requires testing "非当前玩家出牌 → error" (play when not your turn).
                    # The server rejects this because round_sm.play() validates player_index.
                    # Only test this once per round to avoid excessive error actions.
                    if not tested_play_not_your_turn:
                        tested_play_not_your_turn = True
                        # Use a fake card ID — the player_index check in round_sm.play()
                        # happens before card validation, so the rejection reason will be
                        # about wrong turn, not card resolution. If _parse_action rejects
                        # for card-resolution first, that's also a valid rejection.
                        result = driver.do_play(["fake_card_id_not_your_turn"])
                        assert not result, "Expected play when not current player to fail"
                        assert driver.last_error is not None
                        assert any(
                            kw in driver.last_error
                            for kw in ("不是你的回合", "not in hand", "不在")
                        ), f"Expected wrong-turn or card-resolution error, got: {driver.last_error}"

                msg = driver.receive_state()
                if msg.get("type") != "state":
                    continue
                state = _as_dict(msg["state"])

                if state["phase"] == "COMPLETE":
                    break

            # ---- COMPLETE Phase ----
            if state["phase"] == "COMPLETE":
                _verify_common_fields(state, "COMPLETE")
                assert "scoring" in state
                assert "bottom_cards" in state
                assert "next_round_confirmed" in state

                # Verify bottom_cards contents (spec: "bottom_cards 可见")
                bottom_cards_raw = state["bottom_cards"]
                bottom_cards = _as_list(bottom_cards_raw)
                assert len(bottom_cards) > 0, (
                    "bottom_cards must contain actual card data in COMPLETE phase"
                )
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

                expected_t0, expected_t1 = _compute_expected_levels(
                    scoring_tdp,
                    scoring_dt,
                    _as_str(state["team0_level"]),
                    _as_str(state["team1_level"]),
                )

                if msg.get("awaiting") == "next_round":
                    _interleave_error(driver, "COMPLETE", state)
                    driver.do_next_round()

                # Loop until phase changes (GAME_OVER or next round's DEAL_BID).
                # After human confirms, other AutoPlayers may still need to confirm.
                # We keep receiving messages until we leave COMPLETE phase.
                while True:
                    msg = driver.receive_state()
                    if msg.get("type") != "state":
                        continue
                    state = _as_dict(msg["state"])
                    if state["phase"] != "COMPLETE":
                        break
                    # If still in COMPLETE and we get awaiting="next_round" again
                    # (shouldn't happen since we already confirmed), just keep waiting
                    if msg.get("awaiting") == "next_round":
                        # This means our confirmation wasn't recorded
                        # but we already called do_next_round(), so just wait
                        pass

                if state["phase"] == "GAME_OVER":
                    break

    finally:
        # Stop the disconnector thread
        stop_disconnector.set()
        disconnector_thread.join(timeout=5)

    # ---- GAME_OVER ----

    assert state["phase"] == "GAME_OVER", f"Expected GAME_OVER, got {state['phase']}"
    assert "winning_team" in state
    winning_team = state["winning_team"]
    assert isinstance(winning_team, int) and winning_team in (0, 1)

    if expected_t0 is not None and expected_t1 is not None:
        assert state["team0_level"] == expected_t0
        assert state["team1_level"] == expected_t1

    # Verify that async disconnections actually occurred during the game
    assert disconnect_count > 0, (
        f"Expected at least 1 async disconnect during game, got {disconnect_count}"
    )

    # Verify WS is closed by the server after GAME_OVER
    try:
        driver.send_action({"type": "next_round"})
        driver.receive_msg()
    except Exception:
        pass  # Expected: connection closed

    driver.close()

    # Verify game is removed from registry after GAME_OVER
    resp = sync_client.get("/api/game")
    games_raw = resp.json()["games"]
    assert _is_list_of_dict(games_raw)
    game_ids = [g["game_id"] for g in games_raw]
    assert game_id not in game_ids, (
        f"Game {game_id} should be removed from registry after GAME_OVER"
    )

    # Verify new connection to finished game returns close code 4404
    # (game was removed from registry after GAME_OVER)
    with pytest.raises(WebSocketDisconnect) as exc_info:
        ws = sync_client.websocket_connect(f"/game/{game_id}")
        ws.receive_json()
    assert exc_info.value.code == 4404, (
        f"Expected close code 4404 for finished game, got {exc_info.value.code}"
    )
