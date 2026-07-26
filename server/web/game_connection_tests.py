"""Black-box tests for game WebSocket connections."""

import pytest
from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
)
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from server.web.application_test_client import (
    SyncServerClient,
)
from server.web.application_test_client import (
    game_id_from as _game_id_from,
)
from server.web.application_test_client import (
    is_dict as _is_dict,
)
from server.web.application_test_client import (
    is_list_of_dict as _is_list_of_dict,
)

pytest_plugins: tuple[str, ...] = (
    "server.web.application_test_fixtures",
)


_WS_ERRORS: tuple[type[Exception], ...] = (
    WebSocketDisconnect,
    ClosedResourceError,
    BrokenResourceError,
    EndOfStream,
)


def _receive_ws_dict(
    websocket: WebSocketTestSession,
) -> dict[str, object]:
    message = websocket.receive_json()
    assert _is_dict(message)
    return message


def _seat_ws_path(
    game_id: str, *, seat: str = "c", user_id: str = "user-2"
) -> str:
    return f"/game/{game_id}/seat/{seat}?user_id={user_id}"


def _prepare_ws_game(
    sync_client: SyncServerClient,
    game_id: str,
    *,
    seat: str = "c",
    user_id: str = "user-2",
) -> None:
    attach_response = sync_client.post(
        f"/api/game/{game_id}/seat/{seat}?user_id={user_id}"
    )
    assert attach_response.status_code == 200
    fill_response = sync_client.post(
        f"/api/game/{game_id}/bots?policy=auto&user_id={user_id}"
    )
    assert fill_response.status_code == 200


def _listed_seat(
    sync_client: SyncServerClient,
    game_id: str,
    *,
    seat: str,
    user_id: str,
) -> dict[str, object]:
    response = sync_client.get(f"/api/game?user_id={user_id}")
    assert response.status_code == 200
    document = response.json()
    assert _is_dict(document)
    games = document["games"]
    assert _is_list_of_dict(games)
    matching = [game for game in games if game["game_id"] == game_id]
    assert len(matching) == 1
    seats = matching[0]["seats"]
    assert _is_list_of_dict(seats)
    matching_seats = [item for item in seats if item["seat"] == seat]
    assert len(matching_seats) == 1
    return matching_seats[0]


def test_delete_game_after_ws_connection(
    sync_client: SyncServerClient,
) -> None:
    """
    DELETE removes a game after a client has used seq=0 to fetch state.
    """
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)

    with sync_client.websocket_connect(_seat_ws_path(game_id)) as ws:
        # Get initial state via seq=0 (no auto-push on connect)
        ws.send_json({"seq": 0})
        initial = _receive_ws_dict(ws)
        assert _is_dict(initial)
        assert initial["type"] == "state"

    deleted = sync_client.delete(f"/api/game/{game_id}")
    listed = sync_client.get("/api/game")

    assert deleted.status_code == 200
    document = listed.json()
    assert _is_dict(document)
    games = document["games"]
    assert _is_list_of_dict(games)
    assert all(game["game_id"] != game_id for game in games)


# ---- User player index ----


def test_user_player_can_connect_to_requested_player(
    sync_client: SyncServerClient,
) -> None:
    """A user can enter a specific numbered player via WebSocket."""
    create_resp = sync_client.post("/api/game")
    assert create_resp.status_code == 201
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id, seat="a", user_id="user-0")
    with sync_client.websocket_connect(
        _seat_ws_path(game_id, seat="a", user_id="user-0")
    ) as ws:
        ws.send_json({"seq": 0})
        data = _receive_ws_dict(ws)
        assert _is_dict(data)
        assert data["type"] == "state"
        state = data["state"]
        assert _is_dict(state)
        assert "phase" in state


# ---- WebSocket: Connect ----


def test_ws_seq_zero_after_connect_receives_state(
    sync_client: SyncServerClient,
) -> None:
    """
    Connecting to a game via WebSocket and sending seq=0 should receive
    a state message.
    """
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    with sync_client.websocket_connect(_seat_ws_path(game_id)) as ws:
        # Client must send seq=0 to get initial state (no auto-push on
        # connect)
        ws.send_json({"seq": 0})
        data = _receive_ws_dict(ws)
        assert _is_dict(data)
        assert data["type"] == "state"


def test_ws_connect_nonexistent_rejected(
    sync_client: SyncServerClient,
) -> None:
    """Connecting to a nonexistent game should be rejected."""
    with pytest.raises(_WS_ERRORS):
        with sync_client.websocket_connect(
            _seat_ws_path("nonexistent999")
        ) as ws:
            _receive_ws_dict(ws)


def test_ws_connect_missing_user_id_rejected(
    sync_client: SyncServerClient,
) -> None:
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with sync_client.websocket_connect(
            f"/game/{game_id}/seat/b"
        ) as ws:
            _receive_ws_dict(ws)

    assert exc_info.value.code == 4410


def test_ws_connect_invalid_player_rejected(
    sync_client: SyncServerClient,
) -> None:
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with sync_client.websocket_connect(
            f"/game/{game_id}/seat/4?user_id=user-4"
        ) as ws:
            _receive_ws_dict(ws)

    assert exc_info.value.code == 4410


def test_ws_connect_rejects_player_stealing(
    sync_client: SyncServerClient,
) -> None:
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id, seat="b", user_id="user-1")

    with sync_client.websocket_connect(
        _seat_ws_path(game_id, seat="b", user_id="user-1")
    ) as ws1:
        ws1.send_json({"seq": 0})
        data1 = _receive_ws_dict(ws1)
        assert _is_dict(data1)
        assert data1["type"] == "state"

        with pytest.raises(WebSocketDisconnect) as exc_info:
            with sync_client.websocket_connect(
                _seat_ws_path(game_id, seat="b", user_id="user-other")
            ) as ws2:
                _receive_ws_dict(ws2)

        assert exc_info.value.code == 4409


def test_ws_connect_allows_same_user_to_reenter_player(
    sync_client: SyncServerClient,
) -> None:
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id, seat="d", user_id="user-3")

    with sync_client.websocket_connect(
        _seat_ws_path(game_id, seat="d", user_id="user-3")
    ) as ws1:
        ws1.send_json({"seq": 0})
        data1 = _receive_ws_dict(ws1)
        assert _is_dict(data1)
        assert data1["type"] == "state"

        with sync_client.websocket_connect(
            _seat_ws_path(game_id, seat="d", user_id="user-3")
        ) as ws2:
            ws2.send_json({"seq": 0})
            data2 = _receive_ws_dict(ws2)
            assert _is_dict(data2)
            assert data2["type"] == "state"


def test_ws_connect_takeover_closes_old_connection(
    sync_client: SyncServerClient,
) -> None:
    """When a game already has an active WebSocket connection, a second
    connection should take over: the old connection is closed and the
    new
    one is accepted.
    """
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)

    with sync_client.websocket_connect(_seat_ws_path(game_id)) as ws1:
        ws1.send_json({"seq": 0})
        data1 = _receive_ws_dict(ws1)
        assert _is_dict(data1)
        assert data1["type"] == "state"
        first_seat = _listed_seat(
            sync_client,
            game_id,
            seat="c",
            user_id="user-2",
        )
        first_player = first_seat["player"]
        assert _is_dict(first_player)
        assert first_player["connected"] is True

        with sync_client.websocket_connect(
            _seat_ws_path(game_id)
        ) as ws2:
            ws2.send_json({"seq": 0})
            data2 = _receive_ws_dict(ws2)
            assert _is_dict(data2)
            assert data2["type"] == "state"
            replacement_seat = _listed_seat(
                sync_client,
                game_id,
                seat="c",
                user_id="user-2",
            )
            replacement_player = replacement_seat["player"]
            assert _is_dict(replacement_player)
            assert replacement_player["connected"] is True
            with pytest.raises(_WS_ERRORS):
                _receive_ws_dict(ws1)


# ---- WebSocket: Actions with response verification ----
# Each WS action test sends a message and then verifies that the server
# produces a response (either a state update or an error). This ensures
# the server actually processes the message rather than silently
# discarding it.


def test_ws_bid_action_receives_response(
    sync_client: SyncServerClient,
) -> None:
    """
    Sending a bid action via WebSocket should produce a response from
    the server.
    """
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    with sync_client.websocket_connect(_seat_ws_path(game_id)) as ws:
        ws.send_json({"seq": 0})
        initial = _receive_ws_dict(ws)
        assert _is_dict(initial)
        assert initial["type"] == "state"
        seq = initial["seq"]
        ws.send_json({"type": "bid", "cards": [], "seq": seq})
        response = _receive_ws_dict(ws)
        assert response["type"] == "state"


def test_ws_play_action_receives_response(
    sync_client: SyncServerClient,
) -> None:
    """Sending a play action via WebSocket should produce a response."""
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    with sync_client.websocket_connect(_seat_ws_path(game_id)) as ws:
        ws.send_json({"seq": 0})
        initial = _receive_ws_dict(ws)
        assert _is_dict(initial)
        assert initial["type"] == "state"
        seq = initial["seq"]
        ws.send_json({"type": "play", "cards": [], "seq": seq})
        response = _receive_ws_dict(ws)
        assert response["type"] == "state"


def test_ws_next_round_action_receives_response(
    sync_client: SyncServerClient,
) -> None:
    """Sending a next_round action should produce a response."""
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    with sync_client.websocket_connect(_seat_ws_path(game_id)) as ws:
        ws.send_json({"seq": 0})
        initial = _receive_ws_dict(ws)
        assert _is_dict(initial)
        assert initial["type"] == "state"
        seq = initial["seq"]
        ws.send_json({"type": "next_round", "seq": seq})
        response = _receive_ws_dict(ws)
        assert response["type"] == "state"


def test_ws_stir_action_receives_response(
    sync_client: SyncServerClient,
) -> None:
    """Sending a stir action should produce a response."""
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    with sync_client.websocket_connect(_seat_ws_path(game_id)) as ws:
        ws.send_json({"seq": 0})
        initial = _receive_ws_dict(ws)
        assert _is_dict(initial)
        assert initial["type"] == "state"
        seq = initial["seq"]
        ws.send_json(
            {"type": "stir", "cards": [], "pass": False, "seq": seq}
        )
        response = _receive_ws_dict(ws)
        assert response["type"] == "state"


def test_ws_discard_action_receives_response(
    sync_client: SyncServerClient,
) -> None:
    """Sending a discard action should produce a response."""
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    with sync_client.websocket_connect(_seat_ws_path(game_id)) as ws:
        ws.send_json({"seq": 0})
        initial = _receive_ws_dict(ws)
        assert _is_dict(initial)
        assert initial["type"] == "state"
        seq = initial["seq"]
        ws.send_json({"type": "discard", "cards": [], "seq": seq})
        response = _receive_ws_dict(ws)
        assert response["type"] == "state"


# ---- WebSocket: Error Handling ----


def test_ws_invalid_action_returns_error(
    sync_client: SyncServerClient,
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
    with sync_client.websocket_connect(_seat_ws_path(game_id)) as ws:
        # Get initial state first
        ws.send_json({"seq": 0})
        initial = _receive_ws_dict(ws)
        assert _is_dict(initial)
        assert initial["type"] == "state"
        seq = initial["seq"]

        ws.send_json(
            {"type": "unknown_action", "data": "value", "seq": seq}
        )
        response = _receive_ws_dict(ws)
        assert response["type"] == "state"
        assert response.get("error") is not None


# ---- Reconnect ----


def test_reconnect_replaces_ws(sync_client: SyncServerClient) -> None:
    """Reconnecting to a game should replace the WebSocket reference."""
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    # First connection
    with sync_client.websocket_connect(_seat_ws_path(game_id)) as ws:
        ws.send_json({"seq": 0})
        data = _receive_ws_dict(ws)
        assert _is_dict(data)
    # Second connection (reconnect)
    with sync_client.websocket_connect(_seat_ws_path(game_id)) as ws:
        ws.send_json({"seq": 0})
        raw = _receive_ws_dict(ws)
        assert _is_dict(raw)
        assert raw["type"] == "state"


# ---- WebSocket: Seq Protocol ----


def test_seq_in_state_message(sync_client: SyncServerClient) -> None:
    """State messages must include a 'seq' field starting from 1."""
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    with sync_client.websocket_connect(_seat_ws_path(game_id)) as ws:
        # Send action with seq=0 to get initial state
        ws.send_json({"seq": 0})
        data = _receive_ws_dict(ws)
        assert _is_dict(data)
        assert data["type"] == "state"
        assert "seq" in data
        assert isinstance(data["seq"], int)
        assert data["seq"] >= 1


def test_seq_mismatch_returns_state_without_error(
    sync_client: SyncServerClient,
) -> None:
    """
    When action seq doesn't match, server returns state and ignores the
    action.
    """
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    _prepare_ws_game(sync_client, game_id)
    with sync_client.websocket_connect(_seat_ws_path(game_id)) as ws:
        # Get initial state through the explicit seq=0 state request.
        ws.send_json({"seq": 0})
        data = _receive_ws_dict(ws)
        assert _is_dict(data)
        assert data["type"] == "state"
        known_seq = data["seq"]
        assert isinstance(known_seq, int)
        assert known_seq >= 1

        # Send action with wrong seq (use a value far from current)
        ws.send_json({"type": "next_round", "seq": 99999})
        data = _receive_ws_dict(ws)
        assert _is_dict(data)
        assert data["type"] == "state"
        assert data.get("error") is None
        # Seq must be >= the previously observed seq
        assert isinstance(data["seq"], int)
        assert data["seq"] >= known_seq


def test_error_merged_into_state(sync_client: SyncServerClient) -> None:
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
    with sync_client.websocket_connect(_seat_ws_path(game_id)) as ws:
        # Get initial state
        ws.send_json({"seq": 0})
        data = _receive_ws_dict(ws)
        assert _is_dict(data)
        assert data["type"] == "state"
        seq = data["seq"]

        # Send invalid action (unknown action type) with current seq.
        # This is rejected by Game's player-message parser, so it
        # reliably produces an error.
        ws.send_json({"type": "nonexistent_action_xyz", "seq": seq})
        data = _receive_ws_dict(ws)
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
    sync_client: SyncServerClient,
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
    with sync_client.websocket_connect(_seat_ws_path(game_id)) as ws:
        # Get initial state
        ws.send_json({"seq": 0})
        data = _receive_ws_dict(ws)
        assert _is_dict(data)
        assert data["type"] == "state"
        seq = data["seq"]

        # Send bid pass with correct seq
        ws.send_json({"type": "bid", "pass": True, "seq": seq})
        data = _receive_ws_dict(ws)
        assert _is_dict(data)
        assert data["type"] == "state"
        # SkipBidAction is now handled in DEAL_BID — may succeed or
        # may fail (e.g. if game has moved past DEAL_BID). Either way,
        # we get a valid state response, not a crash or disconnect.


# ---- Static applications ----
