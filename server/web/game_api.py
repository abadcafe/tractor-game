"""Game REST and WebSocket routes."""

from __future__ import annotations

from typing import TypedDict

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

from server.foundation.result import Rejected
from server.game import SeatId, seat_from_id, seat_id
from server.game_runtime.room import BotKind, GameRoom, RoomSeat
from server.web.game_composition import (
    bot_kind_from_str,
    create_game_room,
)
from server.web.game_connection import handle_game_connection
from server.web.state import ServerState

_SEAT_CAPACITY = 4


class SeatResponse(TypedDict):
    seat: SeatId
    occupied: bool
    connected: bool
    kind: str
    mine: bool
    ready: bool


class ListedGameResponse(TypedDict):
    game_id: str
    user_count: int
    capacity: int
    user_seats: list[SeatId]
    seats: list[SeatResponse]


class SeatOperationResponse(TypedDict):
    ok: bool


def register_game_routes(app: FastAPI, state: ServerState) -> None:
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    async def create_game() -> dict[str, str]:
        game_id = state.registry.create(create_game_room())
        return {"game_id": game_id}

    async def list_games(
        user_id: str | None = None,
    ) -> dict[str, list[ListedGameResponse]]:
        return {
            "games": [
                _listed_game_response(state, game_id, user_id)
                for game_id in state.registry.list_ids()
            ]
        }

    async def delete_game(game_id: str) -> dict[str, bool]:
        room = state.registry.delete(game_id)
        if room is not None:
            await room.close()
        return {"ok": True}

    async def occupy_seat(
        game_id: str,
        seat_id: str,
        user_id: str | None = None,
    ) -> JSONResponse:
        room = state.registry.get(game_id)
        if room is None:
            return _seat_error_response("game not found")
        seat = seat_from_id(seat_id)
        if seat is None:
            return _seat_error_response("invalid seat")
        if user_id is None:
            return _seat_error_response("missing user id")
        result = await room.occupy_seat(seat=seat, user_id=user_id)
        if isinstance(result, Rejected):
            return _seat_error_response(result.reason)
        return _seat_ok_response()

    async def vacate_seat(
        game_id: str,
        seat_id: str,
        user_id: str | None = None,
    ) -> JSONResponse:
        room = state.registry.get(game_id)
        if room is None:
            return _seat_error_response("game not found")
        seat = seat_from_id(seat_id)
        if seat is None:
            return _seat_error_response("invalid seat")
        if user_id is None:
            return _seat_error_response("missing user id")
        result = await room.vacate_seat(seat=seat, user_id=user_id)
        if isinstance(result, Rejected):
            return _seat_error_response(result.reason)
        return _seat_ok_response()

    async def fill_bot_seats(
        game_id: str,
        kind: str | None = None,
        user_id: str | None = None,
    ) -> JSONResponse:
        room = state.registry.get(game_id)
        if room is None:
            return _seat_error_response("game not found")
        bot_kind = _bot_kind_response(kind)
        if isinstance(bot_kind, JSONResponse):
            return bot_kind
        if user_id is None:
            return _seat_error_response("missing user id")
        result = await room.fill_bots(
            kind=bot_kind,
            user_id=user_id,
        )
        if isinstance(result, Rejected):
            return _seat_error_response(result.reason)
        return _seat_ok_response()

    async def websocket_game(
        websocket: WebSocket,
        game_id: str,
        seat_id: str,
        user_id: str | None = None,
    ) -> None:
        room = state.registry.get(game_id)
        if room is None:
            await websocket.close(code=4404, reason="game not found")
            return
        seat = seat_from_id(seat_id)
        if seat is None:
            await websocket.close(code=4410, reason="invalid seat")
            return
        if user_id is None:
            await websocket.close(code=4410, reason="missing user id")
            return

        await handle_game_connection(
            websocket,
            room,
            seat=seat,
            user_id=user_id,
        )

    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route(
        "/api/game", create_game, methods=["POST"], status_code=201
    )
    app.add_api_route("/api/game", list_games, methods=["GET"])
    app.add_api_route(
        "/api/game/{game_id}", delete_game, methods=["DELETE"]
    )
    app.add_api_route(
        "/api/game/{game_id}/seat/{seat_id}",
        occupy_seat,
        methods=["POST"],
    )
    app.add_api_route(
        "/api/game/{game_id}/seat/{seat_id}",
        vacate_seat,
        methods=["DELETE"],
    )
    app.add_api_route(
        "/api/game/{game_id}/bots",
        fill_bot_seats,
        methods=["POST"],
    )
    app.add_api_websocket_route(
        "/game/{game_id}/seat/{seat_id}", websocket_game
    )


def _listed_game_response(
    state: ServerState, game_id: str, user_id: str | None
) -> ListedGameResponse:
    room = state.registry.get(game_id)
    room_seats = _room_seats(room, user_id)
    user_seats: list[SeatId] = [
        seat["seat"] for seat in room_seats if seat["kind"] == "user"
    ]
    return {
        "game_id": game_id,
        "user_count": len(user_seats),
        "capacity": _SEAT_CAPACITY,
        "user_seats": user_seats,
        "seats": room_seats,
    }


def _room_seats(
    room: GameRoom | None, user_id: str | None
) -> list[SeatResponse]:
    if room is None:
        return []
    return [
        _seat_response(seat) for seat in room.seats(user_id=user_id)
    ]


def _seat_response(seat: RoomSeat) -> SeatResponse:
    return {
        "seat": seat_id(seat.seat),
        "occupied": seat.occupied,
        "connected": seat.connected,
        "kind": seat.kind,
        "mine": seat.mine,
        "ready": seat.ready,
    }


def _seat_ok_response() -> JSONResponse:
    content: SeatOperationResponse = {"ok": True}
    return JSONResponse(content)


def _seat_error_response(reason: str) -> JSONResponse:
    return JSONResponse(
        {"ok": False, "error": reason},
        status_code=_seat_error_status(reason),
    )


def _seat_error_status(reason: str) -> int:
    match reason:
        case "game not found":
            return 404
        case "invalid seat" | "missing user id" | "invalid bot kind":
            return 400
        case _:
            return 409


def _bot_kind_response(kind: str | None) -> BotKind | JSONResponse:
    bot_kind = bot_kind_from_str(kind)
    if bot_kind is None:
        return _seat_error_response("invalid bot kind")
    return bot_kind
