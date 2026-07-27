"""Game REST and WebSocket routes."""

from __future__ import annotations

from typing import TypedDict

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

from server.foundation.result import Rejected
from server.game import SeatId, seat_from_id, seat_id
from server.game_runtime import (
    BotPolicyName,
    SeatStatus,
    UserId,
)
from server.game_runtime.player import PlayerDescription
from server.web.game_composition import (
    GameInstance,
    bot_policy_name_from_str,
    create_game_instance,
)
from server.web.game_connection import handle_game_connection
from server.web.state import ServerState

_SEAT_CAPACITY = 4


class SeatResponse(TypedDict):
    seat: SeatId
    player: PlayerDescription | None
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
        game_id = state.registry.create(
            create_game_instance(state.ai_service)
        )
        return {"game_id": game_id}

    async def list_games(
        user_id: str | None = None,
    ) -> dict[str, list[ListedGameResponse]]:
        requester = UserId.parse(user_id)
        return {
            "games": [
                _listed_game_response(
                    state.registry.get(game_id),
                    game_id,
                    requester,
                )
                for game_id in state.registry.list_ids()
            ]
        }

    async def delete_game(game_id: str) -> dict[str, bool]:
        instance = state.registry.delete(game_id)
        if instance is not None:
            await instance.close()
        return {"ok": True}

    async def occupy_seat(
        game_id: str,
        seat_id: str,
        user_id: str | None = None,
    ) -> JSONResponse:
        instance = state.registry.get(game_id)
        if instance is None:
            return _seat_error_response("game not found")
        seat = seat_from_id(seat_id)
        if seat is None:
            return _seat_error_response("invalid seat")
        parsed_user = UserId.parse(user_id)
        if parsed_user is None:
            return _seat_error_response("missing user id")
        result = instance.occupy_seat(
            seat=seat,
            user_id=parsed_user,
        )
        if isinstance(result, Rejected):
            return _seat_error_response(result.reason)
        return _seat_ok_response()

    async def vacate_seat(
        game_id: str,
        seat_id: str,
        user_id: str | None = None,
    ) -> JSONResponse:
        instance = state.registry.get(game_id)
        if instance is None:
            return _seat_error_response("game not found")
        seat = seat_from_id(seat_id)
        if seat is None:
            return _seat_error_response("invalid seat")
        parsed_user = UserId.parse(user_id)
        if parsed_user is None:
            return _seat_error_response("missing user id")
        result = instance.vacate_seat(
            seat=seat,
            user_id=parsed_user,
        )
        if isinstance(result, Rejected):
            return _seat_error_response(result.reason)
        return _seat_ok_response()

    async def fill_bot_seats(
        game_id: str,
        policy: str | None = None,
        user_id: str | None = None,
    ) -> JSONResponse:
        instance = state.registry.get(game_id)
        if instance is None:
            return _seat_error_response("game not found")
        policy_name = _bot_policy_response(policy)
        if isinstance(policy_name, JSONResponse):
            return policy_name
        parsed_user = UserId.parse(user_id)
        if parsed_user is None:
            return _seat_error_response("missing user id")
        result = instance.fill_bots(
            policy=policy_name,
            user_id=parsed_user,
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
        instance = state.registry.get(game_id)
        if instance is None:
            await websocket.close(code=4404, reason="game not found")
            return
        seat = seat_from_id(seat_id)
        if seat is None:
            await websocket.close(code=4410, reason="invalid seat")
            return
        parsed_user = UserId.parse(user_id)
        if parsed_user is None:
            await websocket.close(code=4410, reason="missing user id")
            return
        await handle_game_connection(
            websocket,
            instance,
            seat=seat,
            user_id=parsed_user,
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
    instance: GameInstance | None,
    game_id: str,
    requester: UserId | None,
) -> ListedGameResponse:
    room_seats = _room_seats(instance, requester)
    user_seats: list[SeatId] = []
    for room_seat in room_seats:
        player = room_seat["player"]
        if player is not None and player["kind"] == "human":
            user_seats.append(room_seat["seat"])
    return {
        "game_id": game_id,
        "user_count": len(user_seats),
        "capacity": _SEAT_CAPACITY,
        "user_seats": user_seats,
        "seats": room_seats,
    }


def _room_seats(
    instance: GameInstance | None,
    requester: UserId | None,
) -> list[SeatResponse]:
    if instance is None:
        return []
    return [
        _seat_response(seat)
        for seat in instance.seats(user_id=requester)
    ]


def _seat_response(status: SeatStatus) -> SeatResponse:
    return {
        "seat": seat_id(status.seat),
        "player": status.player,
        "ready": status.ready,
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
        case "invalid seat" | "missing user id" | "invalid bot policy":
            return 400
        case _:
            return 409


def _bot_policy_response(
    policy: str | None,
) -> BotPolicyName | JSONResponse:
    policy_name = bot_policy_name_from_str(policy)
    if policy_name is None:
        return _seat_error_response("invalid bot policy")
    return policy_name
