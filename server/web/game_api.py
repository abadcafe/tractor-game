"""Game REST and WebSocket routes."""

from __future__ import annotations

from typing import TypedDict

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

from server.foundation.result import Rejected
from server.game.room.bot_factory import (
    BotKind,
    bot_kind_from_env,
    bot_kind_from_str,
)
from server.game.room.game_room import GameRoom, RoomPlayer
from server.game.room.player_factory import create_game_room
from server.web.state import ServerState

_PLAYER_CAPACITY = 4


class PlayerResponse(TypedDict):
    index: int
    occupied: bool
    connected: bool
    kind: str
    mine: bool
    ready: bool


class ListedGameResponse(TypedDict):
    game_id: str
    user_count: int
    capacity: int
    user_players: list[int]
    players: list[PlayerResponse]


class PlayerOperationResponse(TypedDict):
    ok: bool


def register_game_routes(app: FastAPI, state: ServerState) -> None:
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    async def create_game() -> dict[str, str]:
        game_id = state.registry.create(create_game_room())
        return {"game_id": game_id}

    async def create_auto_game() -> dict[str, str]:
        room = create_game_room()
        result = await room.fill_empty_players_for_setup(
            kind=bot_kind_from_env(),
            preserve_players={2},
        )
        if isinstance(result, Rejected):
            raise RuntimeError(result.reason)
        game_id = state.registry.create(room)
        return {"game_id": game_id}

    async def list_games(
        user_id: str | None = None,
    ) -> dict[str, list[ListedGameResponse]]:
        return {
            "games": [
                _listed_game_response(state, game["game_id"], user_id)
                for game in state.registry.list_games()
            ]
        }

    async def delete_game(game_id: str) -> dict[str, bool]:
        room = state.registry.get(game_id)
        if room is not None:
            await room.close_all()
        state.registry.delete(game_id)
        return {"ok": True}

    async def attach_player(
        game_id: str,
        player: int,
        user_id: str | None = None,
    ) -> JSONResponse:
        room = state.registry.get(game_id)
        if room is None:
            return _player_error_response("game not found")
        if user_id is None:
            return _player_error_response("missing user id")
        result = await room.attach_player(
            player=player, user_id=user_id
        )
        if isinstance(result, Rejected):
            return _player_error_response(result.reason)
        return _player_ok_response()

    async def detach_player(
        game_id: str,
        player: int,
        user_id: str | None = None,
    ) -> JSONResponse:
        room = state.registry.get(game_id)
        if room is None:
            return _player_error_response("game not found")
        if user_id is None:
            return _player_error_response("missing user id")
        result = await room.detach_player(
            player=player, user_id=user_id
        )
        if isinstance(result, Rejected):
            return _player_error_response(result.reason)
        return _player_ok_response()

    async def fill_bot_players(
        game_id: str,
        kind: str | None = None,
        user_id: str | None = None,
    ) -> JSONResponse:
        room = state.registry.get(game_id)
        if room is None:
            return _player_error_response("game not found")
        bot_kind = _bot_kind_response(kind)
        if isinstance(bot_kind, JSONResponse):
            return bot_kind
        if user_id is None:
            return _player_error_response("missing user id")
        result = await room.fill_bot_players(
            kind=bot_kind,
            user_id=user_id,
        )
        if isinstance(result, Rejected):
            return _player_error_response(result.reason)
        return _player_ok_response()

    async def websocket_game(
        websocket: WebSocket,
        game_id: str,
        player: int,
        user_id: str | None = None,
    ) -> None:
        room = state.registry.get(game_id)
        if room is None:
            await websocket.close(code=4404, reason="game not found")
            return
        if user_id is None:
            await websocket.close(code=4410, reason="missing user id")
            return

        await room.connect_player(
            websocket,
            player=player,
            user_id=user_id,
        )

    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route(
        "/api/game", create_game, methods=["POST"], status_code=201
    )
    app.add_api_route(
        "/api/game/auto",
        create_auto_game,
        methods=["POST"],
        status_code=201,
    )
    app.add_api_route("/api/game", list_games, methods=["GET"])
    app.add_api_route(
        "/api/game/{game_id}", delete_game, methods=["DELETE"]
    )
    app.add_api_route(
        "/api/game/{game_id}/player/{player}",
        attach_player,
        methods=["POST"],
    )
    app.add_api_route(
        "/api/game/{game_id}/player/{player}",
        detach_player,
        methods=["DELETE"],
    )
    app.add_api_route(
        "/api/game/{game_id}/bots",
        fill_bot_players,
        methods=["POST"],
    )
    app.add_api_websocket_route(
        "/game/{game_id}/player/{player}", websocket_game
    )


def _listed_game_response(
    state: ServerState, game_id: str, user_id: str | None
) -> ListedGameResponse:
    room = state.registry.get(game_id)
    players = _room_players(room, user_id)
    user_players = [
        player["index"]
        for player in players
        if player["kind"] == "user"
    ]
    return {
        "game_id": game_id,
        "user_count": len(user_players),
        "capacity": _PLAYER_CAPACITY,
        "user_players": user_players,
        "players": players,
    }


def _room_players(
    room: GameRoom | None, user_id: str | None
) -> list[PlayerResponse]:
    if room is None:
        return []
    return [
        _player_response(player)
        for player in room.players(user_id=user_id)
    ]


def _player_response(player: RoomPlayer) -> PlayerResponse:
    return {
        "index": player.index,
        "occupied": player.occupied,
        "connected": player.connected,
        "kind": player.kind,
        "mine": player.mine,
        "ready": player.ready,
    }


def _player_ok_response() -> JSONResponse:
    content: PlayerOperationResponse = {"ok": True}
    return JSONResponse(content)


def _player_error_response(reason: str) -> JSONResponse:
    return JSONResponse(
        {"ok": False, "error": reason},
        status_code=_player_error_status(reason),
    )


def _player_error_status(reason: str) -> int:
    match reason:
        case "game not found":
            return 404
        case "invalid player" | "missing user id" | "invalid bot kind":
            return 400
        case _:
            return 409


def _bot_kind_response(kind: str | None) -> BotKind | JSONResponse:
    bot_kind = bot_kind_from_str(kind)
    if bot_kind is None:
        return _player_error_response("invalid bot kind")
    return bot_kind
