"""Game REST and WebSocket routes."""

from __future__ import annotations

from fastapi import FastAPI, WebSocket

from server.app_state import ServerState
from server.player_factory import create_game_with_default_players


def register_game_routes(app: FastAPI, state: ServerState) -> None:
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    async def create_game() -> dict[str, str]:
        game, human = create_game_with_default_players()
        game_id = state.registry.create(game)
        state.human_players[game_id] = human
        return {"game_id": game_id}

    async def list_games() -> dict[str, object]:
        return {"games": state.registry.list_games()}

    async def delete_game(game_id: str) -> dict[str, bool]:
        human = state.human_players.pop(game_id, None)
        if human is not None and human.is_connected():
            await human.close_ws()
        state.registry.delete(game_id)
        return {"ok": True}

    async def websocket_game(
        websocket: WebSocket, game_id: str
    ) -> None:
        game = state.registry.get(game_id)
        if game is None:
            await websocket.close(code=4404, reason="game not found")
            return

        human = state.human_players.get(game_id)
        if human is None:
            await websocket.close(
                code=4403, reason="no human player slot"
            )
            return

        await human.handle_connection(websocket, game)

    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route(
        "/api/game", create_game, methods=["POST"], status_code=201
    )
    app.add_api_route("/api/game", list_games, methods=["GET"])
    app.add_api_route(
        "/api/game/{game_id}", delete_game, methods=["DELETE"]
    )
    app.add_api_websocket_route("/game/{game_id}", websocket_game)
