"""Tractor game server -- REST + WebSocket API layer.

Wires Game, Player, and GameRegistry together. Handles HTTP/WS protocol
concerns only. Contains no game logic.

Server maintains its own human_players mapping (game_id → HumanPlayer)
to route WS connections without hardcoding player indices.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, Response

from server.game import Game
from server.game_registry import GameRegistry
from server.player import AutoPlayer, HumanPlayer, Player

logger = logging.getLogger(__name__)

registry = GameRegistry()
human_players: dict[str, HumanPlayer] = {}


async def _cleanup_loop():
    """Periodic cleanup of expired games."""
    while True:
        await asyncio.sleep(300)
        # Clean up human_players entries for expired games
        expired_ids = set(human_players.keys()) - set(g["game_id"] for g in registry.list_games())
        for gid in expired_ids:
            human = human_players.pop(gid, None)
            if human is not None and human.is_connected():
                await human.close_ws()
        registry.cleanup_expired(max_age_seconds=3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)

# ---- REST Endpoints ----


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/game", status_code=201)
async def create_game():
    """Create a new game with 3 AI players and 1 human player.

    The game starts in WAITING phase. All players must confirm (next_round)
    before the game begins. AutoPlayers confirm via run(); the human player
    confirms when they connect via WebSocket.
    """
    auto_players: list[AutoPlayer] = [AutoPlayer(i) for i in range(3)]
    human = HumanPlayer(3)
    players: list[Player] = [*auto_players, human]
    game = Game(players=players)
    game_id = registry.create(game)
    human_players[game_id] = human

    def on_game_over(g: Game) -> None:
        registry.delete(game_id)
        human_players.pop(game_id, None)

    game.set_on_game_over(on_game_over)

    # Start each AutoPlayer independently — they send next_round to get
    # into the game loop. HumanPlayer starts when their WS connects.
    for ap in auto_players:
        asyncio.create_task(ap.run(game))

    return {"game_id": game_id}


@app.get("/api/game")
async def list_games():
    return {"games": registry.list_games()}


@app.delete("/api/game/{game_id}")
async def delete_game(game_id: str):
    game = registry.get(game_id)
    human = human_players.pop(game_id, None)
    if human is not None and human.is_connected():
        # Push final state before closing so client receives it
        if game is not None:
            await human.on_state(game, seq=game.current_seq)
        await human.close_ws()
    registry.delete(game_id)
    return {"ok": True}


# ---- WebSocket Endpoint ----


@app.websocket("/game/{game_id}")
async def websocket_game(websocket: WebSocket, game_id: str):
    """WebSocket endpoint for game interaction.

    Delegates entirely to HumanPlayer.handle_connection() which
    manages the full WS lifecycle (accept, seq validation, action
    parsing, game.act, cleanup).
    """
    game = registry.get(game_id)
    if game is None:
        await websocket.close(code=4404, reason="game not found")
        return

    human = human_players.get(game_id)
    if human is None:
        await websocket.close(code=4403, reason="no human player slot")
        return

    await human.handle_connection(websocket, game)


# ---- Static files ----

static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))


@app.get("/")
async def index():
    html_path = os.path.join(static_dir, "index.html")
    if os.path.isfile(html_path):
        return FileResponse(html_path)
    return Response(status_code=404, content="Frontend not built. Run: deno task build")


@app.get("/{path:path}")
async def serve_static(path: str):
    """Serve static files from static/ directory. Falls back to index.html for unknown paths."""
    file_path = os.path.normpath(os.path.join(static_dir, path))
    if not file_path.startswith(static_dir + os.sep) and file_path != static_dir:
        return Response(status_code=403, content="Forbidden")
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    # Fallback to index.html for SPA routing
    html_path = os.path.join(static_dir, "index.html")
    if os.path.isfile(html_path):
        return FileResponse(html_path)
    return Response(status_code=404, content="Not found")
