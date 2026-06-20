"""Tractor game server -- REST + WebSocket API layer.

Wires Game, Player, and GameRegistry together. Handles HTTP/WS protocol
concerns only. Contains no game logic.

Server maintains its own human_players mapping (game_id → HumanPlayer)
to route WS connections without hardcoding player indices.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

from server.game import Game
from server.game_registry import GameRegistry
from server.player import AIPlayer, AutoPlayer, HumanPlayer, Player
from server.player.ai.transcript import TranscriptRecordDict

logger = logging.getLogger(__name__)

registry = GameRegistry()
human_players: dict[str, HumanPlayer] = {}
type BotPlayer = AIPlayer | AutoPlayer
_SERVER_LOG_HANDLER_NAME = "tractor-server-stderr"


def _configure_server_logging() -> None:
    """Route project loggers to stderr under uvicorn.

    Uvicorn's default logging config only installs handlers for uvicorn.*
    loggers. Without this handler, INFO logs from server.* modules are
    created but not visible in the terminal.
    """
    server_logger = logging.getLogger("server")
    server_logger.setLevel(logging.INFO)
    if not _has_named_handler(server_logger, _SERVER_LOG_HANDLER_NAME):
        handler = logging.StreamHandler()
        handler.set_name(_SERVER_LOG_HANDLER_NAME)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        ))
        server_logger.addHandler(handler)
    server_logger.propagate = False


def _has_named_handler(target_logger: logging.Logger, name: str) -> bool:
    return any(handler.get_name() == name for handler in target_logger.handlers)


_configure_server_logging()


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
    """Create a new game with 3 bot players and 1 human player.

    The game starts in WAITING phase. All players must confirm (next_round)
    before the game begins. Bot players first request state via seq=0; the
    human client does the same after opening its WebSocket.
    """
    bot_players: list[BotPlayer] = [_create_bot_player(i) for i in range(3)]
    human = HumanPlayer(3)
    players: list[Player] = [*bot_players, human]
    game = Game(players=players)
    game_id = registry.create(game)
    human_players[game_id] = human

    # Start each bot player independently. They send seq=0 to request the
    # current state, then act only from received StateMessage pushes.
    for player in bot_players:
        asyncio.create_task(player.run(game))

    return {"game_id": game_id}


def _create_bot_player(index: int) -> BotPlayer:
    kind = os.environ.get("TRACTOR_BOT_PLAYER", "auto").strip().lower()
    if kind == "ai":
        return AIPlayer(index)
    return AutoPlayer(index)


@app.get("/api/game")
async def list_games():
    return {"games": registry.list_games()}


@app.delete("/api/game/{game_id}")
async def delete_game(game_id: str):
    human = human_players.pop(game_id, None)
    if human is not None and human.is_connected():
        await human.close_ws()
    registry.delete(game_id)
    return {"ok": True}


@app.get("/debug/ai/{game_id}")
async def ai_debug_page(game_id: str) -> Response:
    _game_or_404(game_id)
    html_path = os.path.join(static_dir, "debug-ai.html")
    if os.path.isfile(html_path):
        return FileResponse(html_path)
    return Response(status_code=404, content="Debug frontend not built. Run: deno task build")


@app.websocket("/ws/debug/ai/{game_id}")
async def ai_debug_stream(websocket: WebSocket, game_id: str, player: int | None = None) -> None:
    game = registry.get(game_id)
    if game is None:
        await websocket.close(code=4404, reason="game not found")
        return
    ai_player = _ai_player_at(game, player)
    if ai_player is None:
        await websocket.close(code=4404, reason="ai player not found")
        return

    await websocket.accept()
    queue = ai_player.subscribe_transcript()
    last_sent_event_id = 0
    try:
        for message in ai_player.transcript_stream():
            await _send_ai_debug_message(websocket, message)
            last_sent_event_id = message["event_id"]
        await _stream_live_ai_debug_messages(websocket, queue, last_sent_event_id)
    except WebSocketDisconnect:
        pass
    finally:
        ai_player.unsubscribe_transcript(queue)


def _game_or_404(game_id: str) -> Game:
    game = registry.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="game not found")
    assert isinstance(game, Game)
    return game


def _ai_player_at(game: Game, player: int | None) -> AIPlayer | None:
    if player is None or player < 0 or player >= 4:
        return None
    candidate = game.get_player(player)
    if isinstance(candidate, AIPlayer):
        return candidate
    return None


async def _send_ai_debug_message(websocket: WebSocket, message: TranscriptRecordDict) -> None:
    await websocket.send_json(message)


async def _stream_live_ai_debug_messages(
    websocket: WebSocket,
    queue: asyncio.Queue[TranscriptRecordDict],
    last_sent_event_id: int,
) -> None:
    queue_task = asyncio.create_task(queue.get())
    disconnect_task = asyncio.create_task(_wait_ai_debug_disconnect(websocket))
    try:
        while True:
            done, _pending = await asyncio.wait(
                {queue_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done:
                return
            if queue_task in done:
                message = queue_task.result()
                if message["event_id"] > last_sent_event_id:
                    await _send_ai_debug_message(websocket, message)
                    last_sent_event_id = message["event_id"]
                queue_task = asyncio.create_task(queue.get())
    finally:
        for task in (queue_task, disconnect_task):
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task


async def _wait_ai_debug_disconnect(websocket: WebSocket) -> None:
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
    except WebSocketDisconnect:
        return



# ---- WebSocket Endpoint ----


@app.websocket("/game/{game_id}")
async def websocket_game(websocket: WebSocket, game_id: str):
    """WebSocket endpoint for game interaction.

    Delegates entirely to HumanPlayer.handle_connection() which
    manages the full WS lifecycle and forwards PlayerMessage envelopes
    to Game.receive().
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
    if path.startswith("api/") or path.startswith("ws/"):
        return Response(status_code=404, content="Not found")
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
