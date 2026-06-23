"""AI transcript debug routes."""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress

from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, Response

from server.app_state import ServerState
from server.game import Game
from server.player import AIPlayer
from server.player.ai.transcript import TranscriptRecordDict


def register_ai_debug_routes(
    app: FastAPI, state: ServerState, static_dir: str
) -> None:
    async def ai_debug_page(game_id: str) -> Response:
        _game_or_404(state, game_id)
        html_path = os.path.join(static_dir, "debug-ai.html")
        if os.path.isfile(html_path):
            return FileResponse(html_path)
        return Response(
            status_code=404,
            content="Debug frontend not built. Run: deno task build",
        )

    async def ai_debug_stream(
        websocket: WebSocket, game_id: str, player: int | None = None
    ) -> None:
        game = state.registry.get(game_id)
        if game is None:
            await websocket.close(code=4404, reason="game not found")
            return
        ai_player = _ai_player_at(game, player)
        if ai_player is None:
            await websocket.close(
                code=4404, reason="ai player not found"
            )
            return

        await websocket.accept()
        queue = ai_player.subscribe_transcript()
        last_sent_event_id = 0
        try:
            for message in ai_player.transcript_stream():
                await _send_ai_debug_message(websocket, message)
                last_sent_event_id = message["event_id"]
            await _stream_live_ai_debug_messages(
                websocket, queue, last_sent_event_id
            )
        except WebSocketDisconnect:
            pass
        finally:
            ai_player.unsubscribe_transcript(queue)

    app.add_api_route(
        "/debug/ai/{game_id}", ai_debug_page, methods=["GET"]
    )
    app.add_api_websocket_route(
        "/ws/debug/ai/{game_id}", ai_debug_stream
    )


def _game_or_404(state: ServerState, game_id: str) -> Game:
    game = state.registry.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="game not found")
    return game


def _ai_player_at(game: Game, player: int | None) -> AIPlayer | None:
    if player is None or player < 0 or player >= 4:
        return None
    candidate = game.get_player(player)
    if isinstance(candidate, AIPlayer):
        return candidate
    return None


async def _send_ai_debug_message(
    websocket: WebSocket, message: TranscriptRecordDict
) -> None:
    await websocket.send_json(message)


async def _stream_live_ai_debug_messages(
    websocket: WebSocket,
    queue: asyncio.Queue[TranscriptRecordDict],
    last_sent_event_id: int,
) -> None:
    queue_task = asyncio.create_task(queue.get())
    disconnect_task = asyncio.create_task(
        _wait_ai_debug_disconnect(websocket)
    )
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
