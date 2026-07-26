"""LLM policy transcript debug routes."""

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

from server.game import seat_from_id
from server.game_bots.llm import LLMTranscript, TranscriptRecordDict
from server.web.game_composition import GameInstance
from server.web.state import ServerState


def register_llm_debug_routes(
    app: FastAPI, state: ServerState, static_dir: str
) -> None:
    async def llm_debug_page(game_id: str) -> Response:
        _game_or_404(state, game_id)
        html_path = os.path.join(static_dir, "llm-debug", "index.html")
        if os.path.isfile(html_path):
            return FileResponse(html_path)
        return Response(
            status_code=404,
            content="Debug frontend not built. Run: deno task build",
        )

    async def llm_debug_stream(
        websocket: WebSocket,
        game_id: str,
        seat: str | None = None,
    ) -> None:
        game = state.registry.get(game_id)
        if game is None:
            await websocket.close(code=4404, reason="game not found")
            return
        transcript = _transcript_at(game, seat)
        if transcript is None:
            await websocket.close(
                code=4404, reason="llm seat not found"
            )
            return

        await websocket.accept()
        queue = transcript.subscribe()
        last_sent_event_id = 0
        try:
            for message in transcript.stream_dicts():
                await _send_llm_debug_message(websocket, message)
                last_sent_event_id = message["event_id"]
            await _stream_live_llm_debug_messages(
                websocket, queue, last_sent_event_id
            )
        except WebSocketDisconnect:
            pass
        finally:
            transcript.unsubscribe(queue)

    app.add_api_route(
        "/debug/llm/{game_id}", llm_debug_page, methods=["GET"]
    )
    app.add_api_websocket_route(
        "/ws/debug/llm/{game_id}", llm_debug_stream
    )


def _game_or_404(state: ServerState, game_id: str) -> GameInstance:
    game = state.registry.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="game not found")
    return game


def _transcript_at(
    game: GameInstance, seat_id: str | None
) -> LLMTranscript | None:
    if seat_id is None:
        return None
    seat = seat_from_id(seat_id)
    if seat is None:
        return None
    return game.llm_transcripts.at(seat)


async def _send_llm_debug_message(
    websocket: WebSocket, message: TranscriptRecordDict
) -> None:
    await websocket.send_json(message)


async def _stream_live_llm_debug_messages(
    websocket: WebSocket,
    queue: asyncio.Queue[TranscriptRecordDict],
    last_sent_event_id: int,
) -> None:
    queue_task = asyncio.create_task(queue.get())
    disconnect_task = asyncio.create_task(
        _wait_llm_debug_disconnect(websocket)
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
                    await _send_llm_debug_message(websocket, message)
                    last_sent_event_id = message["event_id"]
                queue_task = asyncio.create_task(queue.get())
    finally:
        for task in (queue_task, disconnect_task):
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task


async def _wait_llm_debug_disconnect(websocket: WebSocket) -> None:
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
    except WebSocketDisconnect:
        return
