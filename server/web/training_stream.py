"""WebSocket adapter for direct structured training log reads."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect

from server.foundation import result as _result
from server.training_control.log_feed import TrainingLogFeed
from server.web.state import ServerState


def register_training_stream_route(
    app: FastAPI, state: ServerState
) -> None:
    """Register the direct SQLite structured-log route."""

    async def training_stream(
        websocket: WebSocket,
        run_dir: Path | None = None,
        window: Annotated[int, Query(ge=1)] = 5000,
        event: Annotated[list[str] | None, Query()] = None,
        session_id: str | None = None,
    ) -> None:
        await websocket.accept()
        feed = TrainingLogFeed(
            run_dir=state.training_control_config.resolve_run_dir(
                run_dir
            ),
            window=window,
            event_types=tuple(event or ()),
            session_id=session_id,
        )
        disconnect_task = asyncio.create_task(
            _wait_for_disconnect(websocket)
        )
        try:
            while True:
                read_task = asyncio.create_task(feed.read())
                completed, _pending = await asyncio.wait(
                    (read_task, disconnect_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if disconnect_task in completed:
                    read_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await read_task
                    return
                message = await read_task
                if isinstance(message, _result.Rejected):
                    await websocket.send_json(
                        {"type": "error", "message": message.reason}
                    )
                    return
                batch = message.value
                if batch.reset:
                    await websocket.send_json(
                        {"type": "reset", "window": batch.window}
                    )
                for record in batch.records:
                    await websocket.send_json(
                        {
                            "type": "event",
                            "sequence": record.sequence,
                            "event": record.event,
                        }
                    )
                if not batch.full:
                    sleep_task = asyncio.create_task(asyncio.sleep(1.0))
                    completed, _pending = await asyncio.wait(
                        (sleep_task, disconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if disconnect_task in completed:
                        sleep_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await sleep_task
                        return
        except WebSocketDisconnect:
            return
        finally:
            disconnect_task.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect_task

    app.add_api_websocket_route("/ws/training/logs", training_stream)


async def _wait_for_disconnect(websocket: WebSocket) -> None:
    while True:
        try:
            await websocket.receive_text()
        except WebSocketDisconnect:
            return
