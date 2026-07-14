"""Independent lifecycle, metrics, and log WebSocket adapters."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import Annotated

from anyio import to_thread
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect

from server.foundation import result as _result
from server.training_artifacts import query_checkpoint_invalidation
from server.training_control.process_control import ProcessEnvelope
from server.training_events.queries import query_training_log_tail
from server.training_metrics.queries import (
    query_metrics_cursor,
    query_training_metrics,
)
from server.web.state import ServerState

_POLL_SECONDS = 1.0
_TAIL_BATCH_SIZE = 500

type _StoreId = Annotated[str, Query(pattern=r"^[0-9a-f]{32}$")]


def register_training_stream_route(
    app: FastAPI, state: ServerState
) -> None:
    async def training_logs(
        websocket: WebSocket,
        run_dir: Path | None = None,
        after_sequence: Annotated[int, Query(ge=0)] = 0,
        store_id: _StoreId | None = None,
    ) -> None:
        await websocket.accept()
        canonical = state.training_control_config.resolve_run_dir(
            run_dir
        )
        cursor = after_sequence
        disconnect = asyncio.create_task(
            _wait_for_disconnect(websocket)
        )
        try:
            while True:
                result = await to_thread.run_sync(
                    partial(
                        query_training_log_tail,
                        canonical,
                        after_sequence=cursor,
                        limit=_TAIL_BATCH_SIZE,
                    )
                )
                if isinstance(result, _result.Rejected):
                    await _reject_stream(
                        websocket, disconnect, result.reason
                    )
                    return
                tail = result.value
                if store_id is not None and tail.store_id != store_id:
                    await websocket.send_json(
                        {
                            "type": "replacement",
                            "store_id": tail.store_id,
                        }
                    )
                    return
                for record in tail.events:
                    cursor = record.sequence
                    await websocket.send_json(
                        {
                            "type": "event",
                            "sequence": record.sequence,
                            "event": record.event,
                        }
                    )
                if len(tail.events) == _TAIL_BATCH_SIZE:
                    continue
                if await _sleep_or_disconnected(disconnect):
                    return
        except WebSocketDisconnect:
            return
        finally:
            disconnect.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect

    async def training_metrics(
        websocket: WebSocket,
        run_dir: Path | None = None,
        update_limit: Annotated[int, Query(ge=1, le=5000)] = 500,
        series_points: Annotated[int, Query(ge=1, le=1000)] = 500,
    ) -> None:
        await websocket.accept()
        canonical = state.training_control_config.resolve_run_dir(
            run_dir
        )
        observed_store: str | None = None
        observed_sequence = -1
        disconnect = asyncio.create_task(
            _wait_for_disconnect(websocket)
        )
        try:
            while True:
                cursor_result = await to_thread.run_sync(
                    partial(query_metrics_cursor, canonical)
                )
                if isinstance(cursor_result, _result.Rejected):
                    await _reject_stream(
                        websocket, disconnect, cursor_result.reason
                    )
                    return
                cursor = cursor_result.value
                if (
                    cursor.store_id == observed_store
                    and cursor.through_sequence == observed_sequence
                ):
                    if await _sleep_or_disconnected(disconnect):
                        return
                    continue
                snapshot_result = await to_thread.run_sync(
                    partial(
                        query_training_metrics,
                        canonical,
                        update_limit=update_limit,
                        series_points=series_points,
                    )
                )
                if isinstance(snapshot_result, _result.Rejected):
                    await _reject_stream(
                        websocket, disconnect, snapshot_result.reason
                    )
                    return
                snapshot = snapshot_result.value
                observed_store = snapshot.store_id
                observed_sequence = snapshot.through_sequence
                await websocket.send_json(
                    snapshot.model_dump(mode="json")
                )
                if await _sleep_or_disconnected(disconnect):
                    return
        except WebSocketDisconnect:
            return
        finally:
            disconnect.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect

    async def training_process(
        websocket: WebSocket,
        run_dir: Path | None = None,
        after_revision: Annotated[int, Query(ge=-1)] = -1,
    ) -> None:
        await websocket.accept()
        canonical = state.training_control_config.resolve_run_dir(
            run_dir
        )
        snapshots = state.training_process_control.watch(
            canonical, after_revision=after_revision
        )
        disconnect = asyncio.create_task(
            _wait_for_disconnect(websocket)
        )
        try:
            while True:
                snapshot_task = asyncio.create_task(
                    _next_snapshot(snapshots)
                )
                completed, _pending = await asyncio.wait(
                    (snapshot_task, disconnect),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if disconnect in completed:
                    snapshot_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await snapshot_task
                    return
                snapshot_result = await snapshot_task
                if isinstance(snapshot_result, _result.Rejected):
                    await _reject_stream(
                        websocket, disconnect, snapshot_result.reason
                    )
                    return
                await websocket.send_json(
                    snapshot_result.value.model_dump(mode="json")
                )
        except WebSocketDisconnect:
            return
        finally:
            await snapshots.aclose()
            disconnect.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect

    async def training_checkpoints(
        websocket: WebSocket,
        run_dir: Path | None = None,
        store_id: _StoreId | None = None,
    ) -> None:
        await websocket.accept()
        canonical = state.training_control_config.resolve_run_dir(
            run_dir
        )
        observed_store = store_id
        observed_sequence = -1
        disconnect = asyncio.create_task(
            _wait_for_disconnect(websocket)
        )
        try:
            while True:
                result = await to_thread.run_sync(
                    partial(query_checkpoint_invalidation, canonical)
                )
                if isinstance(result, _result.Rejected):
                    await _reject_stream(
                        websocket, disconnect, result.reason
                    )
                    return
                invalidation = result.value
                if (
                    observed_store is not None
                    and invalidation.store_id != observed_store
                ):
                    await websocket.send_json(
                        {
                            "type": "replacement",
                            "store_id": invalidation.store_id,
                            "through_sequence": (
                                invalidation.through_sequence
                            ),
                        }
                    )
                    return
                observed_store = invalidation.store_id
                if invalidation.through_sequence != observed_sequence:
                    observed_sequence = invalidation.through_sequence
                    await websocket.send_json(
                        {
                            "type": "invalidation",
                            "store_id": invalidation.store_id,
                            "through_sequence": observed_sequence,
                        }
                    )
                if await _sleep_or_disconnected(disconnect):
                    return
        except WebSocketDisconnect:
            return
        finally:
            disconnect.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect

    app.add_api_websocket_route("/ws/training/logs", training_logs)
    app.add_api_websocket_route(
        "/ws/training/metrics", training_metrics
    )
    app.add_api_websocket_route(
        "/ws/training/process", training_process
    )
    app.add_api_websocket_route(
        "/ws/training/checkpoints", training_checkpoints
    )


async def _next_snapshot(
    snapshots: AsyncGenerator[
        _result.Ok[ProcessEnvelope] | _result.Rejected, None
    ],
) -> _result.Ok[ProcessEnvelope] | _result.Rejected:
    return await anext(snapshots)


async def _sleep_or_disconnected(
    disconnect: asyncio.Task[None],
) -> bool:
    sleep = asyncio.create_task(asyncio.sleep(_POLL_SECONDS))
    completed, _pending = await asyncio.wait(
        (sleep, disconnect), return_when=asyncio.FIRST_COMPLETED
    )
    if disconnect in completed:
        sleep.cancel()
        with suppress(asyncio.CancelledError):
            await sleep
        return True
    return False


async def _wait_for_disconnect(websocket: WebSocket) -> None:
    while True:
        try:
            await websocket.receive_text()
        except WebSocketDisconnect:
            return


async def _reject_stream(
    websocket: WebSocket,
    disconnect: asyncio.Task[None],
    reason: str,
) -> None:
    await websocket.send_json({"type": "rejected", "error": reason})
    await disconnect
