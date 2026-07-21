"""Training process-state snapshot SSE endpoint."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import suppress
from pathlib import Path

from fastapi import FastAPI
from starlette.responses import StreamingResponse

from server.foundation import result as _result
from server.training_control.process_inspection import ProcessState
from server.web.state import ServerState
from server.web.training_events.wire import (
    KEEP_ALIVE,
    RETRY,
    ServerEvent,
    event_response,
    rejected_event,
)

_HEARTBEAT_SECONDS = 15.0


def register_process_route(app: FastAPI, state: ServerState) -> None:
    async def training_process(
        run_dir: Path | None = None,
    ) -> StreamingResponse:
        canonical = state.training_control_config.resolve_run_dir(
            run_dir
        )
        return event_response(_process_events(state, canonical))

    app.add_api_route(
        "/api/training/events/process",
        training_process,
        methods=["GET"],
    )


async def _process_events(
    state: ServerState, run_dir: Path
) -> AsyncGenerator[bytes, None]:
    yield RETRY
    snapshots = state.training_process_control.watch(run_dir)
    snapshot_task: asyncio.Task[
        _result.Ok[ProcessState] | _result.Rejected
    ] = asyncio.create_task(anext(snapshots))
    heartbeat: asyncio.Task[None] = asyncio.create_task(
        asyncio.sleep(_HEARTBEAT_SECONDS)
    )
    try:
        while True:
            completed, _pending = await asyncio.wait(
                (snapshot_task, heartbeat),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if snapshot_task in completed:
                result = await snapshot_task
                if isinstance(result, _result.Rejected):
                    yield rejected_event(result.reason).encode()
                    return
                yield ServerEvent(
                    name="process", data=result.value.model_dump_json()
                ).encode()
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat
                heartbeat = asyncio.create_task(
                    asyncio.sleep(_HEARTBEAT_SECONDS)
                )
                snapshot_task = asyncio.create_task(anext(snapshots))
                continue
            await heartbeat
            yield KEEP_ALIVE
            heartbeat = asyncio.create_task(
                asyncio.sleep(_HEARTBEAT_SECONDS)
            )
    finally:
        snapshot_task.cancel()
        with suppress(asyncio.CancelledError):
            await snapshot_task
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat
        await snapshots.aclose()
