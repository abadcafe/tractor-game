"""Complete training-metrics snapshot SSE endpoint."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from functools import partial
from pathlib import Path
from typing import Annotated

from anyio import to_thread
from fastapi import FastAPI, Query
from starlette.responses import StreamingResponse

from server.foundation import result as _result
from server.training_metrics.queries import (
    query_metrics_cursor,
    query_training_metrics,
)
from server.web.state import ServerState
from server.web.training_events.wire import (
    KEEP_ALIVE,
    RETRY,
    ServerEvent,
    event_response,
    rejected_event,
)

_POLL_SECONDS = 1.0
_HEARTBEAT_SECONDS = 15.0


def register_metrics_route(app: FastAPI, state: ServerState) -> None:
    async def training_metrics(
        run_dir: Path | None = None,
        update_limit: Annotated[int, Query(ge=1, le=5000)] = 500,
        series_points: Annotated[int, Query(ge=1, le=1000)] = 500,
    ) -> StreamingResponse:
        canonical = state.training_control_config.resolve_run_dir(
            run_dir
        )
        return event_response(
            _metrics_events(
                canonical,
                update_limit=update_limit,
                series_points=series_points,
            )
        )

    app.add_api_route(
        "/api/training/events/metrics",
        training_metrics,
        methods=["GET"],
    )


async def _metrics_events(
    run_dir: Path,
    *,
    update_limit: int,
    series_points: int,
) -> AsyncGenerator[bytes, None]:
    yield RETRY
    observed_store: str | None = None
    observed_sequence = -1
    last_sent = time.monotonic()
    while True:
        cursor_result = await to_thread.run_sync(
            partial(query_metrics_cursor, run_dir)
        )
        if isinstance(cursor_result, _result.Rejected):
            yield rejected_event(cursor_result.reason).encode()
            return
        cursor = cursor_result.value
        if (
            cursor.store_id != observed_store
            or cursor.through_sequence != observed_sequence
        ):
            snapshot_result = await to_thread.run_sync(
                partial(
                    query_training_metrics,
                    run_dir,
                    update_limit=update_limit,
                    series_points=series_points,
                )
            )
            if isinstance(snapshot_result, _result.Rejected):
                yield rejected_event(snapshot_result.reason).encode()
                return
            snapshot = snapshot_result.value
            observed_store = snapshot.store_id
            observed_sequence = snapshot.through_sequence
            yield ServerEvent(
                name="metrics", data=snapshot.model_dump_json()
            ).encode()
            last_sent = time.monotonic()
        await asyncio.sleep(_POLL_SECONDS)
        if time.monotonic() - last_sent >= _HEARTBEAT_SECONDS:
            yield KEEP_ALIVE
            last_sent = time.monotonic()
