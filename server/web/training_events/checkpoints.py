"""Checkpoint-catalog invalidation SSE endpoint."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from functools import partial
from pathlib import Path
from typing import Annotated, Literal

from anyio import to_thread
from fastapi import FastAPI, Query
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import StreamingResponse

from server.foundation import result as _result
from server.training_artifacts import (
    CheckpointInvalidation,
    query_checkpoint_invalidation,
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

type _StoreId = Annotated[str, Query(pattern=r"^[0-9a-f]{32}$")]


class CheckpointCursorEvent(BaseModel):
    """Current checkpoint-relevant event-store cursor."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    store_id: str | None
    through_sequence: int = Field(ge=0)


def register_checkpoints_route(
    app: FastAPI, state: ServerState
) -> None:
    async def training_checkpoints(
        run_dir: Path | None = None,
        store_id: _StoreId | None = None,
    ) -> StreamingResponse:
        canonical = state.training_control_config.resolve_run_dir(
            run_dir
        )
        return event_response(
            _checkpoint_events(canonical, expected_store=store_id)
        )

    app.add_api_route(
        "/api/training/events/checkpoints",
        training_checkpoints,
        methods=["GET"],
    )


async def _checkpoint_events(
    run_dir: Path, *, expected_store: str | None
) -> AsyncGenerator[bytes, None]:
    yield RETRY
    observed_sequence = -1
    last_sent = time.monotonic()
    while True:
        result = await to_thread.run_sync(
            partial(query_checkpoint_invalidation, run_dir)
        )
        if isinstance(result, _result.Rejected):
            yield rejected_event(result.reason).encode()
            return
        invalidation = result.value
        if invalidation.store_id != expected_store:
            yield _cursor_event("replacement", invalidation).encode()
            return
        if invalidation.through_sequence != observed_sequence:
            observed_sequence = invalidation.through_sequence
            yield _cursor_event("invalidation", invalidation).encode()
            last_sent = time.monotonic()
        await asyncio.sleep(_POLL_SECONDS)
        if time.monotonic() - last_sent >= _HEARTBEAT_SECONDS:
            yield KEEP_ALIVE
            last_sent = time.monotonic()


def _cursor_event(
    name: Literal["invalidation", "replacement"],
    invalidation: CheckpointInvalidation,
) -> ServerEvent:
    return ServerEvent(
        name=name,
        data=CheckpointCursorEvent(
            store_id=invalidation.store_id,
            through_sequence=invalidation.through_sequence,
        ).model_dump_json(),
    )
