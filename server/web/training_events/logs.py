"""Resumable structured-log SSE endpoint."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncGenerator
from functools import partial
from pathlib import Path
from typing import Annotated

from anyio import to_thread
from fastapi import FastAPI, Query, Request
from pydantic import BaseModel, ConfigDict
from starlette.responses import StreamingResponse

from server.foundation import result as _result
from server.training_events.queries import query_training_log_tail
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
_TAIL_BATCH_SIZE = 500
_EVENT_ID = re.compile(r"^([0-9a-f]{32}):([1-9][0-9]*)$")

type _StoreId = Annotated[str, Query(pattern=r"^[0-9a-f]{32}$")]


class StoreReplacement(BaseModel):
    """The selected run now refers to a different event store."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    store_id: str | None


def register_logs_route(app: FastAPI, state: ServerState) -> None:
    async def training_logs(
        request: Request,
        run_dir: Path | None = None,
        after_sequence: Annotated[int, Query(ge=0)] = 0,
        store_id: _StoreId | None = None,
    ) -> StreamingResponse:
        canonical = state.training_control_config.resolve_run_dir(
            run_dir
        )
        return event_response(
            _log_events(
                canonical,
                expected_store=store_id,
                after_sequence=after_sequence,
                last_event_id=request.headers.get("last-event-id"),
            )
        )

    app.add_api_route(
        "/api/training/events/logs", training_logs, methods=["GET"]
    )


async def _log_events(
    run_dir: Path,
    *,
    expected_store: str | None,
    after_sequence: int,
    last_event_id: str | None,
) -> AsyncGenerator[bytes, None]:
    yield RETRY
    resumed = _resume_cursor(
        expected_store=expected_store,
        after_sequence=after_sequence,
        last_event_id=last_event_id,
    )
    if isinstance(resumed, _result.Rejected):
        yield rejected_event(resumed.reason).encode()
        return
    store_id, cursor = resumed.value
    last_sent = time.monotonic()
    while True:
        result = await to_thread.run_sync(
            partial(
                query_training_log_tail,
                run_dir,
                after_sequence=cursor,
                limit=_TAIL_BATCH_SIZE,
            )
        )
        if isinstance(result, _result.Rejected):
            yield rejected_event(result.reason).encode()
            return
        tail = result.value
        if tail.store_id != store_id:
            yield ServerEvent(
                name="replacement",
                data=StoreReplacement(
                    store_id=tail.store_id
                ).model_dump_json(),
            ).encode()
            return
        for record in tail.events:
            assert store_id is not None
            cursor = record.sequence
            yield ServerEvent(
                name="log",
                event_id=f"{store_id}:{record.sequence}",
                data=record.model_dump_json(),
            ).encode()
            last_sent = time.monotonic()
        if len(tail.events) == _TAIL_BATCH_SIZE:
            continue
        await asyncio.sleep(_POLL_SECONDS)
        if time.monotonic() - last_sent >= _HEARTBEAT_SECONDS:
            yield KEEP_ALIVE
            last_sent = time.monotonic()


def _resume_cursor(
    *,
    expected_store: str | None,
    after_sequence: int,
    last_event_id: str | None,
) -> _result.Ok[tuple[str | None, int]] | _result.Rejected:
    if last_event_id is None:
        if expected_store is None and after_sequence != 0:
            return _result.Rejected(
                reason="after_sequence requires store_id"
            )
        return _result.Ok(value=(expected_store, after_sequence))
    match = _EVENT_ID.fullmatch(last_event_id)
    if match is None:
        return _result.Rejected(reason="Last-Event-ID is invalid")
    resumed_store, sequence_text = match.groups()
    if expected_store != resumed_store:
        return _result.Rejected(
            reason="Last-Event-ID does not match store_id"
        )
    return _result.Ok(value=(resumed_store, int(sequence_text)))
