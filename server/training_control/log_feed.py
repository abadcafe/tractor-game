"""Stateful direct-SQLite feed for one training log client."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from anyio import to_thread
from pydantic import BaseModel, ConfigDict

from server.foundation import result as _result
from server.training_control.log_queries import (
    TrainingLogPage,
    TrainingLogRecord,
    query_training_logs,
)


class TrainingLogBatch(BaseModel):
    """Events emitted by one feed poll."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    reset: bool
    window: int
    full: bool
    records: tuple[TrainingLogRecord, ...]


class TrainingLogFeed:
    """Track cursor and database replacement for one WebSocket."""

    def __init__(
        self,
        *,
        run_dir: Path,
        window: int,
        event_types: tuple[str, ...],
        session_id: str | None,
    ) -> None:
        assert window > 0
        self._run_dir = run_dir
        self._window = window
        self._event_types = event_types
        self._session_id = session_id
        self._initialized = False
        self._database_generation: str | None = None
        self._cursor = 0

    async def read(
        self,
    ) -> _result.Ok[TrainingLogBatch] | _result.Rejected:
        page_result = await self._query(
            after_sequence=(
                None if not self._initialized else self._cursor
            )
        )
        if isinstance(page_result, _result.Rejected):
            return page_result
        page = page_result.value
        reset = (
            not self._initialized
            or page.database_generation != self._database_generation
            or page.through_sequence < self._cursor
        )
        if reset and self._initialized:
            page_result = await self._query(after_sequence=None)
            if isinstance(page_result, _result.Rejected):
                return page_result
            page = page_result.value
        self._initialized = True
        self._database_generation = page.database_generation
        if reset or not page.full:
            self._cursor = page.through_sequence
        elif page.records:
            self._cursor = page.records[-1].sequence
        return _result.Ok(
            value=TrainingLogBatch(
                reset=reset,
                window=self._window,
                full=page.full,
                records=page.records,
            )
        )

    async def _query(
        self, *, after_sequence: int | None
    ) -> _result.Ok[TrainingLogPage] | _result.Rejected:
        return await to_thread.run_sync(
            partial(
                query_training_logs,
                self._run_dir,
                after_sequence=after_sequence,
                limit=self._window,
                event_types=self._event_types,
                session_id=self._session_id,
            )
        )
