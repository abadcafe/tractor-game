"""Cursor queries for append-only structured training logs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from server.foundation import result as _result
from server.foundation.json_value import JsonObject
from server.training_events.store import open_reader, training_store_id


class TrainingLogRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sequence: int = Field(gt=0)
    event: JsonObject


class TrainingLogHistoryPage(BaseModel):
    """Newest-first cursor page serialized in chronological order."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    store_id: str | None
    events: tuple[TrainingLogRecord, ...]
    next_before_sequence: int | None = Field(default=None, gt=0)


class TrainingLogTail(BaseModel):
    """One chronological tail batch from an immutable log store."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    store_id: str | None
    events: tuple[TrainingLogRecord, ...]


def query_training_log_history(
    run_dir: Path,
    *,
    before_sequence: int | None,
    limit: int,
) -> _result.Ok[TrainingLogHistoryPage] | _result.Rejected:
    """Read the latest page or the page strictly before a sequence."""
    if before_sequence is not None and before_sequence <= 0:
        return _result.Rejected(
            reason="before_sequence must be positive"
        )
    if limit <= 0 or limit > 1000:
        return _result.Rejected(
            reason="limit must be between 1 and 1000"
        )
    opened = open_reader(run_dir)
    if isinstance(opened, _result.Rejected):
        return opened
    connection = opened.value
    if connection is None:
        return _result.Ok(
            value=TrainingLogHistoryPage(
                store_id=None, events=(), next_before_sequence=None
            )
        )
    try:
        store_id = training_store_id(connection)
        if before_sequence is None:
            rows = connection.execute(
                "SELECT sequence, event_json FROM training_logs "
                "ORDER BY sequence DESC LIMIT ?",
                (limit + 1,),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT sequence, event_json FROM training_logs "
                "WHERE sequence < ? ORDER BY sequence DESC LIMIT ?",
                (before_sequence, limit + 1),
            ).fetchall()
    except sqlite3.Error:
        return _result.Rejected(reason="training logs query failed")
    finally:
        connection.close()
    has_older = len(rows) > limit
    selected = rows[:limit]
    parsed = _parse_records(list(reversed(selected)))
    if isinstance(parsed, _result.Rejected):
        return parsed
    cursor = (
        parsed.value[0].sequence if has_older and parsed.value else None
    )
    return _result.Ok(
        value=TrainingLogHistoryPage(
            store_id=store_id,
            events=parsed.value,
            next_before_sequence=cursor,
        )
    )


def query_training_log_tail(
    run_dir: Path, *, after_sequence: int, limit: int
) -> _result.Ok[TrainingLogTail] | _result.Rejected:
    """Read a chronological incremental batch after a commit cursor."""
    if after_sequence < 0:
        return _result.Rejected(
            reason="after_sequence must be non-negative"
        )
    if limit <= 0 or limit > 1000:
        return _result.Rejected(
            reason="limit must be between 1 and 1000"
        )
    opened = open_reader(run_dir)
    if isinstance(opened, _result.Rejected):
        return opened
    connection = opened.value
    if connection is None:
        return _result.Ok(
            value=TrainingLogTail(store_id=None, events=())
        )
    try:
        store_id = training_store_id(connection)
        rows = connection.execute(
            "SELECT sequence, event_json FROM training_logs "
            "WHERE sequence > ? ORDER BY sequence LIMIT ?",
            (after_sequence, limit),
        ).fetchall()
    except sqlite3.Error:
        return _result.Rejected(reason="training logs query failed")
    finally:
        connection.close()
    parsed = _parse_records(rows)
    if isinstance(parsed, _result.Rejected):
        return parsed
    return _result.Ok(
        value=TrainingLogTail(store_id=store_id, events=parsed.value)
    )


def _parse_records(
    rows: list[tuple[object, object]],
) -> _result.Ok[tuple[TrainingLogRecord, ...]] | _result.Rejected:
    records: list[TrainingLogRecord] = []
    try:
        for sequence, event_json in rows:
            if not isinstance(sequence, int) or not isinstance(
                event_json, str
            ):
                raise ValueError
            decoded = json.loads(event_json)
            if not isinstance(decoded, dict):
                raise ValueError
            records.append(
                TrainingLogRecord.model_validate(
                    {"sequence": sequence, "event": decoded}
                )
            )
    except json.JSONDecodeError, ValidationError, ValueError:
        return _result.Rejected(reason="training log event is invalid")
    return _result.Ok(value=tuple(records))
