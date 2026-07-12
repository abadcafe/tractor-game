"""Read-only SQLite pages for structured training logs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from server.foundation import result as _result
from server.foundation.json_value import JsonObject
from server.training_control.database import (
    open_training_database,
    training_database_generation,
)

_SQLITE_MAX_LIMIT = (1 << 63) - 1


class TrainingLogRecord(BaseModel):
    """One structured event and its stable commit cursor."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sequence: int = Field(gt=0)
    event: JsonObject


class TrainingLogPage(BaseModel):
    """One consistent event page plus the database high-water mark."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    database_generation: str | None
    through_sequence: int = Field(ge=0)
    full: bool
    records: tuple[TrainingLogRecord, ...]


def query_training_logs(
    run_dir: Path,
    *,
    after_sequence: int | None,
    limit: int,
    event_types: tuple[str, ...],
    session_id: str | None,
) -> _result.Ok[TrainingLogPage] | _result.Rejected:
    """Read a latest or incremental page in one SQLite transaction."""
    if after_sequence is not None and after_sequence < 0:
        return _result.Rejected(
            reason="after_sequence must be non-negative"
        )
    if limit <= 0:
        return _result.Rejected(reason="limit must be positive")
    return _query_training_logs(
        run_dir,
        after_sequence=after_sequence,
        limit=limit,
        event_types=event_types,
        session_id=session_id,
        retry_replacement=True,
    )


def _query_training_logs(
    run_dir: Path,
    *,
    after_sequence: int | None,
    limit: int,
    event_types: tuple[str, ...],
    session_id: str | None,
    retry_replacement: bool,
) -> _result.Ok[TrainingLogPage] | _result.Rejected:
    generation = training_database_generation(run_dir)
    if isinstance(generation, _result.Rejected):
        return generation
    opened = open_training_database(run_dir)
    if isinstance(opened, _result.Rejected):
        return opened
    connection = opened.value
    if connection is None:
        return _result.Ok(
            value=TrainingLogPage(
                database_generation=None,
                through_sequence=0,
                full=False,
                records=(),
            )
        )
    try:
        connection.execute("BEGIN")
        through_sequence = _through_sequence(connection)
        rows = _query_rows(
            connection,
            after_sequence=after_sequence,
            limit=limit,
            event_types=event_types,
            session_id=session_id,
        )
        connection.commit()
    except sqlite3.Error:
        connection.rollback()
        return _result.Rejected(reason="training logs query failed")
    finally:
        connection.close()
    parsed = _parse_records(rows)
    if isinstance(parsed, _result.Rejected):
        return parsed
    current_generation = training_database_generation(run_dir)
    if isinstance(current_generation, _result.Rejected):
        return current_generation
    if current_generation.value != generation.value:
        if not retry_replacement:
            return _result.Rejected(
                reason="training database changed during log query"
            )
        return _query_training_logs(
            run_dir,
            after_sequence=after_sequence,
            limit=limit,
            event_types=event_types,
            session_id=session_id,
            retry_replacement=False,
        )
    return _result.Ok(
        value=TrainingLogPage(
            database_generation=generation.value,
            through_sequence=through_sequence,
            full=len(parsed.value) == limit,
            records=parsed.value,
        )
    )


def _query_rows(
    connection: sqlite3.Connection,
    *,
    after_sequence: int | None,
    limit: int,
    event_types: tuple[str, ...],
    session_id: str | None,
) -> list[tuple[object, object]]:
    conditions: list[str] = []
    parameters: list[str | int] = []
    if after_sequence is not None:
        conditions.append("sequence > ?")
        parameters.append(after_sequence)
    if event_types:
        placeholders = ", ".join("?" for _item in event_types)
        conditions.append(f"event_type IN ({placeholders})")
        parameters.extend(event_types)
    if session_id is not None:
        conditions.append("session_id = ?")
        parameters.append(session_id)
    where = (
        "" if not conditions else "WHERE " + " AND ".join(conditions)
    )
    if after_sequence is None:
        query = (
            "SELECT sequence, event_json FROM ("
            "SELECT sequence, event_json FROM training_logs "
            f"{where} ORDER BY sequence DESC LIMIT ?"
            ") ORDER BY sequence"
        )
    else:
        query = (
            "SELECT sequence, event_json FROM training_logs "
            f"{where} ORDER BY sequence LIMIT ?"
        )
    parameters.append(min(limit, _SQLITE_MAX_LIMIT))
    return connection.execute(query, parameters).fetchall()


def _through_sequence(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT coalesce(max(sequence), 0) FROM training_logs"
    ).fetchone()
    assert row is not None
    value = row[0]
    assert isinstance(value, int)
    return value


def _parse_records(
    rows: list[tuple[object, object]],
) -> _result.Ok[tuple[TrainingLogRecord, ...]] | _result.Rejected:
    records: list[TrainingLogRecord] = []
    try:
        for sequence, event_json in rows:
            assert isinstance(sequence, int)
            assert isinstance(event_json, str)
            decoded = json.loads(event_json)
            assert isinstance(decoded, dict)
            records.append(
                TrainingLogRecord.model_validate(
                    {"sequence": sequence, "event": decoded}
                )
            )
    except json.JSONDecodeError, ValidationError:
        return _result.Rejected(reason="training log event is invalid")
    return _result.Ok(value=tuple(records))
