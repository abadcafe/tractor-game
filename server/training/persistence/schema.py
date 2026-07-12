"""Strict SQLite schema and connection policy for training events."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from server.foundation import result as _result

DATABASE_FILENAME = "training.sqlite3"
APPLICATION_ID = 0x54524149
SCHEMA_VERSION = 2

_CREATE_SCHEMA = """
CREATE TABLE training_logs (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_json TEXT NOT NULL CHECK (
        json_valid(event_json) AND json_type(event_json) = 'object'
    ),
    event_type TEXT GENERATED ALWAYS AS (
        json_extract(event_json, '$.event')
    ) STORED NOT NULL,
    recorded_at_ms INTEGER GENERATED ALWAYS AS (
        json_extract(event_json, '$.recorded_at_ms')
    ) STORED NOT NULL CHECK (recorded_at_ms >= 0),
    session_id TEXT GENERATED ALWAYS AS (
        json_extract(event_json, '$.session_id')
    ) STORED,
    process_kind TEXT GENERATED ALWAYS AS (
        json_extract(event_json, '$.process.kind')
    ) STORED NOT NULL,
    process_index INTEGER GENERATED ALWAYS AS (
        json_extract(event_json, '$.process.index')
    ) STORED,
    policy_version INTEGER GENERATED ALWAYS AS (
        json_extract(event_json, '$.context.policy_version')
    ) STORED,
    episode_id INTEGER GENERATED ALWAYS AS (
        json_extract(event_json, '$.context.episode_id')
    ) STORED,
    CHECK (length(event_type) > 0),
    CHECK (length(process_kind) > 0),
    CHECK (process_index IS NULL OR process_index >= 0),
    CHECK (policy_version IS NULL OR policy_version >= 0),
    CHECK (episode_id IS NULL OR episode_id >= 0)
) STRICT;

CREATE INDEX training_logs_event_sequence
ON training_logs(event_type, sequence DESC);

CREATE INDEX training_logs_session_event_sequence
ON training_logs(session_id, event_type, sequence DESC);

CREATE INDEX training_logs_session_policy_event
ON training_logs(session_id, policy_version, event_type);

CREATE INDEX training_logs_session_episode_sequence
ON training_logs(session_id, episode_id, sequence);

CREATE INDEX training_logs_recorded_at
ON training_logs(recorded_at_ms);
"""


def database_path(run_dir: Path) -> Path:
    """Return the canonical SQLite path for one training run."""
    return run_dir / DATABASE_FILENAME


def initialize_database(
    run_dir: Path,
) -> _result.Ok[None] | _result.Rejected:
    """Create or strictly validate the event database."""
    path = database_path(run_dir)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=5.0)
    except sqlite3.Error, OSError:
        return _result.Rejected(
            reason=f"training database could not be opened: {path}"
        )
    try:
        _configure_writer(connection)
        application_id = _pragma_int(connection, "application_id")
        user_version = _pragma_int(connection, "user_version")
        tables = _table_count(connection)
        if tables == 0 and application_id == 0 and user_version == 0:
            connection.executescript(_CREATE_SCHEMA)
            connection.execute(
                f"PRAGMA application_id = {APPLICATION_ID}"
            )
            connection.execute(
                f"PRAGMA user_version = {SCHEMA_VERSION}"
            )
            connection.commit()
        elif (
            application_id != APPLICATION_ID
            or user_version != SCHEMA_VERSION
        ):
            return _result.Rejected(
                reason=f"unsupported training database schema: {path}"
            )
    except sqlite3.Error:
        return _result.Rejected(
            reason=f"training database schema is invalid: {path}"
        )
    finally:
        connection.close()
    return _result.Ok(value=None)


def open_writer(
    run_dir: Path,
) -> _result.Ok[sqlite3.Connection] | _result.Rejected:
    """Open a configured writer after strict schema validation."""
    initialized = initialize_database(run_dir)
    if isinstance(initialized, _result.Rejected):
        return initialized
    path = database_path(run_dir)
    try:
        connection = sqlite3.connect(path, timeout=5.0)
        _configure_writer(connection)
    except sqlite3.Error:
        return _result.Rejected(
            reason=f"training database could not be opened: {path}"
        )
    return _result.Ok(value=connection)


def open_reader(
    run_dir: Path,
) -> _result.Ok[sqlite3.Connection | None] | _result.Rejected:
    """Open an existing database in read-only/query-only mode."""
    path = database_path(run_dir)
    if not path.exists():
        return _result.Ok(value=None)
    try:
        connection = sqlite3.connect(
            f"file:{path}?mode=ro", uri=True, timeout=5.0
        )
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        application_id = _pragma_int(connection, "application_id")
        user_version = _pragma_int(connection, "user_version")
    except sqlite3.Error:
        return _result.Rejected(
            reason=f"training database is unreadable: {path}"
        )
    if (
        application_id != APPLICATION_ID
        or user_version != SCHEMA_VERSION
    ):
        connection.close()
        return _result.Rejected(
            reason=f"unsupported training database schema: {path}"
        )
    return _result.Ok(value=connection)


def _configure_writer(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA trusted_schema = ON")


def _pragma_int(connection: sqlite3.Connection, name: str) -> int:
    row = connection.execute(f"PRAGMA {name}").fetchone()
    assert row is not None
    value = row[0]
    assert isinstance(value, int)
    return value


def _table_count(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT count(*) FROM sqlite_schema WHERE type = 'table' "
        "AND name NOT LIKE 'sqlite_%'"
    ).fetchone()
    assert row is not None
    value = row[0]
    assert isinstance(value, int)
    return value
