"""Strict SQLite store and connection policy for training events."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import uuid4

from pydantic import ConfigDict, TypeAdapter

from server.foundation import result as _result
from server.training_events.contract import EVENT_NAMES, PROCESS_KINDS
from server.training_events.envelope import (
    TRAINING_EVENT_SCHEMA_VERSION,
)

DATABASE_FILENAME = "training.sqlite3"
APPLICATION_ID = 0x54524149
TRAINING_EVENT_STORE_SCHEMA_VERSION = 4

type _SqlRow = tuple[object, ...]
_OPTIONAL_SQL_ROW: TypeAdapter[_SqlRow | None] = TypeAdapter(
    _SqlRow | None,
    config=ConfigDict(strict=True),
)
_SQL_ROWS: TypeAdapter[list[_SqlRow]] = TypeAdapter(
    list[_SqlRow],
    config=ConfigDict(strict=True),
)

_EVENT_NAMES_SQL = ", ".join(f"'{name}'" for name in EVENT_NAMES)
_PROCESS_KINDS_SQL = ", ".join(f"'{kind}'" for kind in PROCESS_KINDS)

_CREATE_SCHEMA = f"""
CREATE TABLE training_log_store (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    store_id TEXT NOT NULL CHECK (
        length(store_id) = 32
        AND store_id NOT GLOB '*[^0-9a-f]*'
    )
) STRICT;

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
    process_kind TEXT GENERATED ALWAYS AS (
        json_extract(event_json, '$.process.kind')
    ) STORED NOT NULL,
    process_index INTEGER GENERATED ALWAYS AS (
        json_extract(event_json, '$.process.index')
    ) STORED,
    policy_version INTEGER GENERATED ALWAYS AS (
        json_extract(event_json, '$.context.policy_version')
    ) STORED,
    rollout_id TEXT GENERATED ALWAYS AS (
        json_extract(event_json, '$.context.rollout_id')
    ) STORED,
    round_id INTEGER GENERATED ALWAYS AS (
        json_extract(event_json, '$.context.round_id')
    ) STORED,
    CHECK (
        json_type(event_json, '$.schema_version') IS 'integer'
        AND json_extract(
            event_json, '$.schema_version'
        ) = {TRAINING_EVENT_SCHEMA_VERSION}
    ),
    CHECK (json_type(event_json, '$.recorded_at_ms') IS 'integer'),
    CHECK (json_type(event_json, '$.process') IS 'object'),
    CHECK (json_type(event_json, '$.context') IS 'object'),
    CHECK (json_type(event_json, '$.fields') IS 'object'),
    CHECK (
        json_type(event_json, '$.process.kind') IS 'text'
        AND length(json_extract(event_json, '$.process.kind')) > 0
    ),
    CHECK (
        json_type(event_json, '$.process.pid') IS 'integer'
        AND json_extract(event_json, '$.process.pid') > 0
    ),
    CHECK (
        json_type(event_json, '$.process.index') IS NULL
        OR json_type(event_json, '$.process.index') IS 'null'
        OR (
            json_type(event_json, '$.process.index') IS 'integer'
            AND json_extract(event_json, '$.process.index') >= 0
        )
    ),
    CHECK (
        json_type(event_json, '$.context.policy_version') IS NULL
        OR (
            json_type(
                event_json, '$.context.policy_version'
            ) IS 'integer'
            AND json_extract(
                event_json, '$.context.policy_version'
            ) >= 0
        )
    ),
    CHECK (
        json_type(event_json, '$.context.round_id') IS NULL
        OR (
            json_type(event_json, '$.context.round_id') IS 'integer'
            AND json_extract(event_json, '$.context.round_id') >= 0
        )
    ),
    CHECK (
        json_type(event_json, '$.context.seat') IS NULL
        OR (
            json_type(event_json, '$.context.seat') IS 'text'
            AND json_extract(event_json, '$.context.seat')
                IN ('a', 'b', 'c', 'd')
        )
    ),
    CHECK (
        json_type(event_json, '$.context.rollout_id') IS NULL
        OR (
            json_type(event_json, '$.context.rollout_id') IS 'text'
            AND length(json_extract(
                event_json, '$.context.rollout_id'
            )) > 0
        )
    ),
    CHECK (
        json_type(event_json, '$.error') IS NULL
        OR (
            json_type(event_json, '$.error') IS 'text'
            AND length(json_extract(event_json, '$.error')) > 0
        )
    ),
    CHECK (json_type(event_json, '$.fields.reason') IS NULL),
    CHECK (json_type(event_json, '$.fields.error') IS NULL),
    CHECK (json_type(event_json, '$.fields.outcome') IS NULL),
    CHECK (json_type(event_json, '$.fields.level') IS NULL),
    CHECK (json_type(event_json, '$.fields.session_id') IS NULL),
    CHECK (json_type(event_json, '$.level') IS NULL),
    CHECK (json_type(event_json, '$.outcome') IS NULL),
    CHECK (json_type(event_json, '$.session_id') IS NULL),
    CHECK (length(event_type) > 0),
    CHECK (event_type IN ({_EVENT_NAMES_SQL})),
    CHECK (length(process_kind) > 0),
    CHECK (process_kind IN ({_PROCESS_KINDS_SQL})),
    CHECK (process_index IS NULL OR process_index >= 0),
    CHECK (policy_version IS NULL OR policy_version >= 0),
    CHECK (rollout_id IS NULL OR length(rollout_id) > 0),
    CHECK (round_id IS NULL OR round_id >= 0)
) STRICT;

CREATE INDEX training_logs_event_sequence
ON training_logs(event_type, sequence DESC);

CREATE INDEX training_logs_policy_event_sequence
ON training_logs(policy_version, event_type, sequence);

CREATE INDEX training_logs_rollout_event_sequence
ON training_logs(rollout_id, event_type, sequence);

CREATE INDEX training_logs_round_sequence
ON training_logs(round_id, sequence);

CREATE INDEX training_logs_recorded_at
ON training_logs(recorded_at_ms);
"""


def database_path(run_dir: Path) -> Path:
    """Return the canonical SQLite path for one training run."""
    return run_dir / DATABASE_FILENAME


def training_store_id(connection: sqlite3.Connection) -> str:
    """Read the immutable identity of one initialized log store."""
    row = _OPTIONAL_SQL_ROW.validate_python(
        connection.execute(
            "SELECT store_id FROM training_log_store "
            + "WHERE singleton = 1"
        ).fetchone()
    )
    if row is None or not isinstance(row[0], str) or not row[0]:
        raise sqlite3.DatabaseError(
            "training log store identity is invalid"
        )
    return row[0]


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
            _ = connection.executescript(_CREATE_SCHEMA)
            _ = connection.execute(
                "INSERT INTO training_log_store(singleton, store_id) "
                + "VALUES (1, ?)",
                (uuid4().hex,),
            )
            _ = connection.execute(
                f"PRAGMA application_id = {APPLICATION_ID}"
            )
            _ = connection.execute(
                "PRAGMA user_version = "
                + f"{TRAINING_EVENT_STORE_SCHEMA_VERSION}"
            )
            connection.commit()
        elif (
            application_id != APPLICATION_ID
            or user_version != TRAINING_EVENT_STORE_SCHEMA_VERSION
        ):
            return _result.Rejected(
                reason=f"unsupported training database schema: {path}"
            )
        if not _schema_is_complete(connection):
            return _result.Rejected(
                reason=f"training database schema is invalid: {path}"
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
        _ = connection.execute("PRAGMA query_only = ON")
        _ = connection.execute("PRAGMA busy_timeout = 5000")
        application_id = _pragma_int(connection, "application_id")
        user_version = _pragma_int(connection, "user_version")
    except sqlite3.Error:
        return _result.Rejected(
            reason=f"training database is unreadable: {path}"
        )
    if (
        application_id != APPLICATION_ID
        or user_version != TRAINING_EVENT_STORE_SCHEMA_VERSION
        or not _schema_is_complete(connection)
    ):
        connection.close()
        return _result.Rejected(
            reason=f"unsupported training database schema: {path}"
        )
    return _result.Ok(value=connection)


def _configure_writer(connection: sqlite3.Connection) -> None:
    _ = connection.execute("PRAGMA journal_mode = WAL")
    _ = connection.execute("PRAGMA synchronous = NORMAL")
    _ = connection.execute("PRAGMA busy_timeout = 5000")
    _ = connection.execute("PRAGMA foreign_keys = ON")
    _ = connection.execute("PRAGMA trusted_schema = ON")


def _pragma_int(connection: sqlite3.Connection, name: str) -> int:
    row = _OPTIONAL_SQL_ROW.validate_python(
        connection.execute(f"PRAGMA {name}").fetchone()
    )
    assert row is not None
    value = row[0]
    assert isinstance(value, int)
    return value


def _table_count(connection: sqlite3.Connection) -> int:
    row = _OPTIONAL_SQL_ROW.validate_python(
        connection.execute(
            "SELECT count(*) FROM sqlite_schema WHERE type = 'table' "
            + "AND name NOT LIKE 'sqlite_%'"
        ).fetchone()
    )
    assert row is not None
    value = row[0]
    assert isinstance(value, int)
    return value


def _schema_is_complete(connection: sqlite3.Connection) -> bool:
    try:
        tables = _SQL_ROWS.validate_python(
            connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' "
                + "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        )
        if tables != [("training_log_store",), ("training_logs",)]:
            return False
        row = _OPTIONAL_SQL_ROW.validate_python(
            connection.execute(
                "SELECT store_id FROM training_log_store "
                + "WHERE singleton = 1"
            ).fetchone()
        )
        if row is None or not isinstance(row[0], str):
            return False
        store_id = row[0]
        if len(store_id) != 32 or any(
            character not in "0123456789abcdef"
            for character in store_id
        ):
            return False
        _ = connection.execute(
            "SELECT sequence, event_json, event_type, recorded_at_ms, "
            + "process_kind, process_index, policy_version, "
            + "rollout_id, round_id FROM training_logs LIMIT 0"
        )
    except sqlite3.Error:
        return False
    return True
