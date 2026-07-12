"""Validated read-only access to one training event database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from server.foundation import result as _result

_DATABASE_FILENAME = "training.sqlite3"
_APPLICATION_ID = 0x54524149
_SCHEMA_VERSION = 2


def open_training_database(
    run_dir: Path,
) -> _result.Ok[sqlite3.Connection | None] | _result.Rejected:
    """Open a validated query-only database without creating files."""
    path = run_dir / _DATABASE_FILENAME
    connection: sqlite3.Connection | None = None
    try:
        if not path.exists():
            return _result.Ok(value=None)
        connection = sqlite3.connect(
            f"file:{path}?mode=ro", uri=True, timeout=5.0
        )
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        application_id = _pragma_int(connection, "application_id")
        schema_version = _pragma_int(connection, "user_version")
    except sqlite3.Error, OSError:
        if connection is not None:
            connection.close()
        return _result.Rejected(
            reason=f"training database is unreadable: {path}"
        )
    if (
        application_id != _APPLICATION_ID
        or schema_version != _SCHEMA_VERSION
    ):
        connection.close()
        return _result.Rejected(
            reason=f"unsupported training database schema: {path}"
        )
    return _result.Ok(value=connection)


def training_database_generation(
    run_dir: Path,
) -> _result.Ok[str | None] | _result.Rejected:
    """Return the file identity used to detect atomic replacement."""
    path = run_dir / _DATABASE_FILENAME
    try:
        stat = path.stat()
    except FileNotFoundError:
        return _result.Ok(value=None)
    except OSError:
        return _result.Rejected(
            reason=f"training database is unreadable: {path}"
        )
    return _result.Ok(value=f"{stat.st_dev}:{stat.st_ino}")


def _pragma_int(connection: sqlite3.Connection, name: str) -> int:
    row = connection.execute(f"PRAGMA {name}").fetchone()
    assert row is not None
    value = row[0]
    assert isinstance(value, int)
    return value
