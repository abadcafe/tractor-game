"""Black-box tests for strict training database setup."""

import sqlite3
from pathlib import Path

from server.foundation.result import Ok, Rejected
from server.training.persistence.schema import (
    APPLICATION_ID,
    SCHEMA_VERSION,
    database_path,
    initialize_database,
    open_reader,
)


def test_initialize_database_creates_strict_schema(
    tmp_path: Path,
) -> None:
    result = initialize_database(tmp_path)

    assert isinstance(result, Ok)
    with sqlite3.connect(database_path(tmp_path)) as connection:
        application_id = connection.execute(
            "PRAGMA application_id"
        ).fetchone()
        user_version = connection.execute(
            "PRAGMA user_version"
        ).fetchone()
        tables = connection.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table' "
            "ORDER BY name"
        ).fetchall()
    assert application_id == (APPLICATION_ID,)
    assert user_version == (SCHEMA_VERSION,)
    assert ("training_metrics",) in tables
    assert ("runtime_telemetry",) in tables


def test_open_reader_missing_database_returns_none(
    tmp_path: Path,
) -> None:
    result = open_reader(tmp_path)

    assert isinstance(result, Ok)
    assert result.value is None


def test_initialize_database_rejects_foreign_database(
    tmp_path: Path,
) -> None:
    with sqlite3.connect(database_path(tmp_path)) as connection:
        connection.execute("CREATE TABLE foreign_data (value TEXT)")

    result = initialize_database(tmp_path)

    assert isinstance(result, Rejected)
    assert "unsupported" in result.reason
