"""Black-box tests for strict training database setup."""

import json
import sqlite3
from pathlib import Path

import pytest

from server.foundation.json_value import JsonObject
from server.foundation.result import Ok, Rejected
from server.training_events.store import (
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
    assert SCHEMA_VERSION == 3
    assert tables == [
        ("sqlite_sequence",),
        ("training_log_store",),
        ("training_logs",),
    ]


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


def test_schema_rejects_incomplete_event_envelopes(
    tmp_path: Path,
) -> None:
    result = initialize_database(tmp_path)
    assert isinstance(result, Ok)
    base: JsonObject = {
        "schema_version": 2,
        "event": "update",
        "recorded_at_ms": 1,
        "process": {"kind": "coordinator", "pid": 1},
        "context": {},
        "fields": {},
    }
    invalid: list[JsonObject] = []
    for path in (
        "schema_version",
        "context",
        "fields",
    ):
        payload = dict(base)
        del payload[path]
        invalid.append(payload)
    missing_pid = dict(base)
    missing_pid["process"] = {"kind": "coordinator"}
    invalid.append(missing_pid)
    string_policy = dict(base)
    string_policy["context"] = {"policy_version": "1"}
    invalid.append(string_policy)
    nested_reason = dict(base)
    nested_reason["fields"] = {"reason": "wrong layer"}
    invalid.append(nested_reason)
    unknown_event = dict(base)
    unknown_event["event"] = "unknown"
    invalid.append(unknown_event)
    unknown_process = dict(base)
    unknown_process["process"] = {"kind": "unknown", "pid": 1}
    invalid.append(unknown_process)
    with sqlite3.connect(database_path(tmp_path)) as connection:
        for payload in invalid:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO training_logs(event_json) VALUES (?)",
                    (json.dumps(payload),),
                )


def test_initialize_database_rejects_fake_v3_without_store(
    tmp_path: Path,
) -> None:
    with sqlite3.connect(database_path(tmp_path)) as connection:
        connection.execute("CREATE TABLE training_logs(value TEXT)")
        connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    result = initialize_database(tmp_path)

    assert isinstance(result, Rejected)
    assert "invalid" in result.reason
