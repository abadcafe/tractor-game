"""Black-box tests for strict training database setup."""

import json
import sqlite3
from pathlib import Path

from server.foundation.json_value import JsonObject
from server.foundation.result import Ok, Rejected
from server.training_events.store import (
    APPLICATION_ID,
    TRAINING_EVENT_STORE_SCHEMA_VERSION,
    database_path,
    initialize_database,
    open_reader,
)
from tests.sqlite_support import fetch_optional_row, fetch_rows


def test_initialize_database_creates_strict_schema(
    tmp_path: Path,
) -> None:
    result = initialize_database(tmp_path)

    assert isinstance(result, Ok)
    with sqlite3.connect(database_path(tmp_path)) as connection:
        application_id = fetch_optional_row(
            connection.execute("PRAGMA application_id")
        )
        user_version = fetch_optional_row(
            connection.execute("PRAGMA user_version")
        )
        tables = fetch_rows(
            connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' "
                + "ORDER BY name"
            )
        )
    assert application_id == (APPLICATION_ID,)
    assert user_version == (TRAINING_EVENT_STORE_SCHEMA_VERSION,)
    assert TRAINING_EVENT_STORE_SCHEMA_VERSION == 5
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
        _ = connection.execute("CREATE TABLE foreign_data (value TEXT)")

    result = initialize_database(tmp_path)

    assert isinstance(result, Rejected)
    assert "unsupported" in result.reason


def test_schema_rejects_incomplete_event_envelopes(
    tmp_path: Path,
) -> None:
    result = initialize_database(tmp_path)
    assert isinstance(result, Ok)
    base: JsonObject = {
        "schema_version": 5,
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
    incomplete_round = dict(base)
    incomplete_round["event"] = "round"
    invalid.append(incomplete_round)
    misplaced_round_id = dict(base)
    misplaced_round_id["context"] = {"round_id": 1}
    invalid.append(misplaced_round_id)
    with sqlite3.connect(database_path(tmp_path)) as connection:
        for payload in invalid:
            _ = connection.execute(
                "INSERT OR IGNORE INTO training_logs(event_json) "
                + "VALUES (?)",
                (json.dumps(payload),),
            )
        stored = fetch_optional_row(
            connection.execute("SELECT count(*) FROM training_logs")
        )
    assert stored == (0,)


def test_database_has_one_record_per_rollout_round_identity(
    tmp_path: Path,
) -> None:
    result = initialize_database(tmp_path)
    assert isinstance(result, Ok)
    event: JsonObject = {
        "schema_version": 5,
        "event": "round",
        "recorded_at_ms": 1,
        "process": {"kind": "worker", "index": 0, "pid": 1},
        "context": {
            "policy_version": 0,
            "rollout_id": "rollout-0",
            "worker_index": 0,
            "game_env_index": 0,
            "round_id": 1,
        },
        "fields": {"round_count": 1},
    }
    with sqlite3.connect(database_path(tmp_path)) as connection:
        for _index in range(2):
            _ = connection.execute(
                "INSERT OR IGNORE INTO training_logs(event_json) "
                + "VALUES (?)",
                (json.dumps(event),),
            )
        stored = fetch_optional_row(
            connection.execute("SELECT count(*) FROM training_logs")
        )
    assert stored == (1,)


def test_initialize_database_rejects_incomplete_current_store(
    tmp_path: Path,
) -> None:
    with sqlite3.connect(database_path(tmp_path)) as connection:
        _ = connection.execute("CREATE TABLE training_logs(value TEXT)")
        _ = connection.execute(
            f"PRAGMA application_id = {APPLICATION_ID}"
        )
        _ = connection.execute(
            "PRAGMA user_version = "
            + f"{TRAINING_EVENT_STORE_SCHEMA_VERSION}"
        )

    result = initialize_database(tmp_path)

    assert isinstance(result, Rejected)
    assert "invalid" in result.reason
