"""Black-box tests for the structured event log."""

from __future__ import annotations

import math
import sqlite3
import threading
from pathlib import Path
from typing import cast

import pytest
from pydantic import TypeAdapter

from server.foundation.json_value import JsonObject
from server.foundation.result import Ok, Rejected
from server.training_events import (
    EVENT_NAMES,
    EventContext,
    EventName,
    NullEventSink,
    ProcessIdentity,
    ProcessKind,
    StructuredEventSink,
    writer,
)
from server.training_events.store import (
    database_path,
    initialize_database,
)

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


def test_event_sink_batches_typed_json_and_preserves_context(
    tmp_path: Path,
) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)
    sink = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="worker", index=2),
    )
    sink.emit(
        "round",
        context=EventContext(
            policy_version=3,
            worker_index=2,
            episode_id=7,
        ),
        fields={"sample_count": 11},
    )
    sink.close()

    with sqlite3.connect(database_path(tmp_path)) as connection:
        row = connection.execute(
            "SELECT event_json FROM training_logs"
        ).fetchone()
    assert row is not None
    event_json = row[0]
    assert isinstance(event_json, str)
    event = _JSON_OBJECT_ADAPTER.validate_json(event_json)
    assert event["schema_version"] == 2
    assert event["event"] == "round"
    assert "session_id" not in event
    assert "level" not in event
    assert "error" not in event
    context = event["context"]
    assert isinstance(context, dict)
    assert context["episode_id"] == 7
    assert "rollout_id" not in context


def test_failed_event_uses_same_name_and_top_level_error(
    tmp_path: Path,
) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)
    sink = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="worker"),
    )
    sink.emit("round", fields={"duration_seconds": 1.0}, error="boom")
    sink.close()

    with sqlite3.connect(database_path(tmp_path)) as connection:
        row = connection.execute(
            "SELECT event_json FROM training_logs"
        ).fetchone()
    assert row is not None
    event_json = row[0]
    assert isinstance(event_json, str)
    event = _JSON_OBJECT_ADAPTER.validate_json(event_json)
    assert event["event"] == "round"
    assert event["error"] == "boom"
    fields = event["fields"]
    assert isinstance(fields, dict)
    assert "reason" not in fields


def test_event_context_rejects_invalid_identifiers() -> None:
    with pytest.raises(AssertionError):
        EventContext(policy_version=-1)
    with pytest.raises(AssertionError):
        EventContext(rollout_id="")
    with pytest.raises(AssertionError):
        ProcessIdentity(kind=cast(ProcessKind, "unknown"))


def test_emit_rejects_nonfinite_or_reserved_fields(
    tmp_path: Path,
) -> None:
    sink = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="coordinator"),
    )
    with pytest.raises(AssertionError):
        sink.emit("update", fields={"loss": math.nan})
    with pytest.raises(AssertionError):
        sink.emit("update", fields={"reason": "wrong layer"})
    with pytest.raises(AssertionError):
        sink.emit("update", error=" ")
    sink.close()


def test_null_sink_enforces_the_same_event_contract() -> None:
    sink = NullEventSink()
    with pytest.raises(AssertionError):
        sink.emit("update", fields={"loss": math.inf})
    with pytest.raises(AssertionError):
        sink.emit("update", fields={"error": "wrong layer"})
    with pytest.raises(AssertionError):
        sink.emit(cast(EventName, "update.finished"))


def test_store_accepts_every_contract_event_name(
    tmp_path: Path,
) -> None:
    sink = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="coordinator"),
    )
    for event_name in EVENT_NAMES:
        sink.emit(event_name)
    sink.close()

    with sqlite3.connect(database_path(tmp_path)) as connection:
        count = connection.execute(
            "SELECT count(*) FROM training_logs"
        ).fetchone()
    assert count == (len(EVENT_NAMES),)


def test_writer_close_waits_for_space_without_dropping_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocked_open(_run_dir: Path) -> Rejected:
        entered.set()
        release.wait()
        return Rejected(reason="injected open failure")

    monkeypatch.setattr(writer, "_QUEUE_CAPACITY", 1)
    monkeypatch.setattr(writer, "open_writer", blocked_open)
    sink = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="coordinator"),
    )
    sink.emit("update")
    assert entered.wait(timeout=1.0)
    closer = threading.Thread(target=sink.close)
    closer.start()
    assert closer.is_alive()
    release.set()
    closer.join(timeout=0.5)

    assert not closer.is_alive()
