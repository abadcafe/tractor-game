"""Black-box tests for the structured event log."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import TypeAdapter

from server.foundation.json_value import JsonObject
from server.foundation.result import Ok
from server.training.event_log import (
    EventContext,
    ProcessIdentity,
    StructuredEventSink,
)
from server.training.persistence.schema import (
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
        session_id="session-1",
        process=ProcessIdentity(kind="worker", index=2),
    )
    sink.emit(
        "round.completed",
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
    assert event["event"] == "round.completed"
    assert event["session_id"] == "session-1"
    context = event["context"]
    assert isinstance(context, dict)
    assert context["episode_id"] == 7
