"""Black-box tests for direct structured-log SQLite queries."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from server.foundation.result import Ok, Rejected
from server.training.event_log import (
    ProcessIdentity,
    StructuredEventSink,
)
from server.training.persistence.schema import (
    database_path,
    initialize_database,
)
from server.training_control.log_feed import TrainingLogFeed
from server.training_control.log_queries import query_training_logs


def test_log_query_returns_latest_window_and_global_cursor(
    tmp_path: Path,
) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)
    sink = StructuredEventSink(
        run_dir=tmp_path,
        session_id="session-1",
        process=ProcessIdentity(kind="coordinator", index=0),
    )
    sink.emit("session.started")
    sink.emit("update.completed")
    sink.emit("session.completed")
    sink.close()

    result = query_training_logs(
        tmp_path,
        after_sequence=None,
        limit=2,
        event_types=(),
        session_id=None,
    )

    assert isinstance(result, Ok)
    assert result.value.through_sequence == 3
    assert result.value.full is True
    assert [item.event["event"] for item in result.value.records] == [
        "update.completed",
        "session.completed",
    ]


def test_log_query_applies_cursor_and_filters_without_losing_cursor(
    tmp_path: Path,
) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)
    sink = StructuredEventSink(
        run_dir=tmp_path,
        session_id="session-1",
        process=ProcessIdentity(kind="coordinator"),
    )
    sink.emit("session.started")
    sink.emit("update.completed")
    sink.emit("session.completed")
    sink.close()

    result = query_training_logs(
        tmp_path,
        after_sequence=1,
        limit=5000,
        event_types=("update.completed",),
        session_id="session-1",
    )

    assert isinstance(result, Ok)
    assert result.value.through_sequence == 3
    assert len(result.value.records) == 1
    assert result.value.records[0].sequence == 2


def test_log_query_missing_database_is_empty(tmp_path: Path) -> None:
    result = query_training_logs(
        tmp_path,
        after_sequence=None,
        limit=5000,
        event_types=(),
        session_id=None,
    )

    assert isinstance(result, Ok)
    assert result.value.through_sequence == 0
    assert result.value.records == ()


def test_log_query_rejects_foreign_database(tmp_path: Path) -> None:
    with sqlite3.connect(database_path(tmp_path)) as connection:
        connection.execute("CREATE TABLE foreign_data (value TEXT)")

    result = query_training_logs(
        tmp_path,
        after_sequence=None,
        limit=5000,
        event_types=(),
        session_id=None,
    )

    assert isinstance(result, Rejected)
    assert "unsupported" in result.reason


def test_log_query_rejects_non_positive_window(tmp_path: Path) -> None:
    result = query_training_logs(
        tmp_path,
        after_sequence=None,
        limit=0,
        event_types=(),
        session_id=None,
    )

    assert isinstance(result, Rejected)


def test_log_query_accepts_window_above_sqlite_integer_range(
    tmp_path: Path,
) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)

    result = query_training_logs(
        tmp_path,
        after_sequence=None,
        limit=1 << 100,
        event_types=(),
        session_id=None,
    )

    assert isinstance(result, Ok)


async def test_log_feed_pages_bursts_and_resets_after_replacement(
    tmp_path: Path,
) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)
    sink = StructuredEventSink(
        run_dir=tmp_path,
        session_id="session-1",
        process=ProcessIdentity(kind="coordinator"),
    )
    sink.emit("session.started")
    sink.close()
    feed = TrainingLogFeed(
        run_dir=tmp_path,
        window=2,
        event_types=(),
        session_id=None,
    )

    first = await feed.read()
    assert isinstance(first, Ok)
    assert first.value.reset is True
    sink = StructuredEventSink(
        run_dir=tmp_path,
        session_id="session-1",
        process=ProcessIdentity(kind="coordinator"),
    )
    for decision_index in range(3):
        sink.emit(
            "decision.completed",
            fields={"decision_index": decision_index},
        )
    sink.close()
    burst_page = await feed.read()
    assert isinstance(burst_page, Ok)
    assert burst_page.value.full is True
    assert burst_page.value.window == 2
    assert len(burst_page.value.records) == 2
    final_page = await feed.read()
    assert isinstance(final_page, Ok)
    assert final_page.value.full is False
    assert len(final_page.value.records) == 1

    for path in tmp_path.glob("training.sqlite3*"):
        path.unlink()
    reinitialized = initialize_database(tmp_path)
    assert isinstance(reinitialized, Ok)
    replacement_sink = StructuredEventSink(
        run_dir=tmp_path,
        session_id="session-2",
        process=ProcessIdentity(kind="coordinator"),
    )
    replacement_sink.emit("session.started")
    replacement_sink.close()
    replacement = await feed.read()

    assert isinstance(replacement, Ok)
    assert replacement.value.reset is True
    assert len(replacement.value.records) == 1
