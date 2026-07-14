"""Black-box tests for raw training event history and tail queries."""

from __future__ import annotations

from pathlib import Path

from server.foundation.result import Ok
from server.training_events import ProcessIdentity, StructuredEventSink
from server.training_events.queries import (
    query_training_log_history,
    query_training_log_tail,
)
from server.training_events.store import initialize_database


def test_history_pages_backward_and_preserves_store_identity(
    tmp_path: Path,
) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)
    sink = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="coordinator"),
    )
    sink.emit("initialize")
    sink.emit("update")
    sink.emit("training")
    sink.close()

    latest = query_training_log_history(
        tmp_path, before_sequence=None, limit=2
    )

    assert isinstance(latest, Ok)
    assert latest.value.store_id is not None
    assert [item.sequence for item in latest.value.events] == [2, 3]
    assert latest.value.next_before_sequence == 2
    older = query_training_log_history(
        tmp_path,
        before_sequence=latest.value.next_before_sequence,
        limit=2,
    )
    assert isinstance(older, Ok)
    assert older.value.store_id == latest.value.store_id
    assert [item.sequence for item in older.value.events] == [1]


def test_tail_returns_events_after_sequence(tmp_path: Path) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)
    sink = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="coordinator"),
    )
    sink.emit("initialize")
    sink.emit("training")
    sink.close()

    tail = query_training_log_tail(
        tmp_path, after_sequence=1, limit=100
    )

    assert isinstance(tail, Ok)
    assert tail.value.store_id is not None
    assert [item.sequence for item in tail.value.events] == [2]


def test_replacement_changes_store_id_even_when_sequence_restarts(
    tmp_path: Path,
) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)
    first = query_training_log_tail(
        tmp_path, after_sequence=0, limit=100
    )
    assert isinstance(first, Ok)
    assert first.value.store_id is not None
    for path in tmp_path.glob("training.sqlite3*"):
        path.unlink()
    replacement = initialize_database(tmp_path)
    assert isinstance(replacement, Ok)
    second = query_training_log_tail(
        tmp_path, after_sequence=0, limit=100
    )
    assert isinstance(second, Ok)
    assert second.value.store_id is not None
    assert second.value.store_id != first.value.store_id


def test_missing_store_returns_explicit_no_store(
    tmp_path: Path,
) -> None:
    history = query_training_log_history(
        tmp_path, before_sequence=None, limit=100
    )
    tail = query_training_log_tail(
        tmp_path, after_sequence=0, limit=100
    )

    assert isinstance(history, Ok)
    assert isinstance(tail, Ok)
    assert history.value.store_id is None
    assert tail.value.store_id is None
