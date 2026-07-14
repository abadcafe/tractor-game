"""Tests for checkpoint catalog invalidation ownership."""

from pathlib import Path

from server.foundation.result import Ok
from server.training_artifacts.invalidation import (
    query_checkpoint_invalidation,
)
from server.training_events import ProcessIdentity, StructuredEventSink
from server.training_events.store import initialize_database


def test_failed_checkpoint_invalidates_and_replacement_changes_store(
    tmp_path: Path,
) -> None:
    initialized = initialize_database(tmp_path)
    assert isinstance(initialized, Ok)
    first = query_checkpoint_invalidation(tmp_path)
    assert isinstance(first, Ok)
    assert first.value.store_id is not None
    assert first.value.through_sequence == 0

    sink = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="coordinator"),
    )
    sink.emit("checkpoint", error="state snapshot failed")
    sink.close()
    failed = query_checkpoint_invalidation(tmp_path)
    assert isinstance(failed, Ok)
    assert failed.value.store_id == first.value.store_id
    assert failed.value.through_sequence == 1

    for path in tmp_path.glob("training.sqlite3*"):
        path.unlink()
    replacement = initialize_database(tmp_path)
    assert isinstance(replacement, Ok)
    replaced = query_checkpoint_invalidation(tmp_path)
    assert isinstance(replaced, Ok)
    assert replaced.value.store_id is not None
    assert replaced.value.store_id != first.value.store_id
    assert replaced.value.through_sequence == 0
