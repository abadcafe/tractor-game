"""Store-aware invalidation cursor for checkpoint artifacts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from server.foundation import result as _result
from server.training_events.store import open_reader, training_store_id


class CheckpointInvalidation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    store_id: str | None
    through_sequence: int = Field(ge=0)


def query_checkpoint_invalidation(
    run_dir: Path,
) -> _result.Ok[CheckpointInvalidation] | _result.Rejected:
    """Read the newest event that can change checkpoint artifacts."""
    opened = open_reader(run_dir)
    if isinstance(opened, _result.Rejected):
        return opened
    connection = opened.value
    if connection is None:
        return _result.Ok(
            value=CheckpointInvalidation(
                store_id=None, through_sequence=0
            )
        )
    try:
        store_id = training_store_id(connection)
        row = connection.execute(
            "SELECT coalesce(max(sequence), 0) FROM training_logs "
            "WHERE event_type IN ('initialize', 'checkpoint')"
        ).fetchone()
    except sqlite3.Error:
        return _result.Rejected(
            reason="checkpoint invalidation query failed"
        )
    finally:
        connection.close()
    if row is None or not isinstance(row[0], int):
        return _result.Rejected(
            reason="checkpoint invalidation query failed"
        )
    return _result.Ok(
        value=CheckpointInvalidation(
            store_id=store_id, through_sequence=row[0]
        )
    )
