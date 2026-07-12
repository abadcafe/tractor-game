"""Strict SQLite schema and connection policy for training data."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from server.foundation import result as _result

DATABASE_FILENAME = "training.sqlite3"
APPLICATION_ID = 0x54524149
SCHEMA_VERSION = 1

_METRIC_COLUMNS = """
    total_games INTEGER NOT NULL CHECK (total_games >= 0),
    total_samples INTEGER NOT NULL CHECK (total_samples >= 0),
    total_updates INTEGER NOT NULL UNIQUE CHECK (total_updates >= 0),
    process_games_per_second REAL NOT NULL,
    process_samples_per_second REAL NOT NULL,
    last_round_decisions_per_second REAL NOT NULL,
    last_team0_reward REAL NOT NULL,
    last_team1_reward REAL NOT NULL,
    last_generated_action_count INTEGER NOT NULL CHECK (
        last_generated_action_count >= 0
    ),
    last_accepted_action_count INTEGER NOT NULL CHECK (
        last_accepted_action_count >= 0
    ),
    last_decision_count INTEGER NOT NULL CHECK (
        last_decision_count >= 0
    ),
    last_average_action_choices REAL NOT NULL,
    policy_loss REAL,
    value_loss REAL,
    entropy REAL,
    approx_kl REAL,
    clip_fraction REAL,
    ppo_update_seconds REAL CHECK (
        ppo_update_seconds IS NULL OR ppo_update_seconds >= 0
    ),
    ppo_minibatch_loss_seconds REAL CHECK (
        ppo_minibatch_loss_seconds IS NULL
        OR ppo_minibatch_loss_seconds >= 0
    ),
    ppo_observation_batch_seconds REAL CHECK (
        ppo_observation_batch_seconds IS NULL
        OR ppo_observation_batch_seconds >= 0
    ),
    ppo_observation_encode_seconds REAL CHECK (
        ppo_observation_encode_seconds IS NULL
        OR ppo_observation_encode_seconds >= 0
    ),
    ppo_value_head_seconds REAL CHECK (
        ppo_value_head_seconds IS NULL OR ppo_value_head_seconds >= 0
    ),
    ppo_argument_select_seconds REAL CHECK (
        ppo_argument_select_seconds IS NULL
        OR ppo_argument_select_seconds >= 0
    ),
    ppo_argument_decode_seconds REAL CHECK (
        ppo_argument_decode_seconds IS NULL
        OR ppo_argument_decode_seconds >= 0
    ),
    ppo_argument_distribution_seconds REAL CHECK (
        ppo_argument_distribution_seconds IS NULL
        OR ppo_argument_distribution_seconds >= 0
    ),
    ppo_backward_seconds REAL CHECK (
        ppo_backward_seconds IS NULL OR ppo_backward_seconds >= 0
    ),
    ppo_optimizer_step_seconds REAL CHECK (
        ppo_optimizer_step_seconds IS NULL
        OR ppo_optimizer_step_seconds >= 0
    ),
    ppo_argument_decode_fraction REAL CHECK (
        ppo_argument_decode_fraction IS NULL
        OR (
            ppo_argument_decode_fraction >= 0
            AND ppo_argument_decode_fraction <= 1
        )
    ),
    ppo_argument_trace_batch_count INTEGER CHECK (
        ppo_argument_trace_batch_count IS NULL
        OR ppo_argument_trace_batch_count >= 0
    ),
    ppo_argument_trace_row_count INTEGER CHECK (
        ppo_argument_trace_row_count IS NULL
        OR ppo_argument_trace_row_count >= 0
    ),
    ppo_argument_trace_token_count INTEGER CHECK (
        ppo_argument_trace_token_count IS NULL
        OR ppo_argument_trace_token_count >= 0
    ),
    ppo_argument_trace_valid_token_count INTEGER CHECK (
        ppo_argument_trace_valid_token_count IS NULL
        OR ppo_argument_trace_valid_token_count >= 0
    ),
    ppo_argument_trace_padding_token_count INTEGER CHECK (
        ppo_argument_trace_padding_token_count IS NULL
        OR ppo_argument_trace_padding_token_count >= 0
    ),
    checkpoint_path TEXT
"""

_CREATE_SCHEMA = f"""
CREATE TABLE training_metrics (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at_ms INTEGER NOT NULL CHECK (recorded_at_ms >= 0),
    {_METRIC_COLUMNS}
) STRICT;

CREATE TABLE runtime_telemetry (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at_ms INTEGER NOT NULL CHECK (recorded_at_ms >= 0),
    process_label TEXT NOT NULL CHECK (length(process_label) > 0),
    stage TEXT NOT NULL CHECK (stage IN (
        'coordinator', 'loading', 'rollout', 'inference', 'update',
        'checkpoint', 'idle', 'failed', 'completed', 'stopped'
    )),
    total_rounds INTEGER NOT NULL CHECK (total_rounds >= 0),
    total_updates INTEGER NOT NULL CHECK (total_updates >= 0),
    progress_numerator INTEGER NOT NULL CHECK (progress_numerator >= 0),
    progress_denominator INTEGER NOT NULL CHECK (
        progress_denominator >= 0
    ),
    measurements_json TEXT NOT NULL CHECK (
        json_valid(measurements_json)
        AND json_type(measurements_json) = 'object'
    ),
    CHECK (progress_numerator <= progress_denominator)
) STRICT;

CREATE INDEX runtime_telemetry_process_sequence
ON runtime_telemetry(process_label, sequence DESC);

CREATE INDEX runtime_telemetry_recorded_at
ON runtime_telemetry(recorded_at_ms);
"""


def database_path(run_dir: Path) -> Path:
    """Return the canonical SQLite path for one training task."""
    return run_dir / DATABASE_FILENAME


def initialize_database(
    run_dir: Path,
) -> _result.Ok[None] | _result.Rejected:
    """Create or strictly validate the training database."""
    path = database_path(run_dir)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=5.0)
    except sqlite3.Error, OSError:
        return _result.Rejected(
            reason=f"training database could not be opened: {path}"
        )
    try:
        _configure_writer(connection)
        application_id = _pragma_int(connection, "application_id")
        user_version = _pragma_int(connection, "user_version")
        tables = _table_count(connection)
        if tables == 0 and application_id == 0 and user_version == 0:
            connection.executescript(_CREATE_SCHEMA)
            connection.execute(
                f"PRAGMA application_id = {APPLICATION_ID}"
            )
            connection.execute(
                f"PRAGMA user_version = {SCHEMA_VERSION}"
            )
            connection.commit()
        elif (
            application_id != APPLICATION_ID
            or user_version != SCHEMA_VERSION
        ):
            return _result.Rejected(
                reason=f"unsupported training database schema: {path}"
            )
    except sqlite3.Error:
        return _result.Rejected(
            reason=f"training database schema is invalid: {path}"
        )
    finally:
        connection.close()
    return _result.Ok(value=None)


def open_writer(
    run_dir: Path,
) -> _result.Ok[sqlite3.Connection] | _result.Rejected:
    """Open a configured writer after strict schema validation."""
    initialized = initialize_database(run_dir)
    if isinstance(initialized, _result.Rejected):
        return initialized
    path = database_path(run_dir)
    try:
        connection = sqlite3.connect(path, timeout=5.0)
        _configure_writer(connection)
    except sqlite3.Error:
        return _result.Rejected(
            reason=f"training database could not be opened: {path}"
        )
    return _result.Ok(value=connection)


def open_reader(
    run_dir: Path,
) -> _result.Ok[sqlite3.Connection | None] | _result.Rejected:
    """Open an existing database in read-only/query-only mode."""
    path = database_path(run_dir)
    if not path.exists():
        return _result.Ok(value=None)
    try:
        connection = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            timeout=5.0,
        )
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        application_id = _pragma_int(connection, "application_id")
        user_version = _pragma_int(connection, "user_version")
    except sqlite3.Error:
        return _result.Rejected(
            reason=f"training database is unreadable: {path}"
        )
    if (
        application_id != APPLICATION_ID
        or user_version != SCHEMA_VERSION
    ):
        connection.close()
        return _result.Rejected(
            reason=f"unsupported training database schema: {path}"
        )
    return _result.Ok(value=connection)


def _configure_writer(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA trusted_schema = OFF")


def _pragma_int(connection: sqlite3.Connection, name: str) -> int:
    row = connection.execute(f"PRAGMA {name}").fetchone()
    assert row is not None
    value = row[0]
    assert isinstance(value, int)
    return value


def _table_count(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT count(*) FROM sqlite_schema WHERE type = 'table' "
        "AND name NOT LIKE 'sqlite_%'"
    ).fetchone()
    assert row is not None
    value = row[0]
    assert isinstance(value, int)
    return value
