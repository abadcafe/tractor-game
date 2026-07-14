"""Non-blocking batched writer for structured training events."""

from __future__ import annotations

import json
import math
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from server.foundation.json_value import JsonObject, JsonValue
from server.foundation.result import Rejected
from server.training_events.contract import (
    EVENT_NAMES,
    EventContext,
    EventName,
    ProcessIdentity,
)
from server.training_events.store import open_writer

_QUEUE_CAPACITY = 8192
_BATCH_SIZE = 256
_FLUSH_INTERVAL_SECONDS = 0.02


@dataclass(frozen=True, slots=True)
class _PendingEvent:
    event_type: EventName
    recorded_at_ms: int
    process: ProcessIdentity
    context: EventContext
    fields: JsonObject
    error: str | None


@dataclass(frozen=True, slots=True)
class _StopWriter:
    pass


type _QueueItem = _PendingEvent | _StopWriter


@dataclass(slots=True)
class _WriterState:
    run_dir: Path
    items: queue.Queue[_QueueItem] = field(
        default_factory=lambda: queue.Queue(maxsize=_QUEUE_CAPACITY)
    )
    dropped_count: int = 0
    dropped_lock: threading.Lock = field(default_factory=threading.Lock)
    thread: threading.Thread | None = None

    def start(self) -> None:
        assert self.thread is None
        self.thread = threading.Thread(
            target=self._run,
            name="training-event-writer",
            daemon=True,
        )
        self.thread.start()

    def offer(self, event: _PendingEvent) -> None:
        try:
            self.items.put_nowait(event)
        except queue.Full:
            self._add_dropped(1)

    def close(self) -> None:
        thread = self.thread
        if thread is None:
            return
        while thread.is_alive():
            try:
                self.items.put(_StopWriter(), timeout=0.1)
                break
            except queue.Full:
                continue
        thread.join()
        assert not thread.is_alive()
        self.thread = None

    def _run(self) -> None:
        opened = open_writer(self.run_dir)
        if isinstance(opened, Rejected):
            return
        connection = opened.value
        pending: list[_PendingEvent] = []
        stopping = False
        try:
            while not stopping:
                try:
                    item = self.items.get(
                        timeout=_FLUSH_INTERVAL_SECONDS
                    )
                except queue.Empty:
                    item = None
                if isinstance(item, _StopWriter):
                    stopping = True
                elif isinstance(item, _PendingEvent):
                    pending.append(item)
                while len(pending) < _BATCH_SIZE:
                    try:
                        queued = self.items.get_nowait()
                    except queue.Empty:
                        break
                    if isinstance(queued, _StopWriter):
                        stopping = True
                        break
                    pending.append(queued)
                if pending and (
                    stopping
                    or len(pending) >= _BATCH_SIZE
                    or item is None
                ):
                    self._write_batch(connection, pending)
                    pending.clear()
            if pending:
                self._write_batch(connection, pending)
        finally:
            connection.close()

    def _write_batch(
        self,
        connection: sqlite3.Connection,
        events: list[_PendingEvent],
    ) -> None:
        dropped = self._take_dropped()
        emitted = events
        if dropped > 0:
            reference = events[0]
            recorded_at_ms = time.time_ns() // 1_000_000
            emitted = [
                _PendingEvent(
                    event_type="logging.drop",
                    recorded_at_ms=recorded_at_ms,
                    process=reference.process,
                    context=EventContext(),
                    fields={"count": dropped},
                    error=None,
                ),
                *events,
            ]
        rows = [(_serialize_event(event),) for event in emitted]
        try:
            connection.executemany(
                "INSERT INTO training_logs(event_json) VALUES (?)", rows
            )
            connection.commit()
        except sqlite3.Error:
            connection.rollback()
            self._add_dropped(dropped + len(events))

    def _add_dropped(self, count: int) -> None:
        with self.dropped_lock:
            self.dropped_count += count

    def _take_dropped(self) -> int:
        with self.dropped_lock:
            count = self.dropped_count
            self.dropped_count = 0
            return count


_states_lock = threading.Lock()
_states: dict[tuple[int, Path], _WriterState] = {}


@dataclass(frozen=True, slots=True)
class StructuredEventSink:
    """Picklable producer facade with process-local writer ownership."""

    run_dir: Path
    process: ProcessIdentity

    def emit(
        self,
        event_type: EventName,
        *,
        fields: JsonObject | None = None,
        context: EventContext | None = None,
        error: str | None = None,
    ) -> None:
        assert event_type in EVENT_NAMES
        emitted_fields = _validate_event(
            fields=fields, context=context, error=error
        )
        state = _writer_state(self.run_dir)
        state.offer(
            _PendingEvent(
                event_type=event_type,
                recorded_at_ms=time.time_ns() // 1_000_000,
                process=self.process,
                context=context or EventContext(),
                fields=emitted_fields,
                error=error,
            )
        )

    def close(self) -> None:
        key = (os.getpid(), self.run_dir.resolve())
        with _states_lock:
            state = _states.pop(key, None)
        if state is not None:
            state.close()


class EventSink(Protocol):
    """Ancillary event boundary consumed by hot training code."""

    def emit(
        self,
        event_type: EventName,
        *,
        fields: JsonObject | None = None,
        context: EventContext | None = None,
        error: str | None = None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class NullEventSink:
    """Explicit no-observation sink for isolated unit tests."""

    def emit(
        self,
        event_type: EventName,
        *,
        fields: JsonObject | None = None,
        context: EventContext | None = None,
        error: str | None = None,
    ) -> None:
        assert event_type in EVENT_NAMES
        _validate_event(fields=fields, context=context, error=error)


def _writer_state(run_dir: Path) -> _WriterState:
    key = (os.getpid(), run_dir.resolve())
    with _states_lock:
        state = _states.get(key)
        if state is None:
            state = _WriterState(run_dir=key[1])
            state.start()
            _states[key] = state
        return state


def _serialize_event(event: _PendingEvent) -> str:
    payload: JsonObject = {
        "schema_version": 2,
        "event": event.event_type,
        "recorded_at_ms": event.recorded_at_ms,
        "process": {
            "kind": event.process.kind,
            "index": event.process.index,
            "pid": os.getpid(),
        },
        "context": _context_json(event.context),
        "fields": event.fields,
    }
    if event.error is not None:
        payload["error"] = event.error
    return json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _context_json(context: EventContext) -> JsonObject:
    values: tuple[tuple[str, JsonValue], ...] = (
        ("policy_version", context.policy_version),
        ("rollout_id", context.rollout_id),
        ("worker_index", context.worker_index),
        ("model_rank_index", context.model_rank_index),
        ("game_env_index", context.game_env_index),
        ("episode_id", context.episode_id),
        ("player_index", context.player_index),
        ("decision_index", context.decision_index),
        ("request_id", context.request_id),
        ("batch_id", context.batch_id),
    )
    return {key: value for key, value in values if value is not None}


def _assert_finite_json(value: JsonValue) -> None:
    if isinstance(value, float):
        assert math.isfinite(value)
        return
    if isinstance(value, list):
        for item in value:
            _assert_finite_json(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite_json(item)


def _validate_event(
    *,
    fields: JsonObject | None,
    context: EventContext | None,
    error: str | None,
) -> JsonObject:
    assert error is None or (error.strip() == error and error)
    if context is not None:
        _context_json(context)
    emitted_fields = fields or {}
    assert not {
        "reason",
        "error",
        "outcome",
        "level",
        "session_id",
    }.intersection(emitted_fields)
    _assert_finite_json(emitted_fields)
    return emitted_fields
