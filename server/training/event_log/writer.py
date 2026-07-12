"""Non-blocking batched writer for structured training events."""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from server.foundation.json_value import JsonObject, JsonValue
from server.foundation.result import Rejected
from server.training.persistence.schema import open_writer

type EventLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

_QUEUE_CAPACITY = 8192
_BATCH_SIZE = 256
_FLUSH_INTERVAL_SECONDS = 0.02


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """Stable process dimensions attached to every event."""

    kind: str
    index: int | None = None

    def __post_init__(self) -> None:
        assert self.kind
        assert self.index is None or self.index >= 0


@dataclass(frozen=True, slots=True)
class EventContext:
    """Correlation identifiers shared by related events."""

    policy_version: int | None = None
    rollout_id: str | None = None
    worker_index: int | None = None
    model_rank_index: int | None = None
    game_env_index: int | None = None
    episode_id: int | None = None
    player_index: int | None = None
    decision_index: int | None = None
    request_id: int | None = None
    batch_id: int | None = None


@dataclass(frozen=True, slots=True)
class _PendingEvent:
    event_type: str
    level: EventLevel
    recorded_at_ms: int
    session_id: str | None
    process: ProcessIdentity
    context: EventContext
    fields: JsonObject


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
        try:
            self.items.put(_StopWriter(), timeout=1.0)
        except queue.Full:
            self._add_dropped(1)
        thread.join(timeout=5.0)
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
                    event_type="logging.dropped",
                    level="WARNING",
                    recorded_at_ms=recorded_at_ms,
                    session_id=reference.session_id,
                    process=reference.process,
                    context=EventContext(),
                    fields={"count": dropped},
                ),
                _PendingEvent(
                    event_type="logging.recovered",
                    level="WARNING",
                    recorded_at_ms=recorded_at_ms,
                    session_id=reference.session_id,
                    process=reference.process,
                    context=EventContext(),
                    fields={"dropped_count": dropped},
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
    session_id: str | None
    process: ProcessIdentity

    def emit(
        self,
        event_type: str,
        *,
        fields: JsonObject | None = None,
        context: EventContext | None = None,
        level: EventLevel = "INFO",
    ) -> None:
        assert event_type
        state = _writer_state(self.run_dir)
        state.offer(
            _PendingEvent(
                event_type=event_type,
                level=level,
                recorded_at_ms=time.time_ns() // 1_000_000,
                session_id=self.session_id,
                process=self.process,
                context=context or EventContext(),
                fields=fields or {},
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

    @property
    def session_id(self) -> str | None: ...

    def emit(
        self,
        event_type: str,
        *,
        fields: JsonObject | None = None,
        context: EventContext | None = None,
        level: EventLevel = "INFO",
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class NullEventSink:
    """Explicit no-observation sink for isolated unit tests."""

    session_id: str | None = None

    def emit(
        self,
        event_type: str,
        *,
        fields: JsonObject | None = None,
        context: EventContext | None = None,
        level: EventLevel = "INFO",
    ) -> None:
        del event_type, fields, context, level


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
        "schema_version": 1,
        "event": event.event_type,
        "level": event.level,
        "recorded_at_ms": event.recorded_at_ms,
        "session_id": event.session_id,
        "process": {
            "kind": event.process.kind,
            "index": event.process.index,
            "pid": os.getpid(),
        },
        "context": _context_json(event.context),
        "fields": event.fields,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


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
    return {key: value for key, value in values}
