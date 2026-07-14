"""Structured training event persistence."""

from server.training_events.contract import (
    EVENT_NAMES,
    PROCESS_KINDS,
    EventContext,
    EventName,
    ProcessIdentity,
    ProcessKind,
)
from server.training_events.queries import (
    TrainingLogHistoryPage,
    TrainingLogRecord,
    TrainingLogTail,
    query_training_log_history,
    query_training_log_tail,
)
from server.training_events.writer import (
    EventSink,
    NullEventSink,
    StructuredEventSink,
)

__all__ = [
    "EventContext",
    "EVENT_NAMES",
    "EventName",
    "EventSink",
    "NullEventSink",
    "ProcessIdentity",
    "PROCESS_KINDS",
    "ProcessKind",
    "StructuredEventSink",
    "TrainingLogHistoryPage",
    "TrainingLogRecord",
    "TrainingLogTail",
    "query_training_log_history",
    "query_training_log_tail",
]
