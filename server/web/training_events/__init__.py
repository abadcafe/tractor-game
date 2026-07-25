"""Training dashboard server-sent event adapters."""

from fastapi import FastAPI

from server.web.state import ServerState
from server.web.training_events.checkpoints import (
    register_checkpoints_route,
)
from server.web.training_events.lifecycle import EventStreamLifecycle
from server.web.training_events.logs import register_logs_route
from server.web.training_events.metrics import register_metrics_route
from server.web.training_events.process import register_process_route

__all__ = [
    "EventStreamLifecycle",
    "register_training_event_routes",
]


def register_training_event_routes(
    app: FastAPI,
    state: ServerState,
    lifecycle: EventStreamLifecycle,
) -> None:
    """Register independent training dashboard event streams."""
    register_process_route(app, state, lifecycle)
    register_metrics_route(app, state, lifecycle)
    register_logs_route(app, state, lifecycle)
    register_checkpoints_route(app, state, lifecycle)
