"""Training dashboard server-sent event adapters."""

from fastapi import FastAPI

from server.web.state import ServerState
from server.web.training_events.checkpoints import (
    register_checkpoints_route,
)
from server.web.training_events.logs import register_logs_route
from server.web.training_events.metrics import register_metrics_route
from server.web.training_events.process import register_process_route

__all__ = ["register_training_event_routes"]


def register_training_event_routes(
    app: FastAPI, state: ServerState
) -> None:
    """Register independent training dashboard event streams."""
    register_process_route(app, state)
    register_metrics_route(app, state)
    register_logs_route(app, state)
    register_checkpoints_route(app, state)
