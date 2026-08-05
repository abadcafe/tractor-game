"""Composition root for the Tractor game web application."""

from __future__ import annotations

from pathlib import Path
from typing import final

from fastapi import FastAPI

from server.web.ai_api import register_ai_routes
from server.web.game_api import register_game_routes
from server.web.lifespan import lifespan_for
from server.web.state import ServerState
from server.web.static_assets import register_static_routes
from server.web.training_api import register_training_routes
from server.web.training_events import register_training_event_routes
from server.web.training_events.lifecycle import EventStreamLifecycle

__all__ = ["WebApplication"]


@final
class WebApplication:
    """Own one ASGI application and all of its process-local state."""

    __slots__ = ("_asgi", "_event_stream_lifecycle")

    def __init__(self) -> None:
        state = ServerState()
        event_stream_lifecycle = EventStreamLifecycle()
        asgi = FastAPI(
            lifespan=lifespan_for(state, event_stream_lifecycle)
        )
        static_dir = str(Path(__file__).resolve().parents[2] / "static")

        register_game_routes(asgi, state)
        register_ai_routes(asgi, state)
        register_training_routes(asgi, state)
        register_training_event_routes(
            asgi, state, event_stream_lifecycle
        )
        register_static_routes(asgi, static_dir)

        self._asgi = asgi
        self._event_stream_lifecycle = event_stream_lifecycle

    @property
    def asgi(self) -> FastAPI:
        """Return the composed ASGI application."""
        return self._asgi

    def begin_shutdown(self) -> None:
        """Stop every application-owned event stream."""
        self._event_stream_lifecycle.close()
