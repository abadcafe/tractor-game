"""ASGI app assembly for the Tractor game server."""

from __future__ import annotations

import os

from fastapi import FastAPI

from server.web.ai_debug_api import register_ai_debug_routes
from server.web.game_api import register_game_routes
from server.web.lifespan import lifespan_for
from server.web.logging_config import configure_server_logging
from server.web.state import ServerState
from server.web.static_assets import register_static_routes
from server.web.training_api import register_training_routes
from server.web.training_stream import register_training_stream_route

configure_server_logging()

state = ServerState()
registry = state.registry
static_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "static")
)

app = FastAPI(lifespan=lifespan_for(state))
register_game_routes(app, state)
register_ai_debug_routes(app, state, static_dir)
register_training_routes(app, state)
register_training_stream_route(app, state)
register_static_routes(app, static_dir)
