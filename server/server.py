"""ASGI app assembly for the Tractor game server."""

from __future__ import annotations

import os

from fastapi import FastAPI

from server.app_state import ServerState
from server.lifespan import lifespan_for
from server.logging_config import configure_server_logging
from server.routes.ai_debug import register_ai_debug_routes
from server.routes.game_api import register_game_routes
from server.routes.static_files import register_static_routes

configure_server_logging()

state = ServerState()
registry = state.registry
human_players = state.human_players
static_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "static")
)

app = FastAPI(lifespan=lifespan_for(state))
register_game_routes(app, state)
register_ai_debug_routes(app, state, static_dir)
register_static_routes(app, static_dir)
