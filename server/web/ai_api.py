"""Whole-controller HTTP endpoint for accelerated AI hosting."""

from __future__ import annotations

from fastapi import FastAPI

from server.game_ai import (
    RemoteDecisionRequest,
    RemoteDecisionResponse,
)
from server.web.state import ServerState


def register_ai_routes(app: FastAPI, state: ServerState) -> None:
    """Expose one idempotent model-controller decision route."""

    async def decide(
        request: RemoteDecisionRequest,
    ) -> RemoteDecisionResponse:
        return await state.ai_sessions.decide(request)

    app.add_api_route(
        "/api/ai/decision",
        decide,
        methods=["POST"],
        response_model=RemoteDecisionResponse,
    )


__all__ = ("register_ai_routes",)
