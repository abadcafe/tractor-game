"""FastAPI lifespan management for server-owned background tasks."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI

from server.web.state import ServerState

_CLEANUP_INTERVAL_SECONDS = 300
_GAME_MAX_AGE_SECONDS = 3600


def lifespan_for(
    state: ServerState,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def lifespan(
        _app: FastAPI,
    ) -> AsyncGenerator[None, None]:
        task = asyncio.create_task(_cleanup_loop(state))
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    return lifespan


async def _cleanup_loop(state: ServerState) -> None:
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
        await state.cleanup_expired_games(
            max_age_seconds=_GAME_MAX_AGE_SECONDS
        )
