"""FastAPI lifespan management for server-owned background tasks."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI

from server.web.state import ServerState
from server.web.training_events.lifecycle import EventStreamLifecycle

_CLEANUP_INTERVAL_SECONDS = 300
_GAME_MAX_AGE_SECONDS = 3600
_LOGGER = logging.getLogger(__name__)


def lifespan_for(
    state: ServerState,
    event_stream_lifecycle: EventStreamLifecycle,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def lifespan(
        _app: FastAPI,
    ) -> AsyncGenerator[None, None]:
        async with asyncio.TaskGroup() as tasks:
            state.tasks.bind(tasks)
            cleanup = tasks.create_task(
                _cleanup_loop(state),
                name="server:game-cleanup",
            )
            _LOGGER.info("server.ready")
            try:
                yield
            finally:
                event_stream_lifecycle.close()
                _ = cleanup.cancel()
                await state.close()
                state.tasks.unbind()
                _LOGGER.info("server.stopped")

    return lifespan


async def _cleanup_loop(state: ServerState) -> None:
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
        await state.cleanup_expired_games(
            max_age_seconds=_GAME_MAX_AGE_SECONDS
        )
