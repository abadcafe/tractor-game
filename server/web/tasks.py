"""Application ownership of dynamically created runtime tasks."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import final, override

from server.game_runtime import SessionTaskOwner


@final
class ApplicationTasks(SessionTaskOwner):
    """Expose one lifespan TaskGroup to composed runtime services."""

    def __init__(self) -> None:
        self._group: asyncio.TaskGroup | None = None

    def bind(self, group: asyncio.TaskGroup) -> None:
        """Bind the one entered TaskGroup for this application life."""
        assert self._group is None
        self._group = group

    def unbind(self) -> None:
        """Make task creation impossible after shutdown begins."""
        assert self._group is not None
        self._group = None

    @override
    def create_task[ResultT](
        self,
        coroutine: Coroutine[object, object, ResultT],
        *,
        name: str,
    ) -> asyncio.Task[ResultT]:
        """Start one named task under the application TaskGroup."""
        group = self._group
        assert group is not None
        return group.create_task(coroutine, name=name)


__all__ = ("ApplicationTasks",)
