"""In-memory registry for independently owned game rooms."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import final

__all__ = ["GameRegistry"]


@final
class GameRegistry[ValueT]:
    """Store values by opaque ID and track last access."""

    def __init__(
        self,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._values: dict[str, ValueT] = {}
        self._last_access: dict[str, float] = {}
        self._clock = clock

    def create(self, value: ValueT) -> str:
        """Store a value under a new opaque ID."""
        value_id = uuid.uuid4().hex
        self._values[value_id] = value
        self._last_access[value_id] = self._clock()
        return value_id

    def get(self, value_id: str) -> ValueT | None:
        """Return a value and refresh its access time."""
        value = self._values.get(value_id)
        if value is not None:
            self._last_access[value_id] = self._clock()
        return value

    def delete(self, value_id: str) -> ValueT | None:
        """Remove and return a value if present."""
        value = self._values.pop(value_id, None)
        _ = self._last_access.pop(value_id, None)
        return value

    def list_ids(self) -> tuple[str, ...]:
        """Return all current IDs."""
        return tuple(self._values)

    def expire(self, max_idle_seconds: int) -> tuple[ValueT, ...]:
        """Remove and return entries older than the idle duration."""
        now = self._clock()
        expired = tuple(
            value_id
            for value_id, last_access in self._last_access.items()
            if now - last_access > max_idle_seconds
        )
        values = tuple(self._values[value_id] for value_id in expired)
        for value_id in expired:
            _ = self.delete(value_id)
        return values
