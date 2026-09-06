"""Strongly identified in-memory ownership of game instances."""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import final

_GAME_ID = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True, slots=True)
class GameId:
    """Canonical process-local identity for one game room."""

    value: str

    def __post_init__(self) -> None:
        assert _GAME_ID.fullmatch(self.value) is not None

    @classmethod
    def create(cls) -> GameId:
        """Generate one canonical opaque identity."""
        return cls(uuid.uuid4().hex)

    @classmethod
    def parse(cls, value: str | None) -> GameId | None:
        """Validate an identity received at an external boundary."""
        if value is None or _GAME_ID.fullmatch(value) is None:
            return None
        return cls(value)


@final
class GameRegistry[ValueT]:
    """Construct and own values under identities known at creation."""

    def __init__(
        self,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._values: dict[GameId, ValueT] = {}
        self._last_access: dict[GameId, float] = {}
        self._clock = clock

    def create(self, factory: Callable[[GameId], ValueT]) -> GameId:
        """Construct a value after allocating its immutable identity."""
        game_id = GameId.create()
        value = factory(game_id)
        self._values[game_id] = value
        self._last_access[game_id] = self._clock()
        return game_id

    def get(self, game_id: GameId) -> ValueT | None:
        """Return a value and refresh its access time."""
        value = self._values.get(game_id)
        if value is not None:
            self._last_access[game_id] = self._clock()
        return value

    def delete(self, game_id: GameId) -> ValueT | None:
        """Remove and return a value if present."""
        value = self._values.pop(game_id, None)
        _ = self._last_access.pop(game_id, None)
        return value

    def list_ids(self) -> tuple[GameId, ...]:
        """Return all current identities."""
        return tuple(self._values)

    def expire(self, max_idle_seconds: int) -> tuple[ValueT, ...]:
        """Remove and return entries older than the idle duration."""
        now = self._clock()
        expired = tuple(
            game_id
            for game_id, last_access in self._last_access.items()
            if now - last_access > max_idle_seconds
        )
        values = tuple(self._values[game_id] for game_id in expired)
        for game_id in expired:
            _ = self.delete(game_id)
        return values


__all__ = ("GameId", "GameRegistry")
