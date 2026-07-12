"""In-memory game registry with timeout cleanup."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import TypedDict


class ListedGame(TypedDict):
    game_id: str


class GameRegistry[GameT]:
    """In-memory storage and lifecycle management of Game objects.

    Provides create, get, delete, list_games, and cleanup_expired
    operations.
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._games: dict[str, GameT] = {}
        self._last_access: dict[str, float] = {}
        self._clock = clock

    def create(self, game: GameT) -> str:
        """Store a game and return its generated ID."""
        game_id = uuid.uuid4().hex
        self._games[game_id] = game
        self._last_access[game_id] = self._clock()
        return game_id

    def get(self, game_id: str) -> GameT | None:
        """Return the game for the given ID, or None if not found.

        Updates the last-access timestamp on hit.
        """
        game = self._games.get(game_id)
        if game is not None:
            self._last_access[game_id] = self._clock()
        return game

    def delete(self, game_id: str) -> None:
        """Remove a game and its timestamp; no-op if missing."""
        self._games.pop(game_id, None)
        self._last_access.pop(game_id, None)

    def list_games(self) -> list[ListedGame]:
        """Return stored game IDs."""
        return [{"game_id": gid} for gid in self._games]

    def cleanup_expired(self, max_age_seconds: int = 3600) -> int:
        """Remove games whose last access is older than max_age_seconds.

        Returns the count of removed games.
        """
        now = self._clock()
        expired_ids = [
            gid
            for gid, last in self._last_access.items()
            if now - last > max_age_seconds
        ]
        for gid in expired_ids:
            del self._games[gid]
            del self._last_access[gid]
        return len(expired_ids)
