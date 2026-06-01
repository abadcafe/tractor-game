"""In-memory game state storage.

Provides GameStore class for persisting and retrieving GameState objects
by game ID.  Replaces the TypeScript localStorage approach.
"""

from __future__ import annotations

import time
import uuid

from server.engine.game_state import GameState


class GameStore:
    """In-memory store for game state objects keyed by game ID."""

    def __init__(self) -> None:
        self._store: dict[str, GameState] = {}
        self._last_accessed: dict[str, float] = {}

    def create(self, state: GameState) -> str:
        """Store a new game state and return its generated ID."""
        game_id = uuid.uuid4().hex
        self._store[game_id] = state.model_copy(deep=True)
        self._last_accessed[game_id] = time.time()
        return game_id

    def get(self, game_id: str) -> GameState | None:
        """Retrieve a game state by ID, or None if not found."""
        state = self._store.get(game_id)
        if state is not None:
            self._last_accessed[game_id] = time.time()
            return state.model_copy(deep=True)
        return None

    def update(self, game_id: str, state: GameState) -> None:
        """Update an existing game state. Raises KeyError if not found."""
        if game_id not in self._store:
            raise KeyError(game_id)
        self._store[game_id] = state.model_copy(deep=True)
        self._last_accessed[game_id] = time.time()

    def delete(self, game_id: str) -> None:
        """Remove a game state by ID. Raises KeyError if not found."""
        if game_id not in self._store:
            raise KeyError(game_id)
        del self._store[game_id]
        self._last_accessed.pop(game_id, None)

    def list_games(self) -> list[str]:
        """Return all stored game IDs."""
        return list(self._store.keys())

    def get_expired_ids(self, max_age_seconds: int) -> list[str]:
        """Return game IDs whose last access is older than max_age_seconds."""
        now = time.time()
        return [
            gid for gid, ts in self._last_accessed.items()
            if now - ts > max_age_seconds
        ]
