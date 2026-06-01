"""Resilience module: session cleanup and server-side settings.

Provides functions for cleaning up expired game sessions and
managing server-side game settings.
"""

from __future__ import annotations

import time

from server.engine.game_state import GameSettings, Rank
from server.storage.game_store import GameStore

# ---- Server-side settings (module-level singleton) ----

_settings = GameSettings()


def cleanup_expired_sessions(store: GameStore, max_age_seconds: int = 3600) -> int:
    """Remove sessions older than max_age_seconds. Return count removed."""
    now = time.time()
    expired_ids = [
        gid for gid, ts in store._last_accessed.items()
        if now - ts > max_age_seconds
    ]
    for gid in expired_ids:
        try:
            store.delete(gid)
        except KeyError:
            pass  # already removed
    return len(expired_ids)


def get_settings() -> GameSettings:
    """Return the current server-side settings."""
    return _settings.model_copy(deep=True)


def update_settings(**kwargs: object) -> None:
    """Update server-side settings with provided keyword arguments."""
    global _settings
    data = _settings.model_dump()
    data.update(kwargs)
    _settings = GameSettings(**data)
