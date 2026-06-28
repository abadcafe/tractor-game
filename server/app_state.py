"""Mutable process state owned by the ASGI server."""

from __future__ import annotations

from dataclasses import dataclass, field

from server.game_registry import GameRegistry
from server.game_room import GameRoom


def _game_registry() -> GameRegistry[GameRoom]:
    return GameRegistry()


@dataclass(slots=True)
class ServerState:
    registry: GameRegistry[GameRoom] = field(
        default_factory=_game_registry
    )

    async def cleanup_expired_games(
        self, *, max_age_seconds: int
    ) -> None:
        self.registry.cleanup_expired(max_age_seconds=max_age_seconds)
