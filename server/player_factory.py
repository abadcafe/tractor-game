"""Factory for creating a playable server-side game room."""

from __future__ import annotations

from server.game_room import GameRoom


def create_game_room() -> GameRoom:
    return GameRoom()
