"""Compose game rooms from runtime and agent modules."""

from __future__ import annotations

import os
import secrets

from server.game import GameConfig, GameSeed
from server.game_agents.factory import DefaultAgentFactory
from server.game_runtime.room import BotKind, GameRoom


def create_game_room() -> GameRoom:
    """Create one independently seeded game room."""
    return GameRoom(
        GameConfig(),
        GameSeed(secrets.randbits(64)),
        DefaultAgentFactory(),
    )


def bot_kind_from_env() -> BotKind:
    """Read the default bot kind from the environment."""
    return (
        bot_kind_from_str(os.environ.get("TRACTOR_BOT_PLAYER", "auto"))
        or BotKind.AUTO
    )


def bot_kind_from_str(value: str | None) -> BotKind | None:
    """Parse a wire bot-kind value."""
    normalized = (value or "").strip().lower()
    if normalized == "ai":
        return BotKind.AI
    if normalized == "auto":
        return BotKind.AUTO
    return None
