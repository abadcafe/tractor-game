"""Compose one game room and its player policies."""

from __future__ import annotations

import secrets

from server.game import GameConfig, GameSeed
from server.game_ai import AIService
from server.game_bots import DefaultBotPlayerFactory
from server.game_runtime import BotPolicyName, GameRoom

type GameInstance = GameRoom


def create_game_instance(ai_service: AIService) -> GameInstance:
    """Create one independently seeded game instance."""
    return GameRoom(
        GameConfig(),
        GameSeed(secrets.randbits(64)),
        DefaultBotPlayerFactory(ai_service),
    )


def bot_policy_name_from_str(
    value: str | None,
) -> BotPolicyName | None:
    """Parse one closed bot-policy name."""
    normalized = (value or "").strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized == "ai":
        return "ai"
    return None


__all__ = (
    "GameInstance",
    "bot_policy_name_from_str",
    "create_game_instance",
)
