"""Compose one game room and its player policies."""

from __future__ import annotations

import secrets

from server.game import GameConfig, GameSeed
from server.game_ai import AIService
from server.game_bots import DefaultBotPlayerFactory
from server.game_runtime import BotPolicyName, GameId, GameRoom
from server.web.tasks import ApplicationTasks

type GameInstance = GameRoom


def create_game_instance(
    game_id: GameId,
    ai_service: AIService,
    tasks: ApplicationTasks,
) -> GameInstance:
    """Create one independently seeded game instance."""
    return GameRoom(
        game_id=game_id,
        config=GameConfig(),
        seed=GameSeed(secrets.randbits(64)),
        bot_factory=DefaultBotPlayerFactory(ai_service),
        task_owner=tasks,
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
