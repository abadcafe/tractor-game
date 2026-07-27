"""Mutable process state owned by the ASGI server."""

from __future__ import annotations

from dataclasses import dataclass, field

from server.game_ai import AIConfig, AIService
from server.game_runtime.registry import GameRegistry
from server.training_control.config import (
    TrainingControlConfig,
    training_control_config,
)
from server.training_control.process_control import (
    TrainingProcessControl,
)
from server.web.game_composition import GameInstance


def _game_registry() -> GameRegistry[GameInstance]:
    return GameRegistry()


def _ai_service() -> AIService:
    return AIService(AIConfig.from_env())


@dataclass(slots=True)
class ServerState:
    registry: GameRegistry[GameInstance] = field(
        default_factory=_game_registry
    )
    training_control_config: TrainingControlConfig = field(
        default_factory=training_control_config
    )
    ai_service: AIService = field(default_factory=_ai_service)
    training_process_control: TrainingProcessControl = field(init=False)

    def __post_init__(self) -> None:
        self.training_process_control = TrainingProcessControl()

    async def cleanup_expired_games(
        self, *, max_age_seconds: int
    ) -> None:
        rooms = self.registry.expire(max_idle_seconds=max_age_seconds)
        for room in rooms:
            await room.close()
