"""Mutable process state owned by the ASGI server."""

from __future__ import annotations

from dataclasses import dataclass, field

from server.game_ai import (
    AIService,
    RemoteSessionRegistry,
    ai_config_from_env,
)
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
    return AIService(ai_config_from_env())


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
    ai_sessions: RemoteSessionRegistry = field(init=False)

    def __post_init__(self) -> None:
        self.training_process_control = TrainingProcessControl()
        self.ai_sessions = RemoteSessionRegistry(self.ai_service)

    async def cleanup_expired_games(
        self, *, max_age_seconds: int
    ) -> None:
        rooms = self.registry.expire(max_idle_seconds=max_age_seconds)
        for room in rooms:
            await room.close()

    async def close(self) -> None:
        """Stop game tasks before closing their shared services."""
        for game_id in self.registry.list_ids():
            room = self.registry.delete(game_id)
            assert room is not None
            await room.close()
        self.ai_sessions.clear()
        await self.ai_service.close()
        await self.training_process_control.close()
