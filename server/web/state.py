"""Mutable process state owned by the ASGI server."""

from __future__ import annotations

from dataclasses import dataclass, field

from server.game.room.game_registry import GameRegistry
from server.game.room.game_room import GameRoom
from server.training_control.config import (
    TrainingControlConfig,
    training_control_config,
)
from server.training_control.process_control import (
    TrainingProcessControl,
)


def _game_registry() -> GameRegistry[GameRoom]:
    return GameRegistry()


@dataclass(slots=True)
class ServerState:
    registry: GameRegistry[GameRoom] = field(
        default_factory=_game_registry
    )
    training_control_config: TrainingControlConfig = field(
        default_factory=training_control_config
    )
    training_process_control: TrainingProcessControl = field(init=False)

    def __post_init__(self) -> None:
        self.training_process_control = TrainingProcessControl(
            runtime_root=self.training_control_config.control_runtime_dir,
        )

    async def cleanup_expired_games(
        self, *, max_age_seconds: int
    ) -> None:
        self.registry.cleanup_expired(max_age_seconds=max_age_seconds)
