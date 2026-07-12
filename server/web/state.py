"""Mutable process state owned by the ASGI server."""

from __future__ import annotations

from dataclasses import dataclass, field

from server.game.room.game_registry import GameRegistry
from server.game.room.game_room import GameRoom
from server.training_control.cli_client import TrainingCliClient
from server.training_control.config import (
    TrainingControlConfig,
    training_control_config,
)
from server.training_control.init_control import TrainingInitControl
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
    training_cli_client: TrainingCliClient = field(
        default_factory=TrainingCliClient
    )
    training_process_control: TrainingProcessControl = field(init=False)
    training_init_control: TrainingInitControl = field(
        default_factory=TrainingInitControl
    )

    def __post_init__(self) -> None:
        self.training_process_control = TrainingProcessControl(
            cli_client=self.training_cli_client
        )

    async def cleanup_expired_games(
        self, *, max_age_seconds: int
    ) -> None:
        self.registry.cleanup_expired(max_age_seconds=max_age_seconds)
