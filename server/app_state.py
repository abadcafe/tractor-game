"""Mutable process state owned by the ASGI server."""

from __future__ import annotations

from dataclasses import dataclass, field

from server.game import Game
from server.game_registry import GameRegistry
from server.player import HumanPlayer


def _human_player_dict() -> dict[str, HumanPlayer]:
    return {}


def _game_registry() -> GameRegistry[Game]:
    return GameRegistry()


@dataclass(slots=True)
class ServerState:
    registry: GameRegistry[Game] = field(default_factory=_game_registry)
    human_players: dict[str, HumanPlayer] = field(
        default_factory=_human_player_dict
    )

    async def cleanup_expired_games(
        self, *, max_age_seconds: int
    ) -> None:
        expired_ids = set(self.human_players.keys()) - set(
            game["game_id"] for game in self.registry.list_games()
        )
        for game_id in expired_ids:
            human = self.human_players.pop(game_id, None)
            if human is not None and human.is_connected():
                await human.close_ws()
        self.registry.cleanup_expired(max_age_seconds=max_age_seconds)
