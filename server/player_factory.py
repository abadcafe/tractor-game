"""Factory for creating a playable server-side game instance."""

from __future__ import annotations

import asyncio
import os

from server.game import Game
from server.player import AIPlayer, AutoPlayer, HumanPlayer, Player

type _BotPlayer = AIPlayer | AutoPlayer


def create_game_with_default_players() -> tuple[Game, HumanPlayer]:
    bot_players: list[_BotPlayer] = [
        _create_bot_player(0),
        _create_bot_player(1),
        _create_bot_player(3),
    ]
    human = HumanPlayer(2)
    players: list[Player] = [
        bot_players[0],
        bot_players[1],
        human,
        bot_players[2],
    ]
    game = Game(players=players)
    for player in bot_players:
        asyncio.create_task(player.run(game))
    return game, human


def _create_bot_player(index: int) -> _BotPlayer:
    kind = os.environ.get("TRACTOR_BOT_PLAYER", "auto").strip().lower()
    if kind == "ai":
        return AIPlayer(index)
    return AutoPlayer(index)
