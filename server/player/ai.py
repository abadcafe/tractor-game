"""AI player implementation."""

from __future__ import annotations

from server.messages import StateMessage
from server.player.base import GameView, Player


class AIPlayer(Player):
    """AI player type.

    This class is intentionally separate from AutoPlayer so future LLM decision
    logic does not couple to the built-in automatic player's random strategy.
    """

    async def on_state(self, game: GameView, message: StateMessage) -> None:
        """Handle a player-facing state push.

        Real LLM decision logic will be added here. For now this player is a
        type boundary only and therefore does not act.
        """
