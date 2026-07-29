"""Trained-model strategic decision policy."""

from __future__ import annotations

from server.foundation.result import Ok, Rejected
from server.game import commands
from server.game_ai import AIControllerPort
from server.game_runtime.player import PlayerView

from ._policy import DecisionCommand, DecisionRequest


class AIPolicy:
    """Drive a seat with one model-backed decision controller."""

    def __init__(self, controller: AIControllerPort) -> None:
        self._controller = controller

    def observe(self, view: PlayerView) -> Ok[None] | Rejected:
        """Record every complete view before any decision."""
        return self._controller.observe(
            seq=view.seq,
            snapshot=view.snapshot,
            error=view.error,
        )

    async def decide(
        self,
        request: DecisionRequest,
    ) -> Ok[DecisionCommand] | Rejected:
        """Await the model runtime without blocking the event loop."""
        result = await self._controller.decide(
            seq=request.view.seq,
            snapshot=request.view.snapshot,
        )
        if isinstance(result, Rejected):
            return result
        command = result.value
        assert isinstance(
            command,
            (
                commands.RevealBid,
                commands.PassBid,
                commands.Stir,
                commands.PassStir,
                commands.Bury,
                commands.Play,
            ),
        )
        return Ok(command)


__all__ = ("AIPolicy",)
