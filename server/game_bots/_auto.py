"""Rule-driven automatic decision policy."""

from __future__ import annotations

import random
from typing import final

from server.foundation.result import Ok
from server.game_auto import choose_auto_command
from server.game_runtime.player import PlayerView

from ._policy import (
    DecisionCommand,
    DecisionRequest,
    DecisionUnavailable,
)


@final
class AutoPolicy:
    """Choose one rule-legal command without external services."""

    def __init__(self, random_source: random.Random) -> None:
        self._random = random_source

    def observe(self, view: PlayerView) -> None:
        """Accept every view without retaining policy state."""
        del view

    async def decide(
        self,
        request: DecisionRequest,
    ) -> Ok[DecisionCommand] | DecisionUnavailable:
        """Choose one legal strategic command."""
        return Ok(
            choose_auto_command(
                actor=request.view.viewer,
                snapshot=request.view.snapshot,
                random_source=self._random,
            )
        )


__all__ = ("AutoPolicy",)
