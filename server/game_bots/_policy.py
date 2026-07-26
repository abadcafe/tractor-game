"""Strategic decision contract consumed by BotPlayer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from server.foundation.result import Ok, Rejected
from server.game import commands
from server.game_runtime.player import PlayerView

type StrategicAction = Literal["bid", "stir", "discard", "play"]
type DecisionCommand = (
    commands.RevealBid
    | commands.PassBid
    | commands.Stir
    | commands.PassStir
    | commands.Bury
    | commands.Play
)


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """One player view proven to require a strategic command."""

    view: PlayerView
    action: StrategicAction

    def __post_init__(self) -> None:
        assert self.view.snapshot.awaiting_action == self.action


class DecisionPolicy(Protocol):
    """Observe every state and decide only strategic actions."""

    def observe(self, view: PlayerView) -> Ok[None] | Rejected:
        """Consume one contiguous player-visible state."""
        ...

    async def decide(
        self,
        request: DecisionRequest,
    ) -> Ok[DecisionCommand] | Rejected:
        """Return one command for the requested strategic action."""
        ...


__all__ = (
    "DecisionCommand",
    "DecisionPolicy",
    "DecisionRequest",
    "StrategicAction",
)
