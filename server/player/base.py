"""Player abstraction for the Tractor game."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from server.protocol import PlayerMessage, StateMessage


class GameView(Protocol):
    """
    Protocol describing the Game interface that Player subclasses rely
    on.

    Players submit raw PlayerMessage envelopes. Game.receive() owns seq
    validation and action parsing.
    """

    async def receive(
        self,
        player_index: int,
        message: PlayerMessage,
    ) -> None: ...


class Player(ABC):
    """Abstract base class for game players.

    The game engine pushes StateMessage to each player via on_state().
    Subclasses submit follow-up PlayerMessage envelopes through
    GameView.
    """

    def __init__(self, index: int) -> None:
        self.index = index

    @abstractmethod
    async def on_state(
        self, game: GameView, message: StateMessage
    ) -> None:
        """Called by Game when it pushes state to this player."""
