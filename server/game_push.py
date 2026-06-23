"""Player-facing state publication for a Game instance."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from server.player.base import GameView, Player
from server.protocol import StateMessage, StateSnapshot


class GameStatePublisher:
    """Owns state sequence numbers and pushes snapshots to players."""

    def __init__(
        self,
        players: Sequence[Player],
        owner: GameView,
        snapshot_for: Callable[[int], StateSnapshot],
    ) -> None:
        self._players = list(players)
        self._owner = owner
        self._snapshot_for = snapshot_for
        self._seq = 1

    def accepts_seq(self, seq: int) -> bool:
        return seq == self._seq

    def player(self, index: int) -> Player:
        return self._players[index]

    def state_message_for(
        self, player_index: int, error: str | None = None
    ) -> StateMessage:
        return StateMessage(
            seq=self._seq,
            state=self._snapshot_for(player_index),
            error=error,
        )

    async def send_to_player(
        self, player_index: int, error: str | None = None
    ) -> None:
        await self._players[player_index].on_state(
            self._owner,
            self.state_message_for(player_index, error=error),
        )

    async def push_to_all(self) -> None:
        self._seq += 1
        for player_index in range(len(self._players)):
            await self.send_to_player(player_index, error=None)
