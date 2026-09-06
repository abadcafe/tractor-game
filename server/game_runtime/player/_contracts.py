"""Runtime contract shared by every player implementation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol

from server.foundation.result import Ok
from server.game import CommandRejected, Seat, commands, snapshots

from ..registry import GameId
from ._views import PlayerDescription, UserId

type PlayerRuntimeStatus = Literal["running", "failed"]


@dataclass(frozen=True, slots=True)
class PlayerView:
    """One complete sequenced view for a specific player."""

    viewer: Seat
    seq: int
    snapshot: snapshots.PlayerSnapshot
    status: PlayerRuntimeStatus
    error: str | None


@dataclass(frozen=True, slots=True)
class PlayerFailure:
    """External operational failure that prevents further commands."""

    error: str

    def __post_init__(self) -> None:
        assert self.error


class ConnectionCloseReason(str, Enum):
    """Why a human transport is being closed."""

    REPLACED = "connection replaced"
    SESSION_CLOSED = "game closed"


class CommandDecoder(Protocol):
    """Lazily decode one untrusted or already typed command."""

    def decode(
        self,
    ) -> Ok[commands.Command] | CommandRejected:
        """Return one typed game command."""
        ...


class PlayerInbox(Protocol):
    """Lossless stream of complete views owned by one Session."""

    async def receive(self) -> PlayerView | None:
        """Return the next view or ``None`` when the Session closes."""
        ...


class PlayerPort(Protocol):
    """Seat-bound game capability available to one player."""

    @property
    def game_id(self) -> GameId:
        """Return the immutable identity of the owning game."""
        ...

    async def player_ready(self) -> None:
        """Join the barrier after the player's inbox can receive."""
        ...

    async def player_initialized(self) -> None:
        """Publish that initial automatic work has settled."""
        ...

    async def request_view(self) -> None:
        """Enqueue a request for the current complete player view."""
        ...

    async def submit(
        self,
        seq: int,
        decoder: CommandDecoder,
    ) -> None:
        """Enqueue a command guarded by the observed sequence."""
        ...

    async def report_failure(self, failure: PlayerFailure) -> None:
        """Enqueue a persistent external player failure."""
        ...


class Player(Protocol):
    """Controller assigned to one seat by a running Session."""

    def lobby_status(
        self,
        requester: UserId | None,
    ) -> PlayerDescription:
        """Return the requester-specific lobby status."""
        ...

    async def run(
        self,
        port: PlayerPort,
        inbox: PlayerInbox,
    ) -> None:
        """Serve views until the Session closes the inbox."""
        ...


class HumanTransport(Protocol):
    """Transient transport connected to a persistent human player."""

    async def send(self, view: PlayerView) -> None:
        """Send one complete view to the remote human."""
        ...

    async def close(self, reason: ConnectionCloseReason) -> None:
        """Close the transport."""
        ...


__all__ = (
    "CommandDecoder",
    "ConnectionCloseReason",
    "HumanTransport",
    "Player",
    "PlayerFailure",
    "PlayerInbox",
    "PlayerPort",
    "PlayerRuntimeStatus",
    "PlayerView",
)
