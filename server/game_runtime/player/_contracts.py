"""Runtime contract shared by every player."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from server.foundation.result import Ok
from server.game import CommandRejected, Seat, commands, snapshots

from ._views import PlayerDescription, UserId


@dataclass(frozen=True, slots=True)
class PlayerView:
    """One complete sequenced view for a specific player."""

    viewer: Seat
    seq: int
    snapshot: snapshots.PlayerSnapshot
    error: str | None


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


class PlayerPort(Protocol):
    """Seat-bound game capability available to one player."""

    async def request_view(self) -> None:
        """Request the current complete player view."""
        ...

    async def submit(
        self,
        seq: int,
        decoder: CommandDecoder,
    ) -> None:
        """Submit a command guarded by the observed sequence."""
        ...


class Player(Protocol):
    """Controller assigned to one seat by a running session."""

    def lobby_status(
        self,
        requester: UserId | None,
    ) -> PlayerDescription:
        """Return the requester-specific lobby status."""
        ...

    async def start(self, port: PlayerPort) -> None:
        """Bind and fully initialize the player exactly once."""
        ...

    async def update(self, view: PlayerView) -> None:
        """Consume one complete latest-state view."""
        ...

    async def stop(self) -> None:
        """Stop the player and release all runtime resources."""
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
    "PlayerPort",
    "PlayerView",
)
