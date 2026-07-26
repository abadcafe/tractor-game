"""Persistent human player with a replaceable transport."""

from __future__ import annotations

from dataclasses import dataclass

from ._contracts import (
    CommandDecoder,
    ConnectionCloseReason,
    HumanTransport,
    PlayerPort,
    PlayerView,
)
from ._views import PlayerDescription, UserId


@dataclass(frozen=True, slots=True)
class _New:
    pass


@dataclass(frozen=True, slots=True)
class _Running:
    port: PlayerPort


@dataclass(frozen=True, slots=True)
class _Stopped:
    pass


type _Lifecycle = _New | _Running | _Stopped


@dataclass(frozen=True, slots=True)
class _Disconnected:
    pass


@dataclass(frozen=True, slots=True)
class _Connected:
    transport: HumanTransport


type _Connection = _Disconnected | _Connected


class HumanPlayer:
    """Human seat controller surviving transport reconnects."""

    def __init__(self, user_id: UserId) -> None:
        self._user_id = user_id
        self._lifecycle: _Lifecycle = _New()
        self._connection: _Connection = _Disconnected()

    @property
    def user_id(self) -> UserId:
        """Return the immutable room owner identity."""
        return self._user_id

    def lobby_status(
        self,
        requester: UserId | None,
    ) -> PlayerDescription:
        """Return a human-only lobby projection."""
        return {
            "kind": "human",
            "connected": isinstance(self._connection, _Connected),
            "mine": requester == self._user_id,
        }

    async def start(self, port: PlayerPort) -> None:
        """Bind the human to the session exactly once."""
        assert isinstance(self._lifecycle, _New)
        self._lifecycle = _Running(port)

    async def connect(self, transport: HumanTransport) -> None:
        """Replace the transport without replacing the player."""
        connection = self._connection
        if isinstance(connection, _Connected):
            if connection.transport is transport:
                return
            await connection.transport.close(
                ConnectionCloseReason.REPLACED
            )
        self._connection = _Connected(transport)

    def disconnect(self, transport: HumanTransport) -> None:
        """Detach only the currently connected transport."""
        connection = self._connection
        if (
            isinstance(connection, _Connected)
            and connection.transport is transport
        ):
            self._connection = _Disconnected()

    async def receive(
        self,
        transport: HumanTransport,
        seq: int,
        decoder: CommandDecoder,
    ) -> None:
        """Forward a frame only from the current transport."""
        connection = self._connection
        if (
            not isinstance(connection, _Connected)
            or connection.transport is not transport
        ):
            return
        lifecycle = self._lifecycle
        assert isinstance(lifecycle, _Running)
        if seq == 0:
            await lifecycle.port.request_view()
            return
        await lifecycle.port.submit(seq, decoder)

    async def update(self, view: PlayerView) -> None:
        """Forward state only when a transport is connected."""
        assert isinstance(self._lifecycle, _Running)
        connection = self._connection
        if isinstance(connection, _Connected):
            await connection.transport.send(view)

    async def stop(self) -> None:
        """Stop the player and close its active transport once."""
        if isinstance(self._lifecycle, _Stopped):
            return
        self._lifecycle = _Stopped()
        connection = self._connection
        self._connection = _Disconnected()
        if isinstance(connection, _Connected):
            await connection.transport.close(
                ConnectionCloseReason.SESSION_CLOSED
            )


__all__ = ("HumanPlayer",)
