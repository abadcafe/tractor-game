"""Human player with one replaceable network transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from ._contracts import (
    CommandDecoder,
    ConnectionCloseReason,
    HumanTransport,
    PlayerInbox,
    PlayerPort,
)
from ._views import PlayerDescription, UserId


@dataclass(frozen=True, slots=True)
class _NotRunning:
    pass


@dataclass(frozen=True, slots=True)
class _Running:
    port: PlayerPort


type _Lifecycle = _NotRunning | _Running


@dataclass(frozen=True, slots=True)
class _Disconnected:
    pass


@dataclass(frozen=True, slots=True)
class _Connected:
    transport: HumanTransport


type _Connection = _Disconnected | _Connected


@final
class HumanPlayer:
    """Human seat controller surviving transport reconnects."""

    def __init__(self, user_id: UserId) -> None:
        self._user_id = user_id
        self._lifecycle: _Lifecycle = _NotRunning()
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

    async def run(
        self,
        port: PlayerPort,
        inbox: PlayerInbox,
    ) -> None:
        """Forward Session-owned views to the active transport."""
        assert isinstance(self._lifecycle, _NotRunning)
        self._lifecycle = _Running(port)
        await port.player_ready()
        await port.player_initialized()
        try:
            while True:
                view = await inbox.receive()
                if view is None:
                    return
                connection = self._connection
                if isinstance(connection, _Connected):
                    await connection.transport.send(view)
        finally:
            self._lifecycle = _NotRunning()
            connection = self._connection
            self._connection = _Disconnected()
            if isinstance(connection, _Connected):
                await connection.transport.close(
                    ConnectionCloseReason.SESSION_CLOSED
                )

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


__all__ = ("HumanPlayer",)
