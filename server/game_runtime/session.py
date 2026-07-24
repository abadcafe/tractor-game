"""Single-event-loop runtime for one immutable game."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from server.foundation.result import Ok
from server.game import (
    CommandRejected,
    GameConfig,
    GameSeed,
    GameState,
    Seat,
    apply,
    commands,
    create,
    observe,
    snapshots,
)

__all__ = [
    "AgentSubmission",
    "CommandDecoder",
    "Delivery",
    "DeliverySink",
    "DisconnectReason",
    "Session",
]


@dataclass(frozen=True, slots=True)
class Delivery:
    """One sequenced player-specific state delivery."""

    viewer: Seat
    seq: int
    snapshot: snapshots.PlayerSnapshot
    error: str | None


class DisconnectReason(str, Enum):
    """Why a runtime connection is being closed."""

    REPLACED = "connection replaced"
    DETACHED = "player detached"
    SESSION_CLOSED = "game closed"


class DeliverySink(Protocol):
    """Consumer of player-specific runtime deliveries."""

    async def offer(self, delivery: Delivery) -> None:
        """Consume the latest state."""

    async def disconnect(self, reason: DisconnectReason) -> None:
        """Close the connection."""


class CommandDecoder(Protocol):
    """Decode an untrusted transport or agent payload."""

    def decode(
        self,
    ) -> Ok[commands.Command] | CommandRejected:
        """Return one typed game command."""
        ...


class AgentSubmission(Protocol):
    """The only capability an agent needs to submit commands."""

    async def submit(
        self,
        seq: int,
        decoder: CommandDecoder,
    ) -> None:
        """Submit a command guarded by the observed sequence."""


_SEATS: tuple[Seat, Seat, Seat, Seat] = (
    Seat.NORTH,
    Seat.WEST,
    Seat.SOUTH,
    Seat.EAST,
)


@dataclass(frozen=True, slots=True)
class _Connection:
    sink: DeliverySink


class Session:
    """Own sequence numbers and connected consumers around a game."""

    def __init__(self, config: GameConfig, seed: GameSeed) -> None:
        self._state: GameState = create(config, seed)
        self._seq = 1
        self._connections: dict[Seat, _Connection] = {}

    def snapshot(self, seat: Seat) -> Delivery:
        """Return the current delivery without changing sequence."""
        return Delivery(
            viewer=seat,
            seq=self._seq,
            snapshot=observe(self._state, seat),
            error=None,
        )

    def submission_for(self, seat: Seat) -> AgentSubmission:
        """Return a seat-bound submission capability."""
        return _SeatSubmission(self, seat)

    async def attach(
        self,
        seat: Seat,
        sink: DeliverySink,
    ) -> None:
        """Attach a sink, disconnecting an older owner if present."""
        current = self._connections.get(seat)
        if current is not None and current.sink is not sink:
            await current.sink.disconnect(DisconnectReason.REPLACED)
        self._connections[seat] = _Connection(sink=sink)

    def owns(self, seat: Seat, sink: DeliverySink) -> bool:
        """Return whether a sink still owns a seat connection."""
        current = self._connections.get(seat)
        return current is not None and current.sink is sink

    def detach(self, seat: Seat, sink: DeliverySink) -> None:
        """Detach only if the supplied sink is still current."""
        if self.owns(seat, sink):
            self._connections.pop(seat)

    async def disconnect(
        self,
        seat: Seat,
        reason: DisconnectReason,
    ) -> None:
        """Disconnect the current sink for one seat."""
        current = self._connections.pop(seat, None)
        if current is not None:
            await current.sink.disconnect(reason)

    def connected(self, seat: Seat) -> bool:
        """Return whether a seat currently has a delivery sink."""
        return seat in self._connections

    async def receive(
        self,
        seat: Seat,
        seq: int,
        decoder: CommandDecoder,
    ) -> None:
        """Receive one frame under the state sequence gate."""
        if seq == 0 or seq != self._seq:
            await self._send(seat, error=None)
            return
        decoded = decoder.decode()
        if isinstance(decoded, CommandRejected):
            await self._send(seat, error=decoded.reason)
            return
        result = apply(self._state, seat, decoded.value)
        if isinstance(result, CommandRejected):
            await self._send(seat, error=result.reason)
            return
        assert isinstance(result, Ok)
        self._state = result.value
        self._seq += 1
        await self._broadcast()

    async def close(self) -> None:
        """Disconnect every current sink."""
        connections = tuple(self._connections.values())
        self._connections.clear()
        for connection in connections:
            await connection.sink.disconnect(
                DisconnectReason.SESSION_CLOSED
            )

    async def _broadcast(self) -> None:
        for seat in _SEATS:
            if seat in self._connections:
                await self._send(seat, error=None)

    async def _send(
        self,
        seat: Seat,
        *,
        error: str | None,
    ) -> None:
        current = self._connections.get(seat)
        if current is None:
            return
        await current.sink.offer(
            Delivery(
                viewer=seat,
                seq=self._seq,
                snapshot=observe(self._state, seat),
                error=error,
            )
        )


class _SeatSubmission:
    """Seat-bound agent submission capability."""

    def __init__(self, session: Session, seat: Seat) -> None:
        self._session = session
        self._seat = seat

    async def submit(
        self,
        seq: int,
        decoder: CommandDecoder,
    ) -> None:
        await self._session.receive(self._seat, seq, decoder)
