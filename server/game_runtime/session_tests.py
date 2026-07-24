"""Black-box tests for sequenced game sessions."""

from __future__ import annotations

from dataclasses import dataclass, field

from server.foundation.result import Ok
from server.game import (
    CommandRejected,
    GameConfig,
    GameSeed,
    Seat,
    commands,
)
from server.game_runtime.session import (
    Delivery,
    DisconnectReason,
    Session,
)


def _deliveries() -> list[Delivery]:
    return []


def _disconnects() -> list[DisconnectReason]:
    return []


@dataclass(slots=True)
class _Sink:
    deliveries: list[Delivery] = field(default_factory=_deliveries)
    disconnects: list[DisconnectReason] = field(
        default_factory=_disconnects
    )

    async def offer(self, delivery: Delivery) -> None:
        self.deliveries.append(delivery)

    async def disconnect(
        self,
        reason: DisconnectReason,
    ) -> None:
        self.disconnects.append(reason)


@dataclass(slots=True)
class _Decoder:
    command: commands.Command
    decode_count: int = 0

    def decode(
        self,
    ) -> Ok[commands.Command] | CommandRejected:
        self.decode_count += 1
        return Ok(self.command)


@dataclass(slots=True)
class _RejectingDecoder:
    decode_count: int = 0

    def decode(
        self,
    ) -> Ok[commands.Command] | CommandRejected:
        self.decode_count += 1
        return CommandRejected("bad frame")


async def test_receive_zero_and_stale_return_current_state_lazily() -> (
    None
):
    session = Session(GameConfig(), GameSeed(3))
    sink = _Sink()
    decoder = _Decoder(commands.ConfirmRound())
    await session.attach(Seat.NORTH, sink)

    await session.receive(Seat.NORTH, 0, decoder)
    await session.receive(Seat.NORTH, 99, decoder)

    assert decoder.decode_count == 0
    assert [item.seq for item in sink.deliveries] == [1, 1]
    assert all(item.error is None for item in sink.deliveries)
    assert all(
        item.snapshot.awaiting_action == "next_round"
        for item in sink.deliveries
    )


async def test_receive_accepted_advances_sequence_exactly_once() -> (
    None
):
    session = Session(GameConfig(), GameSeed(5))
    sink = _Sink()
    await session.attach(Seat.NORTH, sink)
    accepted = _Decoder(commands.ConfirmRound())

    await session.receive(Seat.NORTH, 1, accepted)

    assert accepted.decode_count == 1
    assert len(sink.deliveries) == 1
    assert sink.deliveries[0].seq == 2
    assert sink.deliveries[0].error is None
    assert sink.deliveries[0].snapshot.awaiting_action is None
    assert session.snapshot(Seat.NORTH).seq == 2


async def test_receive_decode_rejection_keeps_sequence_and_state() -> (
    None
):
    session = Session(GameConfig(), GameSeed(7))
    sink = _Sink()
    await session.attach(Seat.NORTH, sink)
    decoder = _RejectingDecoder()

    await session.receive(Seat.NORTH, 1, decoder)

    assert decoder.decode_count == 1
    assert sink.deliveries[0].seq == 1
    assert sink.deliveries[0].error == "bad frame"
    assert sink.deliveries[0].snapshot.awaiting_action == "next_round"


async def test_receive_game_rejection_keeps_sequence() -> None:
    session = Session(GameConfig(), GameSeed(9))
    sink = _Sink()
    await session.attach(Seat.NORTH, sink)

    await session.receive(
        Seat.NORTH,
        1,
        _Decoder(commands.PassBid()),
    )

    assert sink.deliveries[0].seq == 1
    assert sink.deliveries[0].error is not None
    assert sink.deliveries[0].snapshot.awaiting_action == "next_round"


async def test_receive_broadcasts_personalized_observations() -> None:
    session = Session(GameConfig(), GameSeed(11))
    north = _Sink()
    west = _Sink()
    await session.attach(Seat.NORTH, north)
    await session.attach(Seat.WEST, west)

    await session.receive(
        Seat.NORTH,
        1,
        _Decoder(commands.ConfirmRound()),
    )

    assert len(north.deliveries) == 1
    assert len(west.deliveries) == 1
    assert north.deliveries[0].seq == west.deliveries[0].seq == 2
    assert north.deliveries[0].snapshot.awaiting_action is None
    assert west.deliveries[0].snapshot.awaiting_action == "next_round"


async def test_attach_replaces_only_the_same_seat_owner() -> None:
    session = Session(GameConfig(), GameSeed(13))
    first = _Sink()
    replacement = _Sink()
    other = _Sink()
    await session.attach(Seat.NORTH, first)
    await session.attach(Seat.WEST, other)

    await session.attach(Seat.NORTH, replacement)

    assert first.disconnects == [DisconnectReason.REPLACED]
    assert replacement.disconnects == []
    assert other.disconnects == []
    assert session.owns(Seat.NORTH, replacement)
    assert not session.owns(Seat.NORTH, first)


async def test_detach_ignores_stale_sink() -> None:
    session = Session(GameConfig(), GameSeed(15))
    stale = _Sink()
    current = _Sink()
    await session.attach(Seat.NORTH, stale)
    await session.attach(Seat.NORTH, current)

    session.detach(Seat.NORTH, stale)

    assert session.connected(Seat.NORTH)
    assert session.owns(Seat.NORTH, current)


async def test_close_disconnects_all_current_sinks_once() -> None:
    session = Session(GameConfig(), GameSeed(17))
    north = _Sink()
    west = _Sink()
    await session.attach(Seat.NORTH, north)
    await session.attach(Seat.WEST, west)

    await session.close()
    await session.close()

    assert north.disconnects == [DisconnectReason.SESSION_CLOSED]
    assert west.disconnects == [DisconnectReason.SESSION_CLOSED]
    assert not session.connected(Seat.NORTH)
    assert not session.connected(Seat.WEST)
