"""Black-box tests for sequenced game-room runtime behavior."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field

from server.foundation.result import Ok, Rejected
from server.game import (
    CommandRejected,
    GameConfig,
    GameSeed,
    Seat,
    commands,
)
from server.game_runtime import (
    BotPolicyName,
    GameId,
    GameRoom,
    UserId,
)
from server.game_runtime.player import (
    ConnectionCloseReason,
    Player,
    PlayerDescription,
    PlayerFailure,
    PlayerInbox,
    PlayerPort,
    PlayerView,
)


def _views() -> list[PlayerView]:
    return []


def _tasks() -> list[asyncio.Task[None]]:
    return []


async def _wait_until(condition: Callable[[], bool]) -> None:
    for _index in range(100):
        if condition():
            return
        await asyncio.sleep(0)
    assert condition()


@dataclass(slots=True)
class _TaskOwner:
    tasks: list[asyncio.Task[None]] = field(default_factory=_tasks)

    def create_task(
        self,
        coroutine: Coroutine[object, object, None],
        *,
        name: str,
    ) -> asyncio.Task[None]:
        task = asyncio.create_task(coroutine, name=name)
        self.tasks.append(task)
        return task


@dataclass(slots=True)
class _Player:
    port: PlayerPort | None = None
    views: list[PlayerView] = field(default_factory=_views)
    stop_count: int = 0

    def lobby_status(
        self,
        requester: UserId | None,
    ) -> PlayerDescription:
        del requester
        return {"kind": "bot", "policy": "auto"}

    async def run(
        self,
        port: PlayerPort,
        inbox: PlayerInbox,
    ) -> None:
        assert self.port is None
        self.port = port
        await port.player_ready()
        await port.player_initialized()
        try:
            while True:
                view = await inbox.receive()
                if view is None:
                    return
                self.views.append(view)
        finally:
            self.stop_count += 1

    def started_port(self) -> PlayerPort:
        assert self.port is not None
        return self.port


def _players() -> dict[Seat, _Player]:
    return {}


@dataclass(slots=True)
class _Factory:
    players: dict[Seat, _Player] = field(default_factory=_players)

    def create(
        self,
        seat: Seat,
        policy_name: BotPolicyName,
    ) -> Ok[Player] | Rejected:
        assert policy_name == "auto"
        player = _Player()
        self.players[seat] = player
        return Ok(player)


@dataclass(slots=True)
class _Transport:
    views: list[PlayerView] = field(default_factory=_views)
    close_count: int = 0

    async def send(self, view: PlayerView) -> None:
        self.views.append(view)

    async def close(self, reason: ConnectionCloseReason) -> None:
        assert reason == ConnectionCloseReason.SESSION_CLOSED
        self.close_count += 1


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


async def _started_room() -> tuple[
    GameRoom,
    UserId,
    _Transport,
    _Factory,
]:
    factory = _Factory()
    room = GameRoom(
        game_id=GameId("1" * 32),
        config=GameConfig(),
        seed=GameSeed(3),
        bot_factory=factory,
        task_owner=_TaskOwner(),
    )
    user_id = UserId("owner")
    assert isinstance(
        room.occupy_seat(seat=Seat.A, user_id=user_id),
        Ok,
    )
    assert isinstance(
        room.fill_bots(policy="auto", user_id=user_id),
        Ok,
    )
    transport = _Transport()
    connected = await room.connect_seat(
        seat=Seat.A,
        user_id=user_id,
        transport=transport,
    )
    assert isinstance(connected, Ok)
    return room, user_id, transport, factory


async def test_start_binds_every_bot_without_unsolicited_view() -> None:
    room, _user_id, transport, factory = await _started_room()

    assert len(factory.players) == 3
    assert all(
        player.port is not None for player in factory.players.values()
    )
    assert all(not player.views for player in factory.players.values())
    assert transport.views == []
    await room.close()


async def test_request_view_returns_complete_current_view() -> None:
    room, _user_id, _transport, factory = await _started_room()
    player = factory.players[Seat.B]

    await player.started_port().request_view()
    await _wait_until(lambda: len(player.views) == 1)

    assert len(player.views) == 1
    view = player.views[0]
    assert view.viewer == Seat.B
    assert view.seq == 1
    assert view.status == "running"
    assert view.error is None
    assert view.snapshot.awaiting_action == "next_round"
    await room.close()


async def test_stale_submission_skips_decode_and_returns_view() -> None:
    room, _user_id, _transport, factory = await _started_room()
    decoder = _Decoder(commands.ConfirmRound())
    player = factory.players[Seat.B]

    await player.started_port().submit(99, decoder)
    await _wait_until(lambda: len(player.views) == 1)

    assert decoder.decode_count == 0
    assert [view.seq for view in player.views] == [1]
    assert player.views[0].error is None
    await room.close()


async def test_accepted_submission_broadcasts_personalized_views() -> (
    None
):
    room, _user_id, transport, factory = await _started_room()
    decoder = _Decoder(commands.ConfirmRound())

    await factory.players[Seat.B].started_port().submit(1, decoder)
    await _wait_until(
        lambda: (
            decoder.decode_count == 1
            and len(transport.views) == 1
            and all(
                len(player.views) == 1
                for player in factory.players.values()
            )
        )
    )

    assert decoder.decode_count == 1
    assert [view.seq for view in transport.views] == [2]
    assert all(
        [view.seq for view in player.views] == [2]
        for player in factory.players.values()
    )
    assert (
        factory.players[Seat.B].views[0].snapshot.awaiting_action
        is None
    )
    assert (
        factory.players[Seat.C].views[0].snapshot.awaiting_action
        == "next_round"
    )
    await room.close()


async def test_decode_rejection_keeps_sequence_and_state() -> None:
    room, _user_id, _transport, factory = await _started_room()
    decoder = _RejectingDecoder()
    player = factory.players[Seat.B]

    await player.started_port().submit(1, decoder)
    await _wait_until(lambda: len(player.views) == 1)

    assert decoder.decode_count == 1
    assert len(player.views) == 1
    view = player.views[0]
    assert view.seq == 1
    assert view.error == "bad frame"
    assert view.snapshot.awaiting_action == "next_round"
    assert all(
        not candidate.views
        for seat, candidate in factory.players.items()
        if seat != Seat.B
    )
    await room.close()


async def test_game_rejection_keeps_sequence() -> None:
    room, _user_id, _transport, factory = await _started_room()
    player = factory.players[Seat.B]

    await player.started_port().submit(1, _Decoder(commands.PassBid()))
    await _wait_until(lambda: len(player.views) == 1)

    view = player.views[0]
    assert view.seq == 1
    assert view.error is not None
    assert view.snapshot.awaiting_action == "next_round"
    await room.close()


async def test_report_failure_persists_complete_failed_views() -> None:
    room, _user_id, transport, factory = await _started_room()
    player = factory.players[Seat.B]

    await player.started_port().report_failure(
        PlayerFailure(error="remote model unavailable")
    )
    await _wait_until(
        lambda: (
            len(transport.views) == 1
            and all(
                len(candidate.views) == 1
                for candidate in factory.players.values()
            )
        )
    )

    assert transport.views[-1].status == "failed"
    assert transport.views[-1].error == "remote model unavailable"
    assert all(
        candidate.views[-1].status == "failed"
        for candidate in factory.players.values()
    )
    await player.started_port().submit(
        1,
        _Decoder(commands.ConfirmRound()),
    )
    await _wait_until(lambda: len(player.views) == 2)
    assert player.views[-1].seq == 1
    assert player.views[-1].status == "failed"
    await room.close()


async def test_close_stops_players_and_disables_session() -> None:
    room, _user_id, transport, factory = await _started_room()

    await room.close()
    await room.close()

    assert transport.close_count == 1
    assert all(
        candidate.stop_count == 1
        for candidate in factory.players.values()
    )


__all__ = ()
