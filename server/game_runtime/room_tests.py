"""Black-box tests for room ownership and runtime composition."""

from __future__ import annotations

from dataclasses import dataclass, field

from server.foundation.result import Ok, Rejected
from server.game import (
    CommandRejected,
    GameConfig,
    GameSeed,
    Seat,
    commands,
)
from server.game_runtime.room import (
    BotKind,
    GameRoom,
    RuntimeAgent,
)
from server.game_runtime.session import (
    AgentSubmission,
    Delivery,
    DisconnectReason,
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

    async def disconnect(self, reason: DisconnectReason) -> None:
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
class _Agent:
    seat: Seat
    target: AgentSubmission
    starts: int = 0
    stops: int = 0
    deliveries: list[Delivery] = field(default_factory=_deliveries)
    disconnects: list[DisconnectReason] = field(
        default_factory=_disconnects
    )

    async def start(self) -> None:
        self.starts += 1

    async def stop(self) -> None:
        self.stops += 1

    async def offer(self, delivery: Delivery) -> None:
        self.deliveries.append(delivery)

    async def disconnect(self, reason: DisconnectReason) -> None:
        self.disconnects.append(reason)


def _agents() -> list[_Agent]:
    return []


@dataclass(slots=True)
class _Factory:
    agents: list[_Agent] = field(default_factory=_agents)

    def create(
        self,
        seat: Seat,
        kind: BotKind,
        submission: AgentSubmission,
    ) -> RuntimeAgent:
        assert kind in (BotKind.AUTO, BotKind.AI)
        agent = _Agent(seat=seat, target=submission)
        self.agents.append(agent)
        return agent


def _room(factory: _Factory | None = None) -> GameRoom:
    return GameRoom(
        GameConfig(),
        GameSeed(19),
        factory if factory is not None else _Factory(),
    )


async def test_players_always_reports_four_empty_seats() -> None:
    room = _room()

    players = room.players(user_id="user")

    assert [player.index for player in players] == [0, 1, 2, 3]
    assert all(not player.occupied for player in players)
    assert all(not player.connected for player in players)
    assert all(player.kind == "empty" for player in players)


async def test_attach_requires_valid_seat_and_nonblank_user() -> None:
    room = _room()

    invalid = await room.attach_player(player=4, user_id="user")
    blank = await room.attach_player(player=0, user_id=" ")

    assert isinstance(invalid, Rejected)
    assert isinstance(blank, Rejected)
    assert all(not player.occupied for player in room.players())


async def test_attach_same_owner_is_idempotent() -> None:
    room = _room()
    first = await room.attach_player(player=0, user_id="owner")
    repeated = await room.attach_player(player=0, user_id="owner")
    occupied = await room.attach_player(player=0, user_id="other")

    assert isinstance(first, Ok)
    assert isinstance(repeated, Ok)
    assert isinstance(occupied, Rejected)
    assert room.players(user_id="owner")[0].mine


async def test_attach_moves_same_user_between_empty_seats() -> None:
    room = _room()
    await room.attach_player(player=0, user_id="owner")

    moved = await room.attach_player(player=2, user_id="owner")

    assert isinstance(moved, Ok)
    players = room.players(user_id="owner")
    assert not players[0].occupied
    assert players[2].occupied
    assert players[2].mine


async def test_detach_requires_matching_owner() -> None:
    room = _room()
    await room.attach_player(player=1, user_id="owner")

    other = await room.detach_player(player=1, user_id="other")
    missing = await room.detach_player(player=2, user_id="owner")
    detached = await room.detach_player(player=1, user_id="owner")

    assert isinstance(other, Rejected)
    assert isinstance(missing, Rejected)
    assert isinstance(detached, Ok)
    assert not room.players()[1].occupied


async def test_fill_bots_preserves_attached_owner() -> None:
    room = _room()
    missing = await room.fill_bots(
        kind=BotKind.AUTO,
        user_id="owner",
    )
    await room.attach_player(player=2, user_id="owner")

    filled = await room.fill_bots(
        kind=BotKind.AI,
        user_id="owner",
    )

    assert isinstance(missing, Rejected)
    assert isinstance(filled, Ok)
    players = room.players(user_id="owner")
    assert players[2].kind == "user"
    assert all(
        player.kind == "ai" for player in players if player.index != 2
    )


async def test_connect_requires_attachment_and_full_occupancy() -> None:
    room = _room()
    sink = _Sink()
    unattached = await room.connect_player(
        player=0,
        user_id="owner",
        sink=sink,
    )
    await room.attach_player(player=0, user_id="owner")

    incomplete = await room.connect_player(
        player=0,
        user_id="owner",
        sink=sink,
    )

    assert isinstance(unattached, Rejected)
    assert isinstance(incomplete, Rejected)
    assert not room.started()


async def test_connect_starts_one_session_and_all_bot_agents() -> None:
    factory = _Factory()
    room = _room(factory)
    await room.attach_player(player=2, user_id="owner")
    await room.fill_bots(
        kind=BotKind.AUTO,
        user_id="owner",
    )
    sink = _Sink()

    connected = await room.connect_player(
        player=2,
        user_id="owner",
        sink=sink,
    )

    assert isinstance(connected, Ok)
    assert room.started()
    assert len(factory.agents) == 3
    assert all(agent.starts == 1 for agent in factory.agents)
    assert sink.deliveries == []
    assert room.players(user_id="owner")[2].connected


async def test_connect_takeover_disconnects_previous_sink() -> None:
    room = _room()
    await room.attach_player(player=0, user_id="owner")
    await room.fill_bots(
        kind=BotKind.AUTO,
        user_id="owner",
    )
    first = _Sink()
    second = _Sink()
    await room.connect_player(
        player=0,
        user_id="owner",
        sink=first,
    )

    connected = await room.connect_player(
        player=0,
        user_id="owner",
        sink=second,
    )

    assert isinstance(connected, Ok)
    assert first.disconnects == [DisconnectReason.REPLACED]
    assert second.disconnects == []


async def test_receive_ignores_stale_connection_owner() -> None:
    room = _room()
    await room.attach_player(player=0, user_id="owner")
    await room.fill_bots(
        kind=BotKind.AUTO,
        user_id="owner",
    )
    stale = _Sink()
    current = _Sink()
    connected = await room.connect_player(
        player=0,
        user_id="owner",
        sink=stale,
    )
    assert isinstance(connected, Ok)
    await room.connect_player(
        player=0,
        user_id="owner",
        sink=current,
    )
    decoder = _Decoder(commands.ConfirmRound())

    await room.receive(Seat.NORTH, stale, 1, decoder)
    await room.receive(Seat.NORTH, current, 0, decoder)

    assert decoder.decode_count == 0
    assert stale.deliveries == []
    assert len(current.deliveries) == 1


async def test_detach_after_start_disconnects_current_sink() -> None:
    room = _room()
    await room.attach_player(player=0, user_id="owner")
    await room.fill_bots(
        kind=BotKind.AUTO,
        user_id="owner",
    )
    sink = _Sink()
    await room.connect_player(
        player=0,
        user_id="owner",
        sink=sink,
    )

    detached = await room.detach_player(
        player=0,
        user_id="owner",
    )

    assert isinstance(detached, Ok)
    assert sink.disconnects == [DisconnectReason.DETACHED]
    assert not room.players()[0].occupied


async def test_close_stops_agents_and_disconnects_human() -> None:
    factory = _Factory()
    room = _room(factory)
    await room.attach_player(player=0, user_id="owner")
    await room.fill_bots(
        kind=BotKind.AUTO,
        user_id="owner",
    )
    sink = _Sink()
    await room.connect_player(
        player=0,
        user_id="owner",
        sink=sink,
    )

    await room.close()

    assert sink.disconnects == [DisconnectReason.SESSION_CLOSED]
    assert all(agent.stops == 1 for agent in factory.agents)
    assert all(
        agent.disconnects == [DisconnectReason.SESSION_CLOSED]
        for agent in factory.agents
    )
