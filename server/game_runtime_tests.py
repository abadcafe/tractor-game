"""Black-box tests for room ownership and player composition."""

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
from server.game_runtime import BotPolicyName, GameRoom, UserId
from server.game_runtime.player import (
    BotPlayerDescription,
    ConnectionCloseReason,
    HumanPlayerDescription,
    Player,
    PlayerDescription,
    PlayerPort,
    PlayerView,
)


def _views() -> list[PlayerView]:
    return []


def _close_reasons() -> list[ConnectionCloseReason]:
    return []


@dataclass(slots=True)
class _Transport:
    views: list[PlayerView] = field(default_factory=_views)
    closes: list[ConnectionCloseReason] = field(
        default_factory=_close_reasons
    )

    async def send(self, view: PlayerView) -> None:
        self.views.append(view)

    async def close(self, reason: ConnectionCloseReason) -> None:
        self.closes.append(reason)


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
class _Bot:
    policy_name: BotPolicyName
    port: PlayerPort | None = None
    views: list[PlayerView] = field(default_factory=_views)
    stop_count: int = 0

    def lobby_status(
        self,
        requester: UserId | None,
    ) -> PlayerDescription:
        del requester
        return BotPlayerDescription(
            kind="bot",
            policy=self.policy_name,
        )

    async def start(self, port: PlayerPort) -> None:
        assert self.port is None
        self.port = port

    async def update(self, view: PlayerView) -> None:
        self.views.append(view)

    async def stop(self) -> None:
        self.stop_count += 1


def _bots() -> list[_Bot]:
    return []


@dataclass(slots=True)
class _Factory:
    bots: list[_Bot] = field(default_factory=_bots)

    def create(
        self,
        seat: Seat,
        policy_name: BotPolicyName,
    ) -> Player:
        del seat
        bot = _Bot(policy_name=policy_name)
        self.bots.append(bot)
        return bot


def _room(factory: _Factory | None = None) -> GameRoom:
    return GameRoom(
        GameConfig(),
        GameSeed(19),
        factory if factory is not None else _Factory(),
    )


def _user(value: str = "owner") -> UserId:
    parsed = UserId.parse(value)
    assert parsed is not None
    return parsed


def _human_at(
    room: GameRoom,
    seat: Seat,
    requester: UserId | None,
) -> HumanPlayerDescription:
    room_seat = next(
        candidate
        for candidate in room.seats(user_id=requester)
        if candidate.seat == seat
    )
    description = room_seat.player
    assert description is not None
    assert description["kind"] == "human"
    return description


async def _start_room(
    room: GameRoom,
    *,
    seat: Seat = Seat.A,
    user: UserId | None = None,
    transport: _Transport | None = None,
) -> tuple[UserId, _Transport]:
    owner = user or _user()
    connection = transport or _Transport()
    assert isinstance(
        room.occupy_seat(seat=seat, user_id=owner),
        Ok,
    )
    assert isinstance(
        room.fill_bots(policy="auto", user_id=owner),
        Ok,
    )
    connected = await room.connect_seat(
        seat=seat,
        user_id=owner,
        transport=connection,
    )
    assert isinstance(connected, Ok)
    return owner, connection


def test_room_reports_four_empty_seats() -> None:
    room = _room()

    room_seats = room.seats(user_id=_user())

    assert [room_seat.seat for room_seat in room_seats] == [
        Seat.A,
        Seat.B,
        Seat.C,
        Seat.D,
    ]
    assert all(room_seat.player is None for room_seat in room_seats)
    assert all(not room_seat.ready for room_seat in room_seats)


def test_user_id_validation_is_outside_room_boundary() -> None:
    assert UserId.parse(None) is None
    assert UserId.parse("") is None
    assert UserId.parse("   ") is None
    assert UserId.parse(" owner ") == UserId("owner")


def test_occupy_is_idempotent_and_rejects_other_user() -> None:
    room = _room()
    owner = _user()

    first = room.occupy_seat(seat=Seat.A, user_id=owner)
    repeated = room.occupy_seat(seat=Seat.A, user_id=owner)
    occupied = room.occupy_seat(
        seat=Seat.A,
        user_id=_user("other"),
    )

    assert isinstance(first, Ok)
    assert isinstance(repeated, Ok)
    assert isinstance(occupied, Rejected)
    assert _human_at(room, Seat.A, owner)["mine"]
    assert not _human_at(room, Seat.A, _user("other"))["mine"]


def test_occupy_moves_same_user_only_to_empty_seat() -> None:
    room = _room()
    owner = _user()
    other = _user("other")
    assert isinstance(
        room.occupy_seat(seat=Seat.A, user_id=owner),
        Ok,
    )
    assert isinstance(
        room.occupy_seat(seat=Seat.C, user_id=other),
        Ok,
    )

    blocked = room.occupy_seat(seat=Seat.C, user_id=owner)
    moved = room.occupy_seat(seat=Seat.B, user_id=owner)

    assert isinstance(blocked, Rejected)
    assert isinstance(moved, Ok)
    assert room.seats()[0].player is None
    moved_player = room.seats(user_id=owner)[1].player
    assert moved_player is not None
    assert moved_player["kind"] == "human"


def test_vacate_requires_matching_human_owner() -> None:
    room = _room()
    owner = _user()
    assert isinstance(
        room.occupy_seat(seat=Seat.B, user_id=owner),
        Ok,
    )

    other = room.vacate_seat(
        seat=Seat.B,
        user_id=_user("other"),
    )
    wrong_seat = room.vacate_seat(seat=Seat.C, user_id=owner)
    removed = room.vacate_seat(seat=Seat.B, user_id=owner)

    assert isinstance(other, Rejected)
    assert isinstance(wrong_seat, Rejected)
    assert isinstance(removed, Ok)
    assert room.seats()[1].player is None


def test_fill_bots_requires_human_and_preserves_human() -> None:
    factory = _Factory()
    room = _room(factory)
    owner = _user()
    missing = room.fill_bots(policy="llm", user_id=owner)
    assert isinstance(
        room.occupy_seat(seat=Seat.C, user_id=owner),
        Ok,
    )

    filled = room.fill_bots(policy="llm", user_id=owner)

    assert isinstance(missing, Rejected)
    assert isinstance(filled, Ok)
    assert len(factory.bots) == 3
    assert all(bot.policy_name == "llm" for bot in factory.bots)
    descriptions = [seat.player for seat in room.seats(user_id=owner)]
    human = descriptions[2]
    assert human is not None
    assert human["kind"] == "human"
    assert all(
        description is not None and description["kind"] == "bot"
        for index, description in enumerate(descriptions)
        if index != 2
    )


async def test_connect_requires_owned_seat_and_complete_roster() -> (
    None
):
    room = _room()
    owner = _user()
    transport = _Transport()

    unattached = await room.connect_seat(
        seat=Seat.A,
        user_id=owner,
        transport=transport,
    )
    assert isinstance(
        room.occupy_seat(seat=Seat.A, user_id=owner),
        Ok,
    )
    incomplete = await room.connect_seat(
        seat=Seat.A,
        user_id=owner,
        transport=transport,
    )

    assert isinstance(unattached, Rejected)
    assert isinstance(incomplete, Rejected)
    assert not room.started()
    assert not _human_at(room, Seat.A, owner)["connected"]


async def test_connect_freezes_roster_and_starts_every_player() -> None:
    factory = _Factory()
    room = _room(factory)
    owner, _transport = await _start_room(
        room,
        seat=Seat.C,
    )

    assert room.started()
    assert all(bot.port is not None for bot in factory.bots)
    assert _human_at(room, Seat.C, owner)["connected"]
    assert isinstance(
        room.occupy_seat(
            seat=Seat.C,
            user_id=owner,
        ),
        Rejected,
    )
    assert isinstance(
        room.vacate_seat(seat=Seat.C, user_id=owner),
        Rejected,
    )
    assert isinstance(
        room.fill_bots(policy="auto", user_id=owner),
        Rejected,
    )


async def test_connect_replaces_transport_but_not_human_player() -> (
    None
):
    room = _room()
    owner = _user()
    first = _Transport()
    await _start_room(room, user=owner, transport=first)
    second = _Transport()

    connected = await room.connect_seat(
        seat=Seat.A,
        user_id=owner,
        transport=second,
    )

    assert isinstance(connected, Ok)
    assert first.closes == [ConnectionCloseReason.REPLACED]
    assert second.closes == []
    assert _human_at(room, Seat.A, owner)["connected"]


async def test_receive_accepts_only_current_transport() -> None:
    room = _room()
    owner = _user()
    stale = _Transport()
    await _start_room(room, user=owner, transport=stale)
    current = _Transport()
    connected = await room.connect_seat(
        seat=Seat.A,
        user_id=owner,
        transport=current,
    )
    assert isinstance(connected, Ok)
    decoder = _Decoder(commands.ConfirmRound())

    await room.receive(
        seat=Seat.A,
        user_id=owner,
        transport=stale,
        seq=1,
        decoder=decoder,
    )
    await room.receive(
        seat=Seat.A,
        user_id=owner,
        transport=current,
        seq=0,
        decoder=decoder,
    )

    assert decoder.decode_count == 0
    assert stale.views == []
    assert len(current.views) == 1
    assert current.views[0].seq == 1


async def test_disconnect_ignores_stale_transport() -> None:
    room = _room()
    owner = _user()
    stale = _Transport()
    await _start_room(room, user=owner, transport=stale)
    current = _Transport()
    connected = await room.connect_seat(
        seat=Seat.A,
        user_id=owner,
        transport=current,
    )
    assert isinstance(connected, Ok)

    room.disconnect_seat(
        seat=Seat.A,
        user_id=owner,
        transport=stale,
    )

    assert _human_at(room, Seat.A, owner)["connected"]


async def test_close_is_idempotent_for_active_and_lobby_rooms() -> None:
    factory = _Factory()
    active = _room(factory)
    _owner, transport = await _start_room(active)
    lobby_factory = _Factory()
    lobby = _room(lobby_factory)
    lobby_owner = _user("lobby")
    assert isinstance(
        lobby.occupy_seat(seat=Seat.A, user_id=lobby_owner),
        Ok,
    )
    assert isinstance(
        lobby.fill_bots(
            policy="auto",
            user_id=lobby_owner,
        ),
        Ok,
    )

    await active.close()
    await active.close()
    await lobby.close()
    await lobby.close()

    assert transport.closes == [ConnectionCloseReason.SESSION_CLOSED]
    assert all(bot.stop_count == 1 for bot in factory.bots)
    assert all(bot.stop_count == 1 for bot in lobby_factory.bots)
    assert not active.started()
    assert isinstance(
        active.occupy_seat(
            seat=Seat.A,
            user_id=_user(),
        ),
        Rejected,
    )
