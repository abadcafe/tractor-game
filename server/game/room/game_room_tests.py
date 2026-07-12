"""Tests for server/game_room.py user/player assignment behavior."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from unittest.mock import AsyncMock

import pytest
from fastapi import WebSocketDisconnect

from server.foundation.result import Ok, Rejected
from server.game.players import (
    AIPlayer,
    AutoPlayer,
    GameView,
    HumanPlayer,
)
from server.game.protocol import PlayerMessage, StateMessage
from server.game.room.game import Game
from server.game.room.game_room import (
    GameRoom,
    PlayerIndex,
    player_index_from,
)

_TEST_PLAYERS: tuple[
    PlayerIndex, PlayerIndex, PlayerIndex, PlayerIndex
] = (0, 1, 2, 3)


class RecordingHumanPlayer(HumanPlayer):
    def __init__(self, index: int) -> None:
        super().__init__(index)
        self.messages: list[StateMessage] = []

    async def on_state(
        self, game: GameView, message: StateMessage
    ) -> None:
        self.messages.append(message)
        await super().on_state(game, message)


def _room() -> tuple[GameRoom, dict[PlayerIndex, RecordingHumanPlayer]]:
    players: dict[PlayerIndex, RecordingHumanPlayer] = {}

    def create_human_player(index: PlayerIndex) -> HumanPlayer:
        player = RecordingHumanPlayer(index)
        players[index] = player
        return player

    return GameRoom(human_player_factory=create_human_player), players


def _disconnecting_websocket() -> AsyncMock:
    websocket = AsyncMock()
    websocket.receive_json = AsyncMock(
        side_effect=WebSocketDisconnect()
    )
    return websocket


def _blocking_websocket(release: asyncio.Event) -> AsyncMock:
    websocket = AsyncMock()

    async def receive_json() -> object:
        await release.wait()
        raise WebSocketDisconnect()

    websocket.receive_json = AsyncMock(side_effect=receive_json)
    return websocket


async def _wait_until(predicate: Callable[[], bool]) -> None:
    for _index in range(100):
        if predicate():
            return
        await asyncio.sleep(0.01)
    assert predicate()


async def _attach_all_humans(room: GameRoom) -> None:
    for index in _TEST_PLAYERS:
        result = await room.attach_player(
            player=index, user_id=f"user-{index}"
        )
        assert isinstance(result, Ok)


async def _connect_and_create_game(
    room: GameRoom, *, player: PlayerIndex = 0
) -> Game:
    result = await room.connect_player(
        _disconnecting_websocket(),
        player=player,
        user_id=f"user-{player}",
    )
    assert isinstance(result, Ok)
    game = room.game
    assert game is not None
    return game


def _player_list(
    players: dict[PlayerIndex, RecordingHumanPlayer],
) -> list[RecordingHumanPlayer]:
    return [players[index] for index in _TEST_PLAYERS]


def test_player_index_from_validates_range() -> None:
    assert player_index_from(0) == 0
    assert player_index_from(3) == 3
    assert player_index_from(-1) is None
    assert player_index_from(4) is None


@pytest.mark.asyncio
async def test_connect_player_requires_attached_user() -> None:
    room, _players = _room()
    websocket = _disconnecting_websocket()

    result = await room.connect_player(
        websocket, player=1, user_id="user-1"
    )

    assert isinstance(result, Rejected)
    assert result.reason == "user is not attached to player"
    websocket.close.assert_awaited_once_with(
        code=4409,
        reason="user is not attached to player",
    )
    assert room.game is None


@pytest.mark.asyncio
async def test_connect_player_requires_four_occupied_players() -> None:
    room, _players = _room()
    attached = await room.attach_player(player=1, user_id="user-1")
    websocket = _disconnecting_websocket()

    result = await room.connect_player(
        websocket, player=1, user_id="user-1"
    )

    assert isinstance(attached, Ok)
    assert isinstance(result, Rejected)
    assert result.reason == "not enough players"
    websocket.close.assert_awaited_once_with(
        code=4409,
        reason="not enough players",
    )
    assert room.game is None


@pytest.mark.asyncio
async def test_connect_player_creates_game_from_seated_players() -> (
    None
):
    room, players = _room()
    attached = await room.attach_player(player=1, user_id="user-1")
    filled = await room.fill_bot_players(kind="auto", user_id="user-1")
    websocket = _disconnecting_websocket()

    result = await room.connect_player(
        websocket, player=1, user_id="user-1"
    )

    assert isinstance(attached, Ok)
    assert isinstance(filled, Ok)
    assert isinstance(result, Ok)
    game = room.game
    assert game is not None
    assert isinstance(game.get_player(0), AutoPlayer)
    assert game.get_player(1) is players[1]
    assert isinstance(game.get_player(2), AutoPlayer)
    assert isinstance(game.get_player(3), AutoPlayer)
    room_players = room.players(user_id="user-1")
    assert [player.kind for player in room_players] == [
        "auto",
        "user",
        "auto",
        "auto",
    ]
    assert room_players[1].connected is False
    assert room_players[1].mine is True


@pytest.mark.asyncio
async def test_attach_player_without_websocket() -> None:
    room, players = _room()

    result = await room.attach_player(player=0, user_id="user-0")

    assert isinstance(result, Ok)
    assert result.value == 0
    assert room.game is None
    assert room.player_at(0) is players[0]
    room_players = room.players(user_id="user-0")
    assert room_players[0].occupied is True
    assert room_players[0].connected is False
    assert room_players[0].mine is True


@pytest.mark.asyncio
async def test_attach_player_switches_user_player_before_game() -> None:
    room, players = _room()

    first = await room.attach_player(player=0, user_id="user-0")
    second = await room.attach_player(player=2, user_id="user-0")

    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert room.player_at(0) is None
    assert room.player_at(2) is players[2]
    room_players = room.players(user_id="user-0")
    assert room_players[0].occupied is False
    assert room_players[0].mine is False
    assert room_players[2].occupied is True
    assert room_players[2].mine is True


@pytest.mark.asyncio
async def test_attach_player_switch_closes_previous_websocket() -> None:
    room, players = _room()
    await _attach_all_humans(room)
    release = asyncio.Event()
    websocket = _blocking_websocket(release)
    connect_task = asyncio.create_task(
        room.connect_player(websocket, player=1, user_id="user-1")
    )

    try:
        await _wait_until(lambda: players[1].is_connected())
        detached = await room.detach_player(player=3, user_id="user-3")

        result = await room.attach_player(player=3, user_id="user-1")

        assert isinstance(detached, Ok)
        assert isinstance(result, Ok)
        websocket.close.assert_awaited_once_with(
            code=4408,
            reason="player detached",
        )
        assert players[1].is_connected() is False
        assert room.players(user_id="user-1")[3].mine is True
    finally:
        release.set()
        connect_result = await connect_task
        assert isinstance(connect_result, Ok)


@pytest.mark.asyncio
async def test_detach_player_detaches_attached_player() -> None:
    room, _players = _room()
    attached = await room.attach_player(player=3, user_id="user-3")

    result = await room.detach_player(player=3, user_id="user-3")

    assert isinstance(attached, Ok)
    assert isinstance(result, Ok)
    room_players = room.players(user_id="user-3")
    assert room_players[3].occupied is False
    assert room_players[3].mine is False


@pytest.mark.asyncio
async def test_detach_player_closes_current_websocket() -> None:
    room, players = _room()
    await _attach_all_humans(room)
    release = asyncio.Event()
    websocket = _blocking_websocket(release)
    connect_task = asyncio.create_task(
        room.connect_player(websocket, player=2, user_id="user-2")
    )

    try:
        await _wait_until(lambda: players[2].is_connected())

        result = await room.detach_player(player=2, user_id="user-2")

        assert isinstance(result, Ok)
        websocket.close.assert_awaited_once_with(
            code=4408,
            reason="player detached",
        )
        assert players[2].is_connected() is False
        assert room.players(user_id="user-2")[2].occupied is False
    finally:
        release.set()
        connect_result = await connect_task
        assert isinstance(connect_result, Ok)


@pytest.mark.asyncio
async def test_detach_player_rejects_other_user() -> None:
    room, _players = _room()
    attached = await room.attach_player(player=1, user_id="user-1")

    result = await room.detach_player(player=1, user_id="user-other")

    assert isinstance(attached, Ok)
    assert isinstance(result, Rejected)
    assert result.reason == "user is not attached to player"
    assert room.players(user_id="user-1")[1].occupied is True


@pytest.mark.asyncio
async def test_connect_player_rejects_unattached_user() -> None:
    room, players = _room()
    await _attach_all_humans(room)
    release = asyncio.Event()
    first = _blocking_websocket(release)
    second = _disconnecting_websocket()
    connect_task = asyncio.create_task(
        room.connect_player(first, player=2, user_id="user-2")
    )

    try:
        await _wait_until(lambda: players[2].is_connected())

        second_result = await room.connect_player(
            second, player=2, user_id="user-other"
        )

        assert isinstance(second_result, Rejected)
        second.close.assert_awaited_once_with(
            code=4409,
            reason="user is not attached to player",
        )
    finally:
        release.set()
        first_result = await connect_task
        assert isinstance(first_result, Ok)


@pytest.mark.asyncio
async def test_connect_player_allows_same_user_to_reenter_player() -> (
    None
):
    room, players = _room()
    await _attach_all_humans(room)
    release = asyncio.Event()
    first = _blocking_websocket(release)
    second = _disconnecting_websocket()
    connect_task = asyncio.create_task(
        room.connect_player(first, player=3, user_id="user-3")
    )

    try:
        await _wait_until(lambda: players[3].is_connected())

        second_result = await room.connect_player(
            second, player=3, user_id="user-3"
        )

        assert isinstance(second_result, Ok)
        first.accept.assert_awaited_once()
        second.accept.assert_awaited_once()
    finally:
        release.set()
        first_result = await connect_task
        assert isinstance(first_result, Ok)


@pytest.mark.asyncio
async def test_connect_player_rejects_blank_user_id() -> None:
    room, _players = _room()
    websocket = _disconnecting_websocket()

    result = await room.connect_player(
        websocket, player=0, user_id="  "
    )

    assert isinstance(result, Rejected)
    websocket.close.assert_awaited_once_with(
        code=4410,
        reason="missing user id",
    )


@pytest.mark.asyncio
async def test_connect_player_allows_new_user_after_game_started() -> (
    None
):
    room, players = _room()
    await _attach_all_humans(room)
    game = await _connect_and_create_game(room)
    await _start_game(game, _player_list(players))
    detached = await room.detach_player(player=0, user_id="user-0")
    attached = await room.attach_player(player=0, user_id="late-user")
    websocket = _disconnecting_websocket()

    result = await room.connect_player(
        websocket, player=0, user_id="late-user"
    )

    assert isinstance(detached, Ok)
    assert isinstance(attached, Ok)
    assert isinstance(result, Ok)
    websocket.accept.assert_awaited_once()
    assert room.players(user_id="late-user")[0].mine is True


@pytest.mark.asyncio
async def test_detach_player_allows_detach_after_game_started() -> None:
    room, players = _room()
    await _attach_all_humans(room)
    game = await _connect_and_create_game(room)
    await _start_game(game, _player_list(players))

    result = await room.detach_player(player=0, user_id="user-0")

    assert isinstance(result, Ok)
    assert room.players(user_id="user-0")[0].occupied is False


@pytest.mark.asyncio
async def test_attach_rejects_occupied_after_game_started() -> None:
    room, players = _room()
    await _attach_all_humans(room)
    game = await _connect_and_create_game(room)
    await _start_game(game, _player_list(players))

    result = await room.attach_player(player=0, user_id="user-other")

    assert isinstance(result, Rejected)
    assert result.reason == "player occupied"


@pytest.mark.asyncio
async def test_fill_auto_bots_preserves_attached_user() -> None:
    room, players = _room()
    attached = await room.attach_player(player=1, user_id="user-1")

    result = await room.fill_bot_players(kind="auto", user_id="user-1")

    assert isinstance(attached, Ok)
    assert isinstance(result, Ok)
    assert room.game is None
    room_players = room.players(user_id="user-1")
    assert [player.kind for player in room_players] == [
        "auto",
        "user",
        "auto",
        "auto",
    ]
    assert [player.occupied for player in room_players] == [
        True,
        True,
        True,
        True,
    ]
    assert room_players[1].mine is True
    assert isinstance(room.player_at(0), AutoPlayer)
    assert room.player_at(1) is players[1]
    assert isinstance(room.player_at(2), AutoPlayer)
    assert isinstance(room.player_at(3), AutoPlayer)


@pytest.mark.asyncio
async def test_fill_empty_players_with_ai_bots() -> None:
    room, players = _room()
    attached = await room.attach_player(player=2, user_id="user-2")

    result = await room.fill_bot_players(kind="ai", user_id="user-2")

    assert isinstance(attached, Ok)
    assert isinstance(result, Ok)
    room_players = room.players(user_id="user-2")
    assert [player.kind for player in room_players] == [
        "ai",
        "ai",
        "user",
        "ai",
    ]
    assert isinstance(room.player_at(0), AIPlayer)
    assert isinstance(room.player_at(1), AIPlayer)
    assert room.player_at(2) is players[2]
    assert isinstance(room.player_at(3), AIPlayer)


@pytest.mark.asyncio
async def test_fill_bot_players_requires_attached_user() -> None:
    room, _players = _room()

    result = await room.fill_bot_players(kind="auto", user_id="user-x")

    assert isinstance(result, Rejected)
    assert result.reason == "user is not attached to a player"
    room_players = room.players(user_id="user-x")
    assert [player.kind for player in room_players] == [
        "empty",
        "empty",
        "empty",
        "empty",
    ]
    for index in _TEST_PLAYERS:
        assert room.player_at(index) is None


@pytest.mark.asyncio
async def test_attach_player_rejects_bot_filled_player() -> None:
    room, _players = _room()
    attached = await room.attach_player(player=2, user_id="user-2")
    filled = await room.fill_bot_players(kind="auto", user_id="user-2")

    result = await room.attach_player(player=0, user_id="user-other")

    assert isinstance(attached, Ok)
    assert isinstance(filled, Ok)
    assert isinstance(result, Rejected)
    assert result.reason == "player occupied"
    assert room.players(user_id="user-other")[0].kind == "auto"


@pytest.mark.asyncio
async def test_fill_bot_players_rejects_after_game_created() -> None:
    room, _players = _room()
    attached = await room.attach_player(player=0, user_id="user-0")
    filled = await room.fill_bot_players(kind="auto", user_id="user-0")
    game = await _connect_and_create_game(room)

    result = await room.fill_bot_players(kind="auto", user_id="user-0")

    assert isinstance(attached, Ok)
    assert isinstance(filled, Ok)
    assert game is room.game
    assert isinstance(result, Rejected)
    assert result.reason == "game already started"


async def _start_game(
    game: Game, players: list[RecordingHumanPlayer]
) -> None:
    for player in players:
        await game.receive(player.index, PlayerMessage(seq=0, raw={}))
    for player in players:
        last_message = player.messages[-1]
        await game.receive(
            player.index,
            PlayerMessage(
                seq=last_message.seq, raw={"type": "next_round"}
            ),
        )
    assert game.is_started() is True
