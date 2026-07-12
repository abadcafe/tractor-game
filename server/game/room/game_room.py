"""User/player assignment and WebSocket entry for one game room."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Literal

from fastapi import WebSocket

from server.foundation.result import Ok, Rejected
from server.game.players import (
    AIPlayer,
    AutoPlayer,
    HumanPlayer,
    Player,
)
from server.game.room.bot_factory import (
    BotKind,
    BotPlayer,
    create_bot_player,
)
from server.game.room.game import Game

type PlayerIndex = Literal[0, 1, 2, 3]
type PlayerKind = Literal["empty", "user", "ai", "auto"]
type HumanPlayerFactory = Callable[[PlayerIndex], HumanPlayer]

_ALL_PLAYERS: tuple[
    PlayerIndex, PlayerIndex, PlayerIndex, PlayerIndex
] = (
    0,
    1,
    2,
    3,
)
_INVALID_PLAYER_CODE = 4410
_PLAYER_UNAVAILABLE_CODE = 4409
_PLAYER_DETACHED_CODE = 4408
_PLAYER_DETACHED_REASON = "player detached"
_NOT_ENOUGH_PLAYERS_REASON = "not enough players"


@dataclass(frozen=True, slots=True)
class RoomPlayer:
    index: PlayerIndex
    occupied: bool
    connected: bool
    kind: PlayerKind
    mine: bool
    ready: bool


class GameRoom:
    """Coordinates users and players around one game."""

    def __init__(
        self,
        *,
        human_player_factory: HumanPlayerFactory | None = None,
    ) -> None:
        self._game: Game | None = None
        self._human_player_factory = (
            human_player_factory or _create_human_player
        )
        self._seat_players: dict[PlayerIndex, Player] = {}
        self._player_users: dict[PlayerIndex, str] = {}
        self._bot_kinds: dict[PlayerIndex, BotKind] = {}
        self._bot_tasks: set[asyncio.Task[None]] = set()

    @property
    def game(self) -> Game | None:
        return self._game

    def player_at(self, player: int) -> Player | None:
        player_index = player_index_from(player)
        if player_index is None:
            return None
        return self._seat_players.get(player_index)

    async def connect_player(
        self,
        websocket: WebSocket,
        *,
        player: int,
        user_id: str,
    ) -> Ok[None] | Rejected:
        connected_player = self._attached_human_player(
            player=player, user_id=user_id
        )
        if isinstance(connected_player, Rejected):
            return await _close_rejected_for_reason(
                websocket, connected_player
            )

        game = self._ensure_game()
        if isinstance(game, Rejected):
            return await _close_rejected_for_reason(websocket, game)

        await connected_player.value.handle_connection(
            websocket, game.value
        )
        return Ok(None)

    async def attach_player(
        self, *, player: int, user_id: str
    ) -> Ok[PlayerIndex] | Rejected:
        player_index = player_index_from(player)
        if player_index is None:
            return Rejected("invalid player")
        if user_id.strip() == "":
            return Rejected("missing user id")
        if player_index in self._bot_kinds:
            return Rejected("player occupied")

        current_user = self._player_users.get(player_index)
        if current_user == user_id:
            return Ok(player_index)
        if current_user is not None:
            return Rejected("player occupied")

        player_obj = self._seat_players.get(player_index)
        if player_obj is None:
            self._seat_players[player_index] = (
                self._human_player_factory(player_index)
            )
        elif not isinstance(player_obj, HumanPlayer):
            return Rejected("player occupied")

        await self._detach_players_for_user(
            user_id, except_player=player_index
        )
        self._player_users[player_index] = user_id
        return Ok(player_index)

    async def detach_player(
        self, *, player: int, user_id: str
    ) -> Ok[None] | Rejected:
        player_index = player_index_from(player)
        if player_index is None:
            return Rejected("invalid player")
        if user_id.strip() == "":
            return Rejected("missing user id")
        if self._player_users.get(player_index) != user_id:
            return Rejected("user is not attached to player")

        await self._detach_player(player_index)
        return Ok(None)

    def players(
        self, *, user_id: str | None = None
    ) -> list[RoomPlayer]:
        ready_players = self._ready_players()
        result: list[RoomPlayer] = []
        for index in _ALL_PLAYERS:
            player_obj = self._seat_players.get(index)
            user_id_for_player = self._player_users.get(index)
            kind = self._player_kind(index, user_id_for_player)
            result.append(
                RoomPlayer(
                    index=index,
                    occupied=kind != "empty",
                    connected=kind == "user"
                    and isinstance(player_obj, HumanPlayer)
                    and player_obj.is_connected(),
                    kind=kind,
                    mine=user_id is not None
                    and user_id_for_player == user_id,
                    ready=index in ready_players,
                )
            )
        return result

    def occupied_players(self) -> list[PlayerIndex]:
        return [
            player.index for player in self.players() if player.occupied
        ]

    async def fill_bot_players(
        self, *, kind: BotKind, user_id: str
    ) -> Ok[None] | Rejected:
        if user_id.strip() == "":
            return Rejected("missing user id")
        if user_id not in self._player_users.values():
            return Rejected("user is not attached to a player")
        return await self.fill_empty_players_for_setup(kind=kind)

    async def fill_empty_players_for_setup(
        self,
        *,
        kind: BotKind,
        preserve_players: Collection[PlayerIndex] = (),
    ) -> Ok[None] | Rejected:
        if self._game is not None:
            return Rejected("game already started")
        preserved = set(preserve_players)
        for player in _ALL_PLAYERS:
            if (
                player in preserved
                or player in self._player_users
                or player in self._bot_kinds
            ):
                continue
            self._fill_player_with_bot(player, kind)
        return Ok(None)

    async def close_all(self) -> None:
        for player in self._seat_players.values():
            if (
                isinstance(player, HumanPlayer)
                and player.is_connected()
            ):
                await player.close_ws()
        for task in self._bot_tasks:
            task.cancel()
        self._bot_tasks.clear()

    def _ready_players(self) -> set[int]:
        if self._game is None:
            return set()
        return set(self._game.snapshot(0).next_round_confirmed)

    def _attached_human_player(
        self, *, player: int, user_id: str
    ) -> Ok[HumanPlayer] | Rejected:
        player_index = player_index_from(player)
        if player_index is None:
            return Rejected("invalid player")
        if user_id.strip() == "":
            return Rejected("missing user id")
        if self._player_users.get(player_index) != user_id:
            return Rejected("user is not attached to player")

        player_obj = self._seat_players.get(player_index)
        if isinstance(player_obj, HumanPlayer):
            return Ok(player_obj)
        return Rejected("player occupied")

    def _ensure_game(self) -> Ok[Game] | Rejected:
        if self._game is not None:
            return Ok(self._game)
        if not self._all_players_occupied():
            return Rejected(_NOT_ENOUGH_PLAYERS_REASON)

        players = self._players_for_game()
        if isinstance(players, Rejected):
            return players
        self._game = Game(players=players.value)
        self._start_bot_players(self._game)
        return Ok(self._game)

    def _players_for_game(self) -> Ok[list[Player]] | Rejected:
        players: list[Player] = []
        for player in _ALL_PLAYERS:
            player_obj = self._seat_players.get(player)
            if player_obj is None:
                return Rejected(_NOT_ENOUGH_PLAYERS_REASON)
            players.append(player_obj)
        return Ok(players)

    def _all_players_occupied(self) -> bool:
        for player in _ALL_PLAYERS:
            if (
                player not in self._player_users
                and player not in self._bot_kinds
            ):
                return False
        return True

    def _start_bot_players(self, game: Game) -> None:
        for player in _ALL_PLAYERS:
            if player not in self._bot_kinds:
                continue
            player_obj = self._seat_players.get(player)
            if isinstance(player_obj, AIPlayer) or isinstance(
                player_obj, AutoPlayer
            ):
                self._start_bot(player_obj, game)

    async def _detach_players_for_user(
        self, user_id: str, *, except_player: PlayerIndex
    ) -> None:
        players_for_user: list[PlayerIndex] = [
            player
            for player, current_user in self._player_users.items()
            if current_user == user_id and player != except_player
        ]
        for player in players_for_user:
            await self._detach_player(player)

    async def _detach_player(self, player: PlayerIndex) -> None:
        self._player_users.pop(player, None)
        player_obj = self._seat_players.get(player)
        if (
            isinstance(player_obj, HumanPlayer)
            and player_obj.is_connected()
        ):
            await player_obj.close_ws(
                code=_PLAYER_DETACHED_CODE,
                reason=_PLAYER_DETACHED_REASON,
            )
        if self._game is None:
            self._seat_players.pop(player, None)

    def _fill_player_with_bot(
        self, player: PlayerIndex, kind: BotKind
    ) -> None:
        bot = create_bot_player(player, kind)
        self._seat_players[player] = bot
        self._bot_kinds[player] = kind

    def _start_bot(self, bot: BotPlayer, game: Game) -> None:
        task = asyncio.create_task(bot.run(game))
        self._bot_tasks.add(task)
        task.add_done_callback(self._bot_tasks.discard)

    def _player_kind(
        self, player: PlayerIndex, user_id: str | None
    ) -> PlayerKind:
        bot_kind = self._bot_kinds.get(player)
        if bot_kind is not None:
            return bot_kind
        if user_id is not None:
            return "user"
        return "empty"


def player_index_from(value: int) -> PlayerIndex | None:
    match value:
        case 0:
            return 0
        case 1:
            return 1
        case 2:
            return 2
        case 3:
            return 3
        case _:
            return None


async def _close_rejected(
    websocket: WebSocket, *, code: int, reason: str
) -> Rejected:
    await websocket.close(code=code, reason=reason)
    return Rejected(reason)


async def _close_rejected_for_reason(
    websocket: WebSocket, rejected: Rejected
) -> Rejected:
    match rejected.reason:
        case "invalid player" | "missing user id":
            code = _INVALID_PLAYER_CODE
        case _:
            code = _PLAYER_UNAVAILABLE_CODE
    return await _close_rejected(
        websocket, code=code, reason=rejected.reason
    )


def _create_human_player(index: PlayerIndex) -> HumanPlayer:
    return HumanPlayer(index)
