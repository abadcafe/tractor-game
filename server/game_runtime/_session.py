"""Single-owner runtime around one immutable game and four players."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Protocol, assert_never, final, override

from server.foundation.result import Ok
from server.game import (
    CommandRejected,
    GameConfig,
    GameSeed,
    GameState,
    Seat,
    SeatMap,
    apply,
    commands,
    create,
    observe,
)
from server.game.rules.cards import CardId

from .player import (
    CommandDecoder,
    Player,
    PlayerFailure,
    PlayerInbox,
    PlayerPort,
    PlayerView,
)
from .registry import GameId

_LOGGER = logging.getLogger(__name__)


class SessionTaskOwner(Protocol):
    """Application-owned capability for one supervised Session task."""

    def create_task(
        self,
        coroutine: Coroutine[object, object, None],
        *,
        name: str,
    ) -> asyncio.Task[None]:
        """Start a task whose exception remains owned by the caller."""
        ...


@dataclass(frozen=True, slots=True)
class _PlayerReady:
    seat: Seat
    completed: asyncio.Future[None]


@dataclass(frozen=True, slots=True)
class _PlayerInitialized:
    seat: Seat


@dataclass(frozen=True, slots=True)
class _ViewRequested:
    seat: Seat


@dataclass(frozen=True, slots=True)
class _CommandSubmitted:
    seat: Seat
    seq: int
    decoder: CommandDecoder


@dataclass(frozen=True, slots=True)
class _FailureReported:
    seat: Seat
    failure: PlayerFailure


@dataclass(frozen=True, slots=True)
class _CloseRequested:
    completed: asyncio.Future[None]


type _SessionMessage = (
    _PlayerReady
    | _PlayerInitialized
    | _ViewRequested
    | _CommandSubmitted
    | _FailureReported
    | _CloseRequested
)


@dataclass(slots=True)
class _PlayerChannel(PlayerInbox):
    queue: asyncio.Queue[PlayerView]

    @override
    async def receive(self) -> PlayerView:
        return await self.queue.get()


def _player_channels() -> SeatMap[_PlayerChannel]:
    return SeatMap(
        a=_PlayerChannel(asyncio.Queue(maxsize=1)),
        b=_PlayerChannel(asyncio.Queue(maxsize=1)),
        c=_PlayerChannel(asyncio.Queue(maxsize=1)),
        d=_PlayerChannel(asyncio.Queue(maxsize=1)),
    )


@final
class Session:
    """Serialize state transitions and supervise every player task."""

    def __init__(
        self,
        *,
        game_id: GameId,
        config: GameConfig,
        seed: GameSeed,
        players: SeatMap[Player],
    ) -> None:
        self._game_id = game_id
        self._state: GameState = create(config, seed)
        self._seq = 1
        self._players = players
        self._channels = _player_channels()
        self._messages: asyncio.Queue[_SessionMessage] = asyncio.Queue()
        self._ready_seats: set[Seat] = set()
        self._ready_waiters: dict[Seat, asyncio.Future[None]] = {}
        self._initialized_seats: set[Seat] = set()
        self._initialized = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._failure: PlayerFailure | None = None
        self._closing = False
        self._closed = False

    async def start(self, owner: SessionTaskOwner) -> None:
        """Start the single Session root under an application owner."""
        assert self._task is None
        task = owner.create_task(
            self._run(),
            name=f"game:{self._game_id.value}:session",
        )
        task.add_done_callback(self._session_task_finished)
        self._task = task
        ready = asyncio.create_task(self._initialized.wait())
        done, _pending = await asyncio.wait(
            (ready, task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if task in done:
            if not ready.done():
                _ = ready.cancel()
                try:
                    await ready
                except asyncio.CancelledError:
                    pass
            task.result()
            raise AssertionError(
                "Session stopped before players were ready"
            )
        assert ready in done

    def _session_task_finished(
        self,
        task: asyncio.Task[None],
    ) -> None:
        if task.cancelled():
            return
        failure = task.exception()
        if failure is None:
            return
        _LOGGER.critical(
            "runtime.task game_id=%s task=session error=%s: %s",
            self._game_id.value,
            type(failure).__name__,
            failure,
            exc_info=(
                type(failure),
                failure,
                failure.__traceback__,
            ),
        )

    def view(self, seat: Seat) -> PlayerView:
        """Return the current view without changing sequence."""
        failure = self._failure
        return PlayerView(
            viewer=seat,
            seq=self._seq,
            snapshot=observe(self._state, seat),
            status="failed" if failure is not None else "running",
            error=None if failure is None else failure.error,
        )

    async def close(self) -> None:
        """Stop the Session root and all four players exactly once."""
        if self._closed:
            return
        self._closed = True
        task = self._task
        if task is None:
            return
        if task.done():
            task.result()
            return
        loop = asyncio.get_running_loop()
        completed: asyncio.Future[None] = loop.create_future()
        self._messages.put_nowait(_CloseRequested(completed))
        await completed
        await task

    async def _run(self) -> None:
        player_tasks: list[asyncio.Task[None]] = []
        async with asyncio.TaskGroup() as group:
            for seat, player in self._players.items():
                task = group.create_task(
                    self._run_player(seat, player),
                    name=(
                        f"game:{self._game_id.value}:player:"
                        f"{seat.value}"
                    ),
                )
                task.add_done_callback(
                    lambda completed, player_seat=seat: (
                        self._player_task_finished(
                            player_seat, completed
                        )
                    )
                )
                player_tasks.append(task)
            try:
                await self._serve()
            finally:
                self._closing = True
                for task in player_tasks:
                    _ = task.cancel()

    async def _run_player(self, seat: Seat, player: Player) -> None:
        await player.run(
            _SeatPort(
                game_id=self._game_id,
                seat=seat,
                messages=self._messages,
            ),
            self._channels.at(seat),
        )
        assert self._closing, "player stopped before Session close"

    def _player_task_finished(
        self,
        seat: Seat,
        task: asyncio.Task[None],
    ) -> None:
        if task.cancelled():
            return
        failure = task.exception()
        if failure is None:
            return
        _LOGGER.critical(
            "runtime.task game_id=%s seat=%s task=player error=%s: %s",
            self._game_id.value,
            seat.value,
            type(failure).__name__,
            failure,
            exc_info=(
                type(failure),
                failure,
                failure.__traceback__,
            ),
        )

    async def _serve(self) -> None:
        while True:
            message = await self._messages.get()
            match message:
                case _PlayerReady(seat=seat, completed=completed):
                    assert seat not in self._ready_seats
                    self._ready_seats.add(seat)
                    self._ready_waiters[seat] = completed
                    if len(self._ready_seats) == 4:
                        for waiter in self._ready_waiters.values():
                            waiter.set_result(None)
                case _PlayerInitialized(
                    seat=seat,
                ):
                    assert seat in self._ready_seats
                    assert seat not in self._initialized_seats
                    self._initialized_seats.add(seat)
                    if len(self._initialized_seats) == 4:
                        self._initialized.set()
                        _LOGGER.info(
                            "game.started game_id=%s seq=%d",
                            self._game_id.value,
                            self._seq,
                        )
                case _ViewRequested(seat=seat):
                    await self._send(seat, error=None)
                case _CommandSubmitted(
                    seat=seat,
                    seq=seq,
                    decoder=decoder,
                ):
                    await self._receive(seat, seq, decoder)
                case _FailureReported(
                    seat=seat,
                    failure=failure,
                ):
                    await self._fail(seat, failure)
                case _CloseRequested(completed=completed):
                    self._closing = True
                    completed.set_result(None)
                    _LOGGER.info(
                        "game.closed game_id=%s seq=%d",
                        self._game_id.value,
                        self._seq,
                    )
                    return
                case _:
                    assert_never(message)

    async def _receive(
        self,
        seat: Seat,
        seq: int,
        decoder: CommandDecoder,
    ) -> None:
        failure = self._failure
        if failure is not None:
            await self._send(seat, error=failure.error)
            return
        if seq != self._seq:
            _LOGGER.info(
                "game.command game_id=%s seat=%s seq=%d "
                + "current_seq=%d error=stale sequence",
                self._game_id.value,
                seat.value,
                seq,
                self._seq,
            )
            await self._send(seat, error=None)
            return
        decoded = decoder.decode()
        if isinstance(decoded, CommandRejected):
            _LOGGER.info(
                "game.command game_id=%s seat=%s seq=%d "
                + "kind=invalid error=%s",
                self._game_id.value,
                seat.value,
                seq,
                decoded.reason,
            )
            await self._send(seat, error=decoded.reason)
            return
        command = decoded.value
        result = apply(self._state, seat, command)
        if isinstance(result, CommandRejected):
            _LOGGER.info(
                "game.command game_id=%s seat=%s seq=%d %s error=%s",
                self._game_id.value,
                seat.value,
                seq,
                _command_log_fields(command),
                result.reason,
            )
            await self._send(seat, error=result.reason)
            return
        assert isinstance(result, Ok)
        self._state = result.value
        self._seq += 1
        _LOGGER.info(
            "game.command game_id=%s seat=%s seq=%d next_seq=%d %s",
            self._game_id.value,
            seat.value,
            seq,
            self._seq,
            _command_log_fields(command),
        )
        await self._broadcast()

    async def _fail(
        self,
        seat: Seat,
        failure: PlayerFailure,
    ) -> None:
        assert self._failure is None
        self._failure = failure
        _LOGGER.error(
            "game.session game_id=%s seat=%s seq=%d error=%s",
            self._game_id.value,
            seat.value,
            self._seq,
            failure.error,
        )
        await self._broadcast()

    async def _broadcast(self) -> None:
        for seat, _player in self._players.items():
            await self._send(seat, error=None)

    async def _send(
        self,
        seat: Seat,
        *,
        error: str | None,
    ) -> None:
        if seat not in self._ready_seats:
            return
        failure = self._failure
        await self._channels.at(seat).queue.put(
            PlayerView(
                viewer=seat,
                seq=self._seq,
                snapshot=observe(self._state, seat),
                status=("failed" if failure is not None else "running"),
                error=failure.error if failure is not None else error,
            )
        )


@final
class _SeatPort(PlayerPort):
    """Seat-bound mailbox capability backed by one Session."""

    def __init__(
        self,
        *,
        game_id: GameId,
        seat: Seat,
        messages: asyncio.Queue[_SessionMessage],
    ) -> None:
        self._game_id = game_id
        self._seat = seat
        self._messages = messages

    @property
    @override
    def game_id(self) -> GameId:
        return self._game_id

    @override
    async def player_ready(self) -> None:
        completed = _completion_future()
        self._messages.put_nowait(_PlayerReady(self._seat, completed))
        await completed

    @override
    async def player_initialized(self) -> None:
        self._messages.put_nowait(_PlayerInitialized(self._seat))

    @override
    async def request_view(self) -> None:
        self._messages.put_nowait(_ViewRequested(self._seat))

    @override
    async def submit(
        self,
        seq: int,
        decoder: CommandDecoder,
    ) -> None:
        self._messages.put_nowait(
            _CommandSubmitted(self._seat, seq, decoder)
        )

    @override
    async def report_failure(self, failure: PlayerFailure) -> None:
        self._messages.put_nowait(_FailureReported(self._seat, failure))


def _completion_future() -> asyncio.Future[None]:
    return asyncio.get_running_loop().create_future()


def _command_log_fields(command: commands.Command) -> str:
    match command:
        case commands.ConfirmRound():
            return "kind=next_round"
        case commands.RevealBid(card_ids=card_ids):
            return _cards_log_fields("bid", card_ids)
        case commands.PassBid():
            return "kind=bid pass=true"
        case commands.Bury(card_ids=card_ids):
            return _cards_log_fields("discard", card_ids)
        case commands.Stir(card_ids=card_ids):
            return _cards_log_fields("stir", card_ids)
        case commands.PassStir():
            return "kind=stir pass=true"
        case commands.Play(card_ids=card_ids):
            return _cards_log_fields("play", card_ids)
        case _:
            assert_never(command)


def _cards_log_fields(kind: str, card_ids: tuple[CardId, ...]) -> str:
    cards = ",".join(str(card_id) for card_id in card_ids)
    return f"kind={kind} cards={cards}"


__all__ = ("Session", "SessionTaskOwner")
