"""Sequenced runtime around one immutable game and four players."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import partial

from server.foundation.result import Ok
from server.game import (
    CommandRejected,
    GameConfig,
    GameSeed,
    GameState,
    Seat,
    SeatMap,
    apply,
    create,
    observe,
)

from .player import (
    CommandDecoder,
    Player,
    PlayerPort,
    PlayerView,
)

__all__ = ("Session",)


class Session:
    """Own game state, sequence gating, and personalized views."""

    def __init__(
        self,
        config: GameConfig,
        seed: GameSeed,
        players: SeatMap[Player],
    ) -> None:
        self._state: GameState = create(config, seed)
        self._seq = 1
        self._players = players
        self._closed = False
        self._started: set[Seat] = set()

    async def start(self) -> None:
        """Start players sequentially through their seat channels."""
        assert not self._started
        for seat, player in self._players.items():
            self._started.add(seat)
            await player.start(
                _SeatPort(
                    request_view=partial(self._request_view, seat),
                    submit=partial(self._receive, seat),
                )
            )

    def view(self, seat: Seat) -> PlayerView:
        """Return the current view without changing sequence."""
        return PlayerView(
            viewer=seat,
            seq=self._seq,
            snapshot=observe(self._state, seat),
            error=None,
        )

    async def close(self) -> None:
        """Stop all four players exactly once."""
        if self._closed:
            return
        self._closed = True
        for player in self._players.values():
            await player.stop()

    async def _request_view(self, seat: Seat) -> None:
        if self._closed:
            return
        await self._send(seat, error=None)

    async def _receive(
        self,
        seat: Seat,
        seq: int,
        decoder: CommandDecoder,
    ) -> None:
        if self._closed:
            return
        if seq != self._seq:
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

    async def _broadcast(self) -> None:
        for seat, _player in self._players.items():
            await self._send(seat, error=None)

    async def _send(
        self,
        seat: Seat,
        *,
        error: str | None,
    ) -> None:
        if seat not in self._started:
            return
        await self._players.at(seat).update(
            PlayerView(
                viewer=seat,
                seq=self._seq,
                snapshot=observe(self._state, seat),
                error=error,
            )
        )


class _SeatPort(PlayerPort):
    """Seat-bound capability backed by one Session."""

    def __init__(
        self,
        *,
        request_view: Callable[[], Awaitable[None]],
        submit: Callable[[int, CommandDecoder], Awaitable[None]],
    ) -> None:
        self._request_view = request_view
        self._submit = submit

    async def request_view(self) -> None:
        await self._request_view()

    async def submit(
        self,
        seq: int,
        decoder: CommandDecoder,
    ) -> None:
        await self._submit(seq, decoder)
