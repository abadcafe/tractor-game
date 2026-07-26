"""Lobby orchestration and one immutable runtime session."""

from __future__ import annotations

from dataclasses import dataclass

from server.foundation.result import Ok, Rejected
from server.game import GameConfig, GameSeed, Seat, seats

from ._roster import (
    BotPlayerFactory,
    RoomAlreadyStarted,
    SeatRoster,
)
from ._session import Session
from .player import (
    BotPolicyName,
    CommandDecoder,
    HumanTransport,
    PlayerDescription,
    UserId,
)

__all__ = (
    "GameRoom",
    "RoomClosed",
    "SeatStatus",
)


class RoomClosed(Rejected):
    """The room has released all runtime resources."""

    def __init__(self) -> None:
        super().__init__("game closed")


@dataclass(frozen=True, slots=True)
class _Lobby:
    pass


@dataclass(frozen=True, slots=True)
class _Active:
    session: Session


@dataclass(frozen=True, slots=True)
class _Closed:
    pass


type _Lifecycle = _Lobby | _Active | _Closed


@dataclass(frozen=True, slots=True)
class SeatStatus:
    """Lobby projection for one stable seat."""

    seat: Seat
    player: PlayerDescription | None
    ready: bool


class GameRoom:
    """Coordinate a pregame roster and its single game session."""

    def __init__(
        self,
        config: GameConfig,
        seed: GameSeed,
        bot_factory: BotPlayerFactory,
    ) -> None:
        self._config = config
        self._seed = seed
        self._roster = SeatRoster(bot_factory)
        self._lifecycle: _Lifecycle = _Lobby()

    def started(self) -> bool:
        """Return whether this room currently owns a session."""
        return isinstance(self._lifecycle, _Active)

    def occupy_seat(
        self,
        *,
        seat: Seat,
        user_id: UserId,
    ) -> Ok[Seat] | Rejected:
        """Assign a human to an empty pregame seat."""
        unavailable = self._lobby_rejection()
        if unavailable is not None:
            return unavailable
        result = self._roster.occupy(seat=seat, user_id=user_id)
        if isinstance(result, Rejected):
            return result
        return Ok(seat)

    def vacate_seat(
        self,
        *,
        seat: Seat,
        user_id: UserId,
    ) -> Ok[None] | Rejected:
        """Remove a matching human before session start."""
        unavailable = self._lobby_rejection()
        if unavailable is not None:
            return unavailable
        return self._roster.vacate(seat=seat, user_id=user_id)

    def fill_bots(
        self,
        *,
        policy: BotPolicyName,
        user_id: UserId,
    ) -> Ok[None] | Rejected:
        """Fill every empty pregame seat with one bot policy."""
        unavailable = self._lobby_rejection()
        if unavailable is not None:
            return unavailable
        return self._roster.fill_bots(
            policy=policy,
            user_id=user_id,
        )

    async def connect_seat(
        self,
        *,
        seat: Seat,
        user_id: UserId,
        transport: HumanTransport,
    ) -> Ok[Seat] | Rejected:
        """Connect the matching human and start a full roster."""
        lifecycle = self._lifecycle
        if isinstance(lifecycle, _Closed):
            return RoomClosed()
        human_result = self._roster.human(
            seat=seat,
            user_id=user_id,
        )
        if isinstance(human_result, Rejected):
            return human_result
        human = human_result.value
        if isinstance(lifecycle, _Lobby):
            players_result = self._roster.freeze()
            if isinstance(players_result, Rejected):
                return players_result
            session = Session(
                self._config,
                self._seed,
                players_result.value,
            )
            await session.start()
            self._lifecycle = _Active(session)
            await human.connect(transport)
            return Ok(seat)
        await human.connect(transport)
        return Ok(seat)

    def disconnect_seat(
        self,
        *,
        seat: Seat,
        user_id: UserId,
        transport: HumanTransport,
    ) -> None:
        """Detach a stale transport without changing ownership."""
        if isinstance(self._lifecycle, _Closed):
            return
        human_result = self._roster.human(
            seat=seat,
            user_id=user_id,
        )
        if isinstance(human_result, Ok):
            human_result.value.disconnect(transport)

    async def receive(
        self,
        *,
        seat: Seat,
        user_id: UserId,
        transport: HumanTransport,
        seq: int,
        decoder: CommandDecoder,
    ) -> None:
        """Forward a frame only through the matching HumanPlayer."""
        if isinstance(self._lifecycle, _Closed):
            return
        human_result = self._roster.human(
            seat=seat,
            user_id=user_id,
        )
        if isinstance(human_result, Rejected):
            return
        await human_result.value.receive(
            transport,
            seq,
            decoder,
        )

    def seats(
        self,
        *,
        user_id: UserId | None = None,
    ) -> list[SeatStatus]:
        """Return all four lobby seat projections."""
        return [
            SeatStatus(
                seat=seat,
                player=self._roster.lobby_status(seat, user_id),
                ready=self._ready(seat),
            )
            for seat in seats()
        ]

    async def close(self) -> None:
        """Close the current lifecycle state exactly once."""
        lifecycle = self._lifecycle
        if isinstance(lifecycle, _Closed):
            return
        self._lifecycle = _Closed()
        if isinstance(lifecycle, _Lobby):
            await self._roster.close_unstarted()
            return
        await lifecycle.session.close()

    def _ready(self, seat: Seat) -> bool:
        lifecycle = self._lifecycle
        if not isinstance(lifecycle, _Active):
            return False
        return (
            seat
            in lifecycle.session.view(
                seat
            ).snapshot.next_round_confirmed
        )

    def _lobby_rejection(self) -> Rejected | None:
        lifecycle = self._lifecycle
        if isinstance(lifecycle, _Lobby):
            return None
        if isinstance(lifecycle, _Closed):
            return RoomClosed()
        return RoomAlreadyStarted()
