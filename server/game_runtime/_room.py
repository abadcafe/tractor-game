"""Lobby orchestration and one immutable runtime session."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import final

from server.foundation.result import Ok, Rejected
from server.game import GameConfig, GameSeed, Seat, seats

from ._roster import (
    BotPlayerFactory,
    RoomAlreadyStarted,
    SeatRoster,
)
from ._session import Session, SessionTaskOwner
from .player import (
    BotPolicyName,
    CommandDecoder,
    HumanTransport,
    PlayerDescription,
    UserId,
)
from .registry import GameId

__all__ = (
    "GameRoom",
    "RoomClosed",
    "SeatStatus",
)

_LOGGER = logging.getLogger(__name__)


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


@final
class GameRoom:
    """Coordinate a pregame roster and its single game session."""

    def __init__(
        self,
        *,
        game_id: GameId,
        config: GameConfig,
        seed: GameSeed,
        bot_factory: BotPlayerFactory,
        task_owner: SessionTaskOwner,
    ) -> None:
        self._game_id = game_id
        self._config = config
        self._seed = seed
        self._roster = SeatRoster(bot_factory)
        self._task_owner = task_owner
        self._lifecycle: _Lifecycle = _Lobby()
        _LOGGER.info(
            "game.created game_id=%s seed=%d",
            game_id.value,
            seed.value,
        )

    @property
    def game_id(self) -> GameId:
        """Return the immutable identity used by runtime boundaries."""
        return self._game_id

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
            _LOGGER.info(
                "game.roster game_id=%s seat=%s "
                + "operation=occupy error=%s",
                self._game_id.value,
                seat.value,
                result.reason,
            )
            return result
        _LOGGER.info(
            "game.roster game_id=%s seat=%s operation=occupy",
            self._game_id.value,
            seat.value,
        )
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
        result = self._roster.vacate(seat=seat, user_id=user_id)
        if isinstance(result, Rejected):
            _LOGGER.info(
                "game.roster game_id=%s seat=%s "
                + "operation=vacate error=%s",
                self._game_id.value,
                seat.value,
                result.reason,
            )
            return result
        _LOGGER.info(
            "game.roster game_id=%s seat=%s operation=vacate",
            self._game_id.value,
            seat.value,
        )
        return result

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
        result = self._roster.fill_bots(
            policy=policy,
            user_id=user_id,
        )
        if isinstance(result, Rejected):
            _LOGGER.info(
                "game.roster game_id=%s operation=fill_bots "
                + "policy=%s error=%s",
                self._game_id.value,
                policy,
                result.reason,
            )
            return result
        _LOGGER.info(
            "game.roster game_id=%s operation=fill_bots policy=%s",
            self._game_id.value,
            policy,
        )
        return result

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
                game_id=self._game_id,
                config=self._config,
                seed=self._seed,
                players=players_result.value,
            )
            await session.start(self._task_owner)
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
            self._roster.close_unstarted()
            _LOGGER.info(
                "game.closed game_id=%s seq=not_started",
                self._game_id.value,
            )
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
