"""Lobby ownership and runtime composition for one game."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol

from server.foundation.result import Ok, Rejected
from server.game import GameConfig, GameSeed, Seat
from server.game import seats as all_seats

from .session import (
    AgentSubmission,
    CommandDecoder,
    DeliverySink,
    DisconnectReason,
    Session,
)

__all__ = [
    "AgentFactory",
    "BotKind",
    "GameRoom",
    "RoomSeat",
    "RuntimeAgent",
]


class BotKind(str, Enum):
    """Bot implementations available to a room."""

    AI = "ai"
    AUTO = "auto"


class RuntimeAgent(DeliverySink, Protocol):
    """Lifecycle of a non-human runtime participant."""

    async def start(self) -> None:
        """Start background decision processing."""

    async def stop(self) -> None:
        """Stop background decision processing."""


class AgentFactory(Protocol):
    """Create one runtime agent for a seat."""

    def create(
        self,
        seat: Seat,
        kind: BotKind,
        submission: AgentSubmission,
    ) -> RuntimeAgent:
        """Create but do not start an agent."""
        ...


type OccupantKind = Literal["empty", "user", "ai", "auto"]


@dataclass(frozen=True, slots=True)
class RoomSeat:
    """Lobby state for one seat."""

    seat: Seat
    occupied: bool
    connected: bool
    kind: OccupantKind
    mine: bool
    ready: bool


@dataclass(frozen=True, slots=True)
class _User:
    user_id: str


@dataclass(frozen=True, slots=True)
class _Bot:
    kind: BotKind


type _Occupant = _User | _Bot


class GameRoom:
    """Own lobby assignments and lazily create one session."""

    def __init__(
        self,
        config: GameConfig,
        seed: GameSeed,
        agent_factory: AgentFactory,
    ) -> None:
        self._config = config
        self._seed = seed
        self._agent_factory = agent_factory
        self._occupants: dict[Seat, _Occupant] = {}
        self._session: Session | None = None
        self._agents: dict[Seat, RuntimeAgent] = {}

    def started(self) -> bool:
        """Return whether the game session has been created."""
        return self._session is not None

    async def occupy_seat(
        self,
        *,
        seat: Seat,
        user_id: str,
    ) -> Ok[Seat] | Rejected:
        """Assign a user to an unoccupied seat."""
        if not user_id.strip():
            return Rejected("missing user id")
        occupant = self._occupants.get(seat)
        if isinstance(occupant, _User):
            if occupant.user_id == user_id:
                return Ok(seat)
            return Rejected("seat occupied")
        if isinstance(occupant, _Bot):
            return Rejected("seat occupied")
        for other, value in tuple(self._occupants.items()):
            if (
                isinstance(value, _User)
                and value.user_id == user_id
                and other != seat
            ):
                await self._remove_user(other)
        self._occupants[seat] = _User(user_id)
        return Ok(seat)

    async def vacate_seat(
        self,
        *,
        seat: Seat,
        user_id: str,
    ) -> Ok[None] | Rejected:
        """Remove a matching user assignment."""
        if not user_id.strip():
            return Rejected("missing user id")
        occupant = self._occupants.get(seat)
        if not isinstance(occupant, _User):
            return Rejected("user does not occupy seat")
        if occupant.user_id != user_id:
            return Rejected("user does not occupy seat")
        await self._remove_user(seat)
        return Ok(None)

    async def fill_bots(
        self,
        *,
        kind: BotKind,
        user_id: str,
    ) -> Ok[None] | Rejected:
        """Fill empty seats after validating lobby ownership."""
        if not user_id.strip():
            return Rejected("missing user id")
        if not any(
            isinstance(value, _User) and value.user_id == user_id
            for value in self._occupants.values()
        ):
            return Rejected("user does not occupy a seat")
        return self.fill_empty_bots(kind=kind)

    def fill_empty_bots(
        self,
        *,
        kind: BotKind,
        preserve: Collection[Seat] = (),
    ) -> Ok[None] | Rejected:
        """Fill empty seats before a session starts."""
        if self._session is not None:
            return Rejected("game already started")
        preserved = set(preserve)
        for seat in all_seats():
            if seat in preserved or seat in self._occupants:
                continue
            self._occupants[seat] = _Bot(kind)
        return Ok(None)

    async def connect_seat(
        self,
        *,
        seat: Seat,
        user_id: str,
        sink: DeliverySink,
    ) -> Ok[Seat] | Rejected:
        """Connect an attached user sink to the game session."""
        attached = self._occupied_user(seat, user_id)
        if isinstance(attached, Rejected):
            return attached
        session = await self._ensure_session()
        if isinstance(session, Rejected):
            return session
        await session.value.attach(attached.value, sink)
        return attached

    def disconnect_seat(
        self,
        seat: Seat,
        sink: DeliverySink,
    ) -> None:
        """Detach a stale transport without changing seat ownership."""
        if self._session is not None:
            self._session.detach(seat, sink)

    async def receive(
        self,
        seat: Seat,
        sink: DeliverySink,
        seq: int,
        decoder: CommandDecoder,
    ) -> None:
        """Forward a frame only from the current connection owner."""
        session = self._session
        if session is None or not session.owns(seat, sink):
            return
        await session.receive(seat, seq, decoder)

    def seats(
        self,
        *,
        user_id: str | None = None,
    ) -> list[RoomSeat]:
        """Return all four lobby seats."""
        return [self._room_seat(seat, user_id) for seat in all_seats()]

    def agent_at(self, seat: Seat) -> RuntimeAgent | None:
        """Return a bot agent for diagnostics."""
        return self._agents.get(seat)

    async def close(self) -> None:
        """Stop agents and close the current session."""
        if self._session is not None:
            session = self._session
            self._session = None
            await session.close()
        agents = tuple(self._agents.values())
        self._agents.clear()
        for agent in agents:
            await agent.stop()

    def _occupied_user(
        self,
        seat: Seat,
        user_id: str,
    ) -> Ok[Seat] | Rejected:
        if not user_id.strip():
            return Rejected("missing user id")
        occupant = self._occupants.get(seat)
        if (
            not isinstance(occupant, _User)
            or occupant.user_id != user_id
        ):
            return Rejected("seat occupied")
        return Ok(seat)

    async def _ensure_session(self) -> Ok[Session] | Rejected:
        if self._session is not None:
            return Ok(self._session)
        if any(seat not in self._occupants for seat in all_seats()):
            return Rejected("not enough occupants")
        session = Session(self._config, self._seed)
        self._session = session
        for seat, occupant in self._occupants.items():
            if not isinstance(occupant, _Bot):
                continue
            agent = self._agent_factory.create(
                seat,
                occupant.kind,
                session.submission_for(seat),
            )
            self._agents[seat] = agent
            await session.attach(seat, agent)
            await agent.start()
        return Ok(session)

    async def _remove_user(self, seat: Seat) -> None:
        self._occupants.pop(seat, None)
        if self._session is not None:
            await self._session.disconnect(
                seat,
                DisconnectReason.DETACHED,
            )

    def _room_seat(
        self,
        seat: Seat,
        user_id: str | None,
    ) -> RoomSeat:
        occupant = self._occupants.get(seat)
        kind = _occupant_kind(occupant)
        return RoomSeat(
            seat=seat,
            occupied=occupant is not None,
            connected=(
                isinstance(occupant, _User)
                and self._session is not None
                and self._session.connected(seat)
            ),
            kind=kind,
            mine=(
                isinstance(occupant, _User)
                and user_id is not None
                and occupant.user_id == user_id
            ),
            ready=self._ready(seat),
        )

    def _ready(self, seat: Seat) -> bool:
        if self._session is None:
            return False
        snapshot = self._session.snapshot(seat).snapshot
        return seat in snapshot.next_round_confirmed


def _occupant_kind(occupant: _Occupant | None) -> OccupantKind:
    if isinstance(occupant, _User):
        return "user"
    if isinstance(occupant, _Bot):
        return occupant.kind.value
    return "empty"
