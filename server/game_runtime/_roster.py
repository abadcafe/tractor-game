"""Pregame ownership of the four player seats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from server.foundation.result import Ok, Rejected
from server.game import Seat, SeatMap, seats

from .player import (
    BotPolicyName,
    HumanPlayer,
    Player,
    PlayerDescription,
    UserId,
)

__all__ = (
    "BotPlayerFactory",
    "RoomAlreadyStarted",
    "SeatRoster",
)


class RoomAlreadyStarted(Rejected):
    """The immutable runtime roster has already been frozen."""

    def __init__(self) -> None:
        super().__init__("game already started")


class RoomIncomplete(Rejected):
    """Not every seat has a runtime player."""

    def __init__(self) -> None:
        super().__init__("not enough players")


class SeatOccupied(Rejected):
    """A requested seat already has a player."""

    def __init__(self) -> None:
        super().__init__("seat occupied")


class UserDoesNotOccupySeat(Rejected):
    """The supplied user does not own the requested seat."""

    def __init__(self) -> None:
        super().__init__("user does not occupy seat")


class BotPlayerFactory(Protocol):
    """Create one automatic player for a requested policy."""

    def create(
        self,
        seat: Seat,
        policy_name: BotPolicyName,
    ) -> Player:
        """Create one unstarted bot player."""
        ...


@dataclass(frozen=True, slots=True)
class _HumanBinding:
    """Private UserId index entry for a human-controlled seat."""

    seat: Seat
    player: HumanPlayer


class SeatRoster:
    """Own mutable pregame seat assignments and freeze them once."""

    def __init__(self, bot_factory: BotPlayerFactory) -> None:
        self._bot_factory = bot_factory
        self._players: dict[Seat, Player] = {}
        self._humans: dict[UserId, _HumanBinding] = {}
        self._frozen = False

    def occupy(
        self,
        *,
        seat: Seat,
        user_id: UserId,
    ) -> Ok[HumanPlayer] | Rejected:
        """Assign one human before the game starts."""
        if self._frozen:
            return RoomAlreadyStarted()
        current = self._humans.get(user_id)
        existing = self._players.get(seat)
        if current is not None and current.seat == seat:
            return Ok(current.player)
        if existing is not None:
            return SeatOccupied()
        if current is not None:
            self._players.pop(current.seat)
            self._humans.pop(user_id)
        player = HumanPlayer(user_id)
        self._players[seat] = player
        self._humans[user_id] = _HumanBinding(
            seat=seat,
            player=player,
        )
        return Ok(player)

    def vacate(
        self,
        *,
        seat: Seat,
        user_id: UserId,
    ) -> Ok[None] | Rejected:
        """Remove a matching human before the game starts."""
        if self._frozen:
            return RoomAlreadyStarted()
        assignment = self._humans.get(user_id)
        if assignment is None or assignment.seat != seat:
            return UserDoesNotOccupySeat()
        self._humans.pop(user_id)
        self._players.pop(seat)
        return Ok(None)

    def fill_bots(
        self,
        *,
        policy: BotPolicyName,
        user_id: UserId,
    ) -> Ok[None] | Rejected:
        """Fill every empty seat after validating a human owner."""
        if self._frozen:
            return RoomAlreadyStarted()
        if user_id not in self._humans:
            return UserDoesNotOccupySeat()
        for seat in seats():
            if seat not in self._players:
                self._players[seat] = self._bot_factory.create(
                    seat,
                    policy,
                )
        return Ok(None)

    def human(
        self,
        *,
        seat: Seat,
        user_id: UserId,
    ) -> Ok[HumanPlayer] | Rejected:
        """Return a matching human through the ownership index."""
        assignment = self._humans.get(user_id)
        if assignment is None or assignment.seat != seat:
            return UserDoesNotOccupySeat()
        return Ok(assignment.player)

    def lobby_status(
        self,
        seat: Seat,
        requester: UserId | None,
    ) -> PlayerDescription | None:
        """Return the lobby projection for one seat."""
        player = self._players.get(seat)
        if player is None:
            return None
        return player.lobby_status(requester)

    def freeze(self) -> Ok[SeatMap[Player]] | Rejected:
        """Freeze and return a total four-player seat map."""
        if self._frozen:
            return RoomAlreadyStarted()
        if any(seat not in self._players for seat in seats()):
            return RoomIncomplete()
        self._frozen = True
        return Ok(
            SeatMap(
                a=self._players[Seat.A],
                b=self._players[Seat.B],
                c=self._players[Seat.C],
                d=self._players[Seat.D],
            )
        )

    async def close_unstarted(self) -> None:
        """Stop every player when a room closes before session start."""
        assert not self._frozen
        players = tuple(self._players.values())
        self._players.clear()
        self._humans.clear()
        for player in players:
            await player.stop()
