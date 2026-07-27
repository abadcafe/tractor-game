"""The complete public interface of the Tractor game domain."""

from __future__ import annotations

from server.foundation.result import Ok, Rejected

from . import commands, snapshots
from ._engine.aggregate import (
    apply_command,
    create_game,
    create_round_from_deal,
)
from ._engine.observation import observe_game
from ._engine.state import GameState
from .commands import Command
from .config import GameConfig, GameSeed
from .positions import RoundDeal
from .rejections import CommandRejected
from .seating import (
    Partnership,
    PartnershipId,
    PartnershipMap,
    Seat,
    SeatId,
    SeatMap,
    next_seat,
    partner_seat,
    partnership_id,
    partnership_of,
    previous_seat,
    seat_from_id,
    seat_id,
    seats,
)
from .snapshots import PlayerSnapshot


def create(config: GameConfig, seed: GameSeed) -> GameState:
    """Create a new immutable game."""
    return create_game(config, seed)


def apply(
    state: GameState,
    actor: Seat,
    command: Command,
) -> Ok[GameState] | CommandRejected:
    """Apply one command and return the latest immutable state."""
    return apply_command(state, actor, command)


def observe(state: GameState, viewer: Seat) -> PlayerSnapshot:
    """Derive the complete reconnect-safe view for one player."""
    return observe_game(state, viewer)


def instantiate(
    position: RoundDeal,
) -> Ok[GameState] | Rejected:
    """Create an immutable state from a complete round position."""
    return create_round_from_deal(position)


__all__ = [
    "CommandRejected",
    "GameConfig",
    "GameSeed",
    "GameState",
    "Partnership",
    "PartnershipId",
    "PartnershipMap",
    "RoundDeal",
    "Seat",
    "SeatId",
    "SeatMap",
    "next_seat",
    "partner_seat",
    "partnership_id",
    "partnership_of",
    "previous_seat",
    "seat_from_id",
    "seat_id",
    "apply",
    "commands",
    "create",
    "instantiate",
    "observe",
    "seats",
    "snapshots",
]
