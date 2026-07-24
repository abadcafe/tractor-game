"""The complete public interface of the Tractor game domain."""

from __future__ import annotations

from server.foundation.result import Ok

from . import commands, snapshots
from ._engine.aggregate import apply_command, create_game
from ._engine.observation import observe_game
from ._engine.state import GameState
from .commands import Command
from .config import GameConfig, GameSeed, Seat
from .rejections import CommandRejected
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


__all__ = [
    "CommandRejected",
    "GameConfig",
    "GameSeed",
    "GameState",
    "Seat",
    "apply",
    "commands",
    "create",
    "observe",
    "snapshots",
]
