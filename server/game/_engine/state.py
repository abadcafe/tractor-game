"""Opaque immutable game-state storage."""

from __future__ import annotations

from typing import ClassVar

from server.game.config import GameConfig, GameSeed

from .phases import GamePhase


class GameState:
    """Opaque state threaded through the public game operations."""

    __slots__: ClassVar[tuple[str, ...]] = ()


class _GameState(GameState):
    __slots__: ClassVar[tuple[str, ...]] = (
        "_config",
        "_phase",
        "_seed",
    )

    _config: GameConfig
    _phase: GamePhase
    _seed: GameSeed

    def __init__(
        self,
        config: GameConfig,
        seed: GameSeed,
        phase: GamePhase,
    ) -> None:
        self._config = config
        self._seed = seed
        self._phase = phase

    def parts(
        self,
    ) -> tuple[GameConfig, GameSeed, GamePhase]:
        """Return storage parts to private engine operations."""
        return (self._config, self._seed, self._phase)


def make_state(
    config: GameConfig,
    seed: GameSeed,
    phase: GamePhase,
) -> GameState:
    return _GameState(config, seed, phase)


def replace_phase(state: GameState, phase: GamePhase) -> GameState:
    config, seed, _ = _parts(state)
    return _GameState(config, seed, phase)


def config_of(state: GameState) -> GameConfig:
    return _parts(state)[0]


def seed_of(state: GameState) -> GameSeed:
    return _parts(state)[1]


def phase_of(state: GameState) -> GamePhase:
    return _parts(state)[2]


def _parts(
    state: GameState,
) -> tuple[GameConfig, GameSeed, GamePhase]:
    assert isinstance(state, _GameState)
    return state.parts()
