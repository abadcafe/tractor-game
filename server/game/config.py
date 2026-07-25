"""Strongly typed configuration for the game domain."""

from __future__ import annotations

from dataclasses import dataclass

from server.game.rules.cards import Rank, suited_ranks


@dataclass(frozen=True, slots=True)
class GameSeed:
    """Seed from which every round shuffle is derived."""

    value: int


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Rules that may vary without changing the game interface."""

    mandatory_levels: tuple[Rank, ...] = (Rank.ACE,)

    def __post_init__(self) -> None:
        assert self.mandatory_levels
        assert self.mandatory_levels[-1] == Rank.ACE
        assert len(set(self.mandatory_levels)) == len(
            self.mandatory_levels
        )
        assert all(
            rank in suited_ranks() for rank in self.mandatory_levels
        )
        positions = tuple(
            suited_ranks().index(rank) for rank in self.mandatory_levels
        )
        assert positions == tuple(sorted(positions))
