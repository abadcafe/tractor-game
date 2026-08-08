"""Immutable values produced by private play-pattern analysis."""

from __future__ import annotations

from dataclasses import dataclass

from server.game.rules._ordering import EffectiveSuit
from server.game.rules.cards import Card
from server.game.rules.cards.faces import CardFace

type PairTier = tuple[CardFace, ...]


@dataclass(frozen=True, slots=True)
class Pattern:
    """One disjoint single, pair, or tractor in a played card set."""

    cards: tuple[Card, ...]
    pair_faces: tuple[CardFace, ...]
    effective_suit: EffectiveSuit

    def __post_init__(self) -> None:
        assert self.cards
        if not self.pair_faces:
            assert len(self.cards) == 1
            return
        assert len(self.cards) == len(self.pair_faces) * 2

    @property
    def pair_count(self) -> int:
        """Return zero for a single and the pair count otherwise."""
        return len(self.pair_faces)

    @property
    def level(self) -> int:
        """Return single=1, pair=2, two-pair tractor=3, and so on."""
        return self.pair_count + 1


@dataclass(frozen=True, slots=True)
class PairRun:
    """Consecutive structural tiers, with equivalent faces per tier."""

    tiers: tuple[PairTier, ...]

    def __post_init__(self) -> None:
        assert len(self.tiers) >= 2
        assert all(tier for tier in self.tiers)


@dataclass(frozen=True, slots=True)
class PatternAnalysis:
    """Patterns for one non-empty effective-suit card set."""

    effective_suit: EffectiveSuit
    patterns: tuple[Pattern, ...]
    pair_faces: tuple[CardFace, ...]
    tractor_runs: tuple[PairRun, ...]

    def __post_init__(self) -> None:
        assert self.patterns
