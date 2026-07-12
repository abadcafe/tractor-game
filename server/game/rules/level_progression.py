"""Level progression rules."""

from __future__ import annotations

from server.game.rules.cards import Rank

LEVELS: tuple[Rank, ...] = (
    Rank.TWO,
    Rank.THREE,
    Rank.FOUR,
    Rank.FIVE,
    Rank.SIX,
    Rank.SEVEN,
    Rank.EIGHT,
    Rank.NINE,
    Rank.TEN,
    Rank.JACK,
    Rank.QUEEN,
    Rank.KING,
    Rank.ACE,
)


def advance_level(level: Rank, change: int) -> Rank:
    """Advance a level by *change* steps, clamped to [TWO, ACE]."""
    idx = LEVELS.index(level)
    new_idx = max(0, min(len(LEVELS) - 1, idx + change))
    return LEVELS[new_idx]
