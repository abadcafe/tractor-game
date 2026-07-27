"""Complete position inputs accepted by the pure game engine."""

from __future__ import annotations

from dataclasses import dataclass

from server.game.config import GameConfig, GameSeed
from server.game.rules.cards import Card, Rank
from server.game.seating import PartnershipMap, Seat


@dataclass(frozen=True, slots=True)
class RoundDeal:
    """A complete shuffled round before its first bid decision."""

    config: GameConfig
    continuation_seed: GameSeed
    round_number: int
    partnership_levels: PartnershipMap[Rank]
    fixed_declarer: Seat | None
    start_actor: Seat
    shuffled_cards: tuple[Card, ...]

    def __post_init__(self) -> None:
        assert self.round_number > 0


__all__ = ("RoundDeal",)
