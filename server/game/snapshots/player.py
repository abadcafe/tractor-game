"""Complete state visible to one player."""

from __future__ import annotations

from pydantic import computed_field

from server.game.rules.cards import Card, Rank, Suit
from server.game.seating import Partnership, Seat

from ._base import (
    AwaitingAction,
    PartnershipLevels,
    RemainingCards,
    RoundPhase,
    SnapshotModel,
)
from .contract import TrumpSnapshot, trump_rank, trump_suit
from .events import (
    BidEventSnapshot,
    BottomExchangeSnapshot,
    StirDeclarationEventSnapshot,
    StirringStateSnapshot,
)
from .review import ScoringSnapshot
from .tricks import CompletedTrickSnapshot, TrickSnapshot

__all__ = ["AwaitingAction", "PlayerSnapshot", "RoundPhase"]


class PlayerSnapshot(SnapshotModel):
    """A complete reconnect-safe projection for one player."""

    phase: RoundPhase
    round_number: int
    hand: tuple[Card, ...]
    remaining_cards: RemainingCards
    bottom_cards: tuple[Card, ...]
    trump: TrumpSnapshot
    declarer: Seat | None
    defender_points: int
    trick: TrickSnapshot | None
    last_completed_trick: CompletedTrickSnapshot | None
    defender_point_cards: tuple[Card, ...]
    awaiting_action: AwaitingAction | None
    scoring: ScoringSnapshot | None
    winning_partnership: Partnership | None
    partnership_levels: PartnershipLevels
    mandatory_levels: tuple[Rank, ...]
    bid_events: tuple[BidEventSnapshot, ...]
    bid_winner: BidEventSnapshot | None
    own_initial_bottom_exchange: BottomExchangeSnapshot | None
    stir_events: tuple[StirDeclarationEventSnapshot, ...]
    stirring_state: StirringStateSnapshot | None
    next_round_confirmed: frozenset[Seat]

    @computed_field
    @property
    def trump_rank(self) -> Rank:
        """Rank of the pending or established contract."""
        return trump_rank(self.trump)

    @computed_field
    @property
    def trump_suit(self) -> Suit | None:
        """Established suit; ``None`` also represents no-trump."""
        return trump_suit(self.trump)
