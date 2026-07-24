"""Complete state visible to one player."""

from __future__ import annotations

from pydantic import computed_field

from server.game.config import Seat
from server.game.rules.cards import Card, Rank, Suit

from ._base import (
    AwaitingAction,
    HandCounts,
    RoundPhase,
    SnapshotModel,
    TeamLevels,
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
    player_hand: tuple[Card, ...]
    player_hand_counts: HandCounts
    bottom_cards: tuple[Card, ...]
    trump: TrumpSnapshot
    declarer_player: Seat | None
    defender_points: int
    trick: TrickSnapshot | None
    last_completed_trick: CompletedTrickSnapshot | None
    defender_point_cards: tuple[Card, ...]
    awaiting_action: AwaitingAction | None
    scoring: ScoringSnapshot | None
    winning_team: int | None
    team_levels: TeamLevels
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

    @computed_field
    @property
    def declarer_team(self) -> int | None:
        """Absolute partnership derived from the declarer seat."""
        if self.declarer_player is None:
            return None
        return int(self.declarer_player) % 2

    @computed_field
    @property
    def team0_level(self) -> Rank:
        """Level of the even-seat partnership."""
        return self.team_levels[0]

    @computed_field
    @property
    def team1_level(self) -> Rank:
        """Level of the odd-seat partnership."""
        return self.team_levels[1]
