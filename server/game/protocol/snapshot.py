"""Full player-facing state snapshot model."""

from __future__ import annotations

from server.game.protocol.bid_snapshot import BidEventSnapshot
from server.game.protocol.scoring_snapshot import ScoringSnapshot
from server.game.protocol.snapshot_common import (
    AwaitingAction,
    RoundPhase,
    SnapshotModel,
)
from server.game.protocol.stir_history_snapshot import (
    BottomExchangeSnapshot,
    StirDeclarationEventSnapshot,
)
from server.game.protocol.stirring_snapshot import StirringStateSnapshot
from server.game.protocol.trick_snapshot import (
    CompletedTrickSnapshot,
    TrickSnapshot,
)
from server.game.rules.cards import Card, Rank, Suit


class StateSnapshot(SnapshotModel):
    """Full player-facing state snapshot.

    ``action_hints`` is a complete closed hint set when non-empty. An
    empty list means no closed hint set is provided, not that the player
    has no legal action. Clients must still allow user input where the
    action type allows free card selection; the backend remains
    authoritative.
    """

    phase: RoundPhase
    player_hand: list[Card]
    player_hand_counts: list[int]
    bottom_cards: list[Card]
    trump_suit: Suit | None
    trump_rank: Rank
    declarer_team: int | None
    declarer_player: int | None
    defender_points: int
    trick: TrickSnapshot | None
    last_completed_trick: CompletedTrickSnapshot | None
    defender_point_cards: list[Card]
    action_hints: list[list[Card]]
    awaiting_action: AwaitingAction | None
    scoring: ScoringSnapshot | None
    winning_team: int | None
    team0_level: Rank
    team1_level: Rank
    bid_events: list[BidEventSnapshot]
    bid_winner: BidEventSnapshot | None
    own_initial_bottom_exchange: BottomExchangeSnapshot | None
    stir_events: list[StirDeclarationEventSnapshot]
    stirring_state: StirringStateSnapshot | None
    next_round_confirmed: list[int]
