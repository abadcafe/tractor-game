"""Project one completed round for review or terminal display."""

from server.game.config import Seat
from server.game.rules.cards import Rank
from server.game.snapshots import PlayerSnapshot
from server.game.snapshots.review import ScoringSnapshot

from .contract_projection import contract_snapshot
from .history_projection import (
    bid_event,
    completed_trick,
    optional_bid_event,
    own_initial_exchange,
    stir_events,
)
from .round_values import CompletedRound


def completed_round_snapshot(
    completed: CompletedRound,
    viewer: Seat,
    confirmed: frozenset[Seat],
    mandatory_levels: tuple[Rank, ...],
    *,
    winning_team: int | None,
) -> PlayerSnapshot:
    return PlayerSnapshot(
        phase="WAITING",
        round_number=completed.round_number,
        player_hand=(),
        player_hand_counts=(0, 0, 0, 0),
        bottom_cards=completed.bottom_cards,
        trump=contract_snapshot(completed.contract),
        declarer_player=completed.contract.declarer,
        defender_points=completed.defender_points,
        trick=None,
        last_completed_trick=completed_trick(completed.last_trick),
        defender_point_cards=completed.defender_point_cards,
        awaiting_action=(
            "next_round"
            if winning_team is None and viewer not in confirmed
            else None
        ),
        scoring=ScoringSnapshot(
            round_winning_team=completed.winning_team,
            defender_points=completed.defender_points,
            total_defender_points=completed.total_defender_points,
            bottom_card_bonus=completed.bottom_card_bonus,
            bottom_cards=completed.bottom_cards,
        ),
        winning_team=winning_team,
        team_levels=completed.levels_after,
        mandatory_levels=mandatory_levels,
        bid_events=tuple(
            bid_event(event) for event in completed.bid_reveals
        ),
        bid_winner=optional_bid_event(completed.declaration),
        own_initial_bottom_exchange=own_initial_exchange(
            completed.exchange_events,
            viewer,
        ),
        stir_events=stir_events(
            completed.stir_events,
            completed.exchange_events,
            viewer,
        ),
        stirring_state=None,
        next_round_confirmed=confirmed,
    )
