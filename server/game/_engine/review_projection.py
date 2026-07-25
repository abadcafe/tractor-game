"""Project one completed round for review or terminal display."""

from server.game.rules.cards import Rank
from server.game.seating import Partnership, Seat, SeatMap
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
    winning_partnership: Partnership | None,
) -> PlayerSnapshot:
    return PlayerSnapshot(
        phase="WAITING",
        round_number=completed.round_number,
        hand=(),
        remaining_cards=SeatMap(a=0, b=0, c=0, d=0),
        bottom_cards=completed.bottom_cards,
        trump=contract_snapshot(completed.contract),
        declarer=completed.contract.declarer,
        defender_points=completed.defender_points,
        trick=None,
        last_completed_trick=completed_trick(completed.last_trick),
        defender_point_cards=completed.defender_point_cards,
        awaiting_action=(
            "next_round"
            if winning_partnership is None and viewer not in confirmed
            else None
        ),
        scoring=ScoringSnapshot(
            winning_partnership=completed.winning_partnership,
            defender_points=completed.defender_points,
            total_defender_points=completed.total_defender_points,
            bottom_card_bonus=completed.bottom_card_bonus,
            bottom_cards=completed.bottom_cards,
        ),
        winning_partnership=winning_partnership,
        partnership_levels=completed.levels_after,
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
