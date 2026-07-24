"""Score a completed round and advance the game."""

from server.game.rules import progression, scoring

from . import phases
from .round_values import (
    CompletedRound,
    InternalCompletedTrick,
    TeamLevels,
)
from .state import GameState, config_of, replace_phase
from .topology import next_seat, partner, team


def finish_round(
    state: GameState,
    phase: phases.Playing,
    last_trick: InternalCompletedTrick,
) -> GameState:
    declarer_team = team(phase.contract.declarer)
    defenders_won_last = team(last_trick.winner) != declarer_team
    result = scoring.score_round(
        defender_points=sum(
            card.points for card in phase.defender_point_cards
        ),
        bottom_cards=phase.bottom_cards,
        last_trick_won_by_defenders=defenders_won_last,
        last_lead_cards=last_trick.slots[0].cards,
        trump_suit=phase.contract.trump_suit,
        trump_rank=phase.contract.trump_rank,
    )
    winning_team = (
        1 - declarer_team if result.defenders_win else declarer_team
    )
    next_declarer = (
        next_seat(phase.contract.declarer)
        if result.defenders_win
        else partner(phase.contract.declarer)
    )
    gains = [0, 0]
    gains[winning_team] = (
        result.defender_level_gain
        if result.defenders_win
        else result.declarer_level_gain
    )
    advances = tuple(
        progression.advance_team(
            level=phase.levels[team_index],
            raw_gain=gains[team_index],
            was_declarer=team_index == declarer_team,
            mandatory_levels=config_of(state).mandatory_levels,
        )
        for team_index in (0, 1)
    )
    levels_after: TeamLevels = (
        advances[0].level,
        advances[1].level,
    )
    completed = CompletedRound(
        round_number=phase.round_number,
        contract=phase.contract,
        winning_team=winning_team,
        next_declarer=next_declarer,
        defender_points=sum(
            card.points for card in phase.defender_point_cards
        ),
        total_defender_points=result.total_defender_points,
        bottom_card_bonus=result.bottom_card_bonus,
        bottom_cards=phase.bottom_cards,
        levels_before=phase.levels,
        levels_after=levels_after,
        last_trick=last_trick,
        declaration=phase.declaration,
        bid_reveals=phase.bid_reveals,
        stir_events=phase.stir_events,
        exchange_events=phase.exchange_events,
        defender_point_cards=phase.defender_point_cards,
    )
    if advances[winning_team].won_game:
        return replace_phase(
            state,
            phases.Finished(
                completed=completed,
                winning_team=winning_team,
            ),
        )
    return replace_phase(
        state,
        phases.RoundReview(
            completed=completed,
            confirmed=frozenset(),
        ),
    )
