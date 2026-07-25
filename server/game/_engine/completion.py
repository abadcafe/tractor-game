"""Score a completed round and advance the game."""

from server.game.rules import progression, scoring
from server.game.seating import (
    Partnership,
    PartnershipMap,
    next_seat,
    partner_seat,
    partnership_of,
)

from . import phases
from .round_values import CompletedRound, InternalCompletedTrick
from .state import GameState, config_of, replace_phase


def finish_round(
    state: GameState,
    phase: phases.Playing,
    last_trick: InternalCompletedTrick,
) -> GameState:
    declarer_partnership = partnership_of(phase.contract.declarer)
    defenders_won_last = (
        partnership_of(last_trick.winner) != declarer_partnership
    )
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
    winning_partnership = (
        _opponent(declarer_partnership)
        if result.defenders_win
        else declarer_partnership
    )
    next_declarer = (
        next_seat(phase.contract.declarer)
        if result.defenders_win
        else partner_seat(phase.contract.declarer)
    )
    winning_gain = (
        result.defender_level_gain
        if result.defenders_win
        else result.declarer_level_gain
    )
    advances = PartnershipMap(
        first=progression.advance_team(
            level=phase.levels.at(Partnership.FIRST),
            raw_gain=(
                winning_gain
                if winning_partnership == Partnership.FIRST
                else 0
            ),
            was_declarer=(declarer_partnership == Partnership.FIRST),
            mandatory_levels=config_of(state).mandatory_levels,
        ),
        second=progression.advance_team(
            level=phase.levels.at(Partnership.SECOND),
            raw_gain=(
                winning_gain
                if winning_partnership == Partnership.SECOND
                else 0
            ),
            was_declarer=(declarer_partnership == Partnership.SECOND),
            mandatory_levels=config_of(state).mandatory_levels,
        ),
    )
    levels_after = PartnershipMap(
        first=advances.at(Partnership.FIRST).level,
        second=advances.at(Partnership.SECOND).level,
    )
    completed = CompletedRound(
        round_number=phase.round_number,
        contract=phase.contract,
        winning_partnership=winning_partnership,
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
    if advances.at(winning_partnership).won_game:
        return replace_phase(
            state,
            phases.Finished(
                completed=completed,
                winning_partnership=winning_partnership,
            ),
        )
    return replace_phase(
        state,
        phases.RoundReview(
            completed=completed,
            confirmed=frozenset(),
        ),
    )


def _opponent(partnership: Partnership) -> Partnership:
    if partnership == Partnership.FIRST:
        return Partnership.SECOND
    return Partnership.FIRST
