"""Scoring module: pure calculation from inputs to RoundResult.

Computes bottom card points, ambush multiplier, total defender points,
level changes, and declarer rotation per spec section 9.

This is NOT a state machine -- it is a pure function.
"""

from pydantic import BaseModel, ConfigDict

from server.sm.card_model import Card, Suit, Rank
from server.sm.constants import (
    SCORE_THRESHOLDS,
    advance_level,
    get_partner_index,
    get_team_index,
    next_player_ccw,
)
from server.sm.types import CompletedTrick
from server.sm.play_rules import decompose


class RoundResult(BaseModel):
    """Result of scoring a round."""

    model_config = ConfigDict(frozen=True)

    team0_new_level: Rank
    team1_new_level: Rank
    next_declarer_team: int
    next_declarer_player: int
    total_defender_points: int
    declarer_level_change: int
    defender_level_change: int = 0
    switch_declarer: bool
    bottom_card_bonus: int


def _compute_ambush_multiplier(
    last_trick: CompletedTrick,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> int:
    """Compute the ambush multiplier based on the last trick's lead cards.

    Uses decompose to determine the lead play structure from the lead cards.
    For a single sub-play: single (pair_count=0) -> x2, pair (pair_count=1) -> x4,
    tractor (pair_count>=2) -> 2^(card_count).
    For multiple sub-plays (throw): take the max multiplier across sub-plays.
    """
    lead_cards = _find_lead_cards(last_trick)
    if not lead_cards:
        return 2

    subs = decompose(lead_cards, trump_suit, trump_rank)

    if len(subs) == 0:
        return 2

    if len(subs) == 1:
        sub = subs[0]
        if sub.pair_count == 0:
            return 2  # single
        elif sub.pair_count == 1:
            return 4  # pair
        else:
            return 2 ** len(sub.cards)  # tractor

    # Multiple sub-plays: it's a throw -> max multiplier across sub-plays
    best = 2
    for sub in subs:
        if sub.pair_count == 0:
            best = max(best, 2)
        elif sub.pair_count == 1:
            best = max(best, 4)
        else:
            best = max(best, 2 ** len(sub.cards))
    return best


def _find_lead_cards(last_trick: CompletedTrick) -> list[Card]:
    """Find the lead player's cards from the trick slots."""
    for slot in last_trick.slots:
        if slot.player == last_trick.lead_player:
            return slot.cards
    # Fallback: if lead_player's slot not found, use any non-empty slot
    for slot in last_trick.slots:
        if slot.cards:
            return slot.cards
    return []


def _determine_level_change(total_points: int) -> tuple[int, bool, int]:
    """Return (declarer_level_change, switch_declarer, defender_level_gain).

    Levels never retreat. When defenders score >= 80, the declarer team
    gets 0 change and the defender team (who becomes the new declarer)
    gains levels according to the formula: max(0, (pts - 80) // 40).

    For defenders < 80, the threshold table is used (declarer gains,
    no switch, defender gets 0).
    """
    for threshold in SCORE_THRESHOLDS:
        if total_points <= threshold.max_points:
            return (threshold.declarer_change, threshold.switch_declarer, 0)

    # Defenders >= 80: switch declarer, new declarer gains levels
    defender_gain = max(0, (total_points - 80) // 40)
    return (0, True, defender_gain)


def calculate_score(
    defender_points: int,
    bottom_cards: list[Card],
    last_trick: CompletedTrick,
    declarer_team: int,
    declarer_player: int,
    team0_level: Rank,
    team1_level: Rank,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> RoundResult:
    """Calculate the score result for a round.

    Pure function: takes inputs, returns RoundResult.
    """
    defender_team = 1 - declarer_team

    # Compute bottom card base points
    bottom_base = sum(card.points for card in bottom_cards)

    # Determine if defender won the last trick (ambush)
    last_trick_winner_team = get_team_index(last_trick.winner)
    is_ambush = last_trick_winner_team == defender_team

    # Compute ambush bonus
    if is_ambush:
        multiplier = _compute_ambush_multiplier(last_trick, trump_suit, trump_rank)
        bottom_card_bonus = bottom_base * multiplier
    else:
        bottom_card_bonus = 0

    total_defender_points = defender_points + bottom_card_bonus

    # Determine level change from scoring table
    declarer_change, switch, defender_gain = _determine_level_change(total_defender_points)

    # Compute new levels (levels never retreat)
    if switch:
        # Declarer team: 0 change; defender team (new declarer): +defender_gain
        if declarer_team == 0:
            team0_new_level = advance_level(team0_level, 0)
            team1_new_level = advance_level(team1_level, defender_gain)
        else:
            team0_new_level = advance_level(team0_level, defender_gain)
            team1_new_level = advance_level(team1_level, 0)
    else:
        # Declarer team: +declarer_change; defender team: 0 change
        if declarer_team == 0:
            team0_new_level = advance_level(team0_level, declarer_change)
            team1_new_level = advance_level(team1_level, 0)
        else:
            team0_new_level = advance_level(team0_level, 0)
            team1_new_level = advance_level(team1_level, declarer_change)

    # Determine next declarer
    if switch:
        next_team = defender_team
        next_player = next_player_ccw(declarer_player)
    else:
        next_team = declarer_team
        next_player = get_partner_index(declarer_player)

    return RoundResult(
        team0_new_level=team0_new_level,
        team1_new_level=team1_new_level,
        next_declarer_team=next_team,
        next_declarer_player=next_player,
        total_defender_points=total_defender_points,
        declarer_level_change=declarer_change,
        defender_level_change=defender_gain if switch else 0,
        switch_declarer=switch,
        bottom_card_bonus=bottom_card_bonus,
    )
