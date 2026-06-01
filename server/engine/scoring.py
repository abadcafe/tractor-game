"""Scoring and level advancement for 升级 (Shengji/Tractor).

Handles round scoring calculation, level advancement, and scoring/level constants.
Ported from src/engine/scoring.ts with Bug #4 and Bug #6 fixes.

Bug #4: Ambush multiplier is now SINGLE x2, PAIR x4, TRACTOR x8 (was fixed x2).
Bug #6: is_game_over uses >= (game ends when reaching A, not passing A).
"""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from server.engine.card import Card, Rank
from server.engine.card_utils import POINTS_MAP
from server.engine.constants import LEVELS
from server.engine.types import PlayType


# ---- Constants ----

TARGET_LEVEL: Rank = Rank.ACE


@dataclass(frozen=True)
class ScoreThreshold:
    """One row of the scoring table."""
    max_points: int
    declarer_change: int
    switch_declarer: bool


SCORE_TABLE: list[ScoreThreshold] = [
    ScoreThreshold(max_points=0,   declarer_change=3,  switch_declarer=False),
    ScoreThreshold(max_points=40,  declarer_change=2,  switch_declarer=False),
    ScoreThreshold(max_points=80,  declarer_change=1,  switch_declarer=False),
    ScoreThreshold(max_points=120, declarer_change=0,  switch_declarer=True),
    ScoreThreshold(max_points=160, declarer_change=-1, switch_declarer=True),
    ScoreThreshold(max_points=200, declarer_change=-2, switch_declarer=True),
]

DEFAULT_SETTINGS: dict = {
    "model": "gpt-4o",
    "base_url": "https://api.openai.com/v1",
    "target_level": Rank.ACE,
    "bottom_card_count": 8,
}


# ---- Result Models ----


class ScoreResult(BaseModel):
    """Result of scoring a round."""

    model_config = ConfigDict(frozen=True)

    declarer_level_change: int
    switch_declarer: bool
    bottom_card_bonus: int
    total_defender_points: int
    team0_new_level: Rank
    team1_new_level: Rank


# ---- Scoring Logic ----


_AMBUSH_MULTIPLIER: dict[PlayType, int] = {
    PlayType.SINGLE: 2,
    PlayType.PAIR: 4,
    PlayType.TRACTOR: 8,
    PlayType.THROW: 2,
}


def _bottom_card_points(bottom_cards: list[Card]) -> int:
    """Sum point values of bottom cards using POINTS_MAP."""
    return sum(POINTS_MAP[card.rank] for card in bottom_cards)


def _determine_level_change(total_points: int) -> tuple[int, bool]:
    """Return (declarer_level_change, switch_declarer) from total defender points.

    Scoring table:
      0 pts           → declarer +3
      1-39 pts        → declarer +2
      40-79 pts       → declarer +1
      80-119 pts      → switch declarer
      120-159 pts     → defender +1
      160-199 pts     → defender +2
      200 pts         → defender +3
    """
    if total_points == 0:
        return (3, False)
    if total_points < 40:
        return (2, False)
    if total_points < 80:
        return (1, False)
    if total_points < 120:
        return (0, True)
    if total_points < 160:
        return (-1, True)
    if total_points < 200:
        return (-2, True)
    return (-3, True)


def _advance_level(current: Rank, change: int) -> Rank:
    """Advance a level by change steps, clamping to LEVELS bounds."""
    idx = LEVELS.index(current)
    new_idx = max(0, min(len(LEVELS) - 1, idx + change))
    return LEVELS[new_idx]


def calculate_score(
    defender_points: int,
    bottom_cards: list[Card],
    last_trick_winner_team: int,
    last_trick_play_type: PlayType,
    declarer_team_index: int,
    declarer_team_level: Rank,
    defender_team_level: Rank,
) -> ScoreResult:
    """Calculate the score result for a round.

    Args:
        defender_points: Points collected by defender from tricks.
        bottom_cards: The 8 bottom cards (底牌).
        last_trick_winner_team: Which team won the last trick (0 or 1).
        last_trick_play_type: Play type of the last trick.
        declarer_team_index: Which team was declarer this round (0 or 1).
        declarer_team_level: Current level of the declarer team.
        defender_team_level: Current level of the defender team.

    Returns:
        ScoreResult with level changes, new levels, and point breakdown.
    """
    defender_team_index = 1 - declarer_team_index

    # Calculate bottom card bonus (扣底)
    bp = _bottom_card_points(bottom_cards)
    ambush = last_trick_winner_team == defender_team_index
    multiplier = _AMBUSH_MULTIPLIER.get(last_trick_play_type, 2) if ambush else 0
    bottom_card_bonus = bp * multiplier

    total_points = defender_points + bottom_card_bonus

    # Determine level change
    declarer_change, switch_declarer = _determine_level_change(total_points)

    # Calculate new levels independently for each team
    if declarer_team_index == 0:
        team0_new_level = _advance_level(declarer_team_level, declarer_change)
        team1_new_level = defender_team_level
    else:
        team0_new_level = declarer_team_level
        team1_new_level = _advance_level(declarer_team_level, declarer_change)

    return ScoreResult(
        declarer_level_change=declarer_change,
        switch_declarer=switch_declarer,
        bottom_card_bonus=bottom_card_bonus,
        total_defender_points=total_points,
        team0_new_level=team0_new_level,
        team1_new_level=team1_new_level,
    )


def is_game_over(level: Rank, target_level: Rank = TARGET_LEVEL) -> bool:
    """Check if a team has reached or passed the target level (game over).

    Bug #6 fix: uses >= instead of > (reaching A is sufficient).
    """
    return LEVELS.index(level) >= LEVELS.index(target_level)
