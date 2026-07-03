"""Game constants for 升级 (Shengji/Tractor) state machines.

Defines player positioning, team mapping, counterclockwise rotation,
card counts, and scoring thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

# ---- Player Positioning ----

PLAYER_COUNT: int = 4

# ---- Team Mapping ----

# Team 0: North(0) + South(2)
TEAM_0: tuple[int, ...] = (0, 2)
# Team 1: West(1) + East(3)
TEAM_1: tuple[int, ...] = (1, 3)

# ---- Counterclockwise Rotation ----

# Player indexes are assigned in counterclockwise player order:
# North(0) → West(1) → South(2) → East(3) → North(0)
CCW_NEXT: MappingProxyType[int, int] = MappingProxyType(
    {0: 1, 1: 2, 2: 3, 3: 0}
)


def _validate_player(player: int) -> None:
    """Raise ValueError if player is not in [0, PLAYER_COUNT)."""
    if player < 0 or player >= PLAYER_COUNT:
        raise ValueError(
            f"Invalid player index {player!r}; must be an int in [0,"
            f"{PLAYER_COUNT})"
        )


def next_player_ccw(current: int) -> int:
    """Return the next player in counterclockwise order."""
    _validate_player(current)
    return CCW_NEXT[current]


def get_team_index(player: int) -> int:
    """Return 0 or 1 indicating which team the player belongs to."""
    _validate_player(player)
    return player % 2


def get_partner_index(player: int) -> int:
    """Return the partner (对家) of the given player."""
    _validate_player(player)
    return (player + 2) % PLAYER_COUNT


# ---- Card Counts ----

BOTTOM_CARD_COUNT: int = 8
TOTAL_CARDS: int = 108  # 2 decks × 54
TOTAL_POINTS: int = 200  # 2 decks × (4 suits × (5+10+10))

# ---- Scoring Thresholds ----


@dataclass(frozen=True)
class ScoreThreshold:
    """A single row of the scoring lookup table.

    ``max_points`` is the upper bound (inclusive) of the defender-score
    range that this threshold covers.  Thresholds must be checked in
    order — the first whose ``max_points >= defender_score`` wins.
    """

    max_points: int
    declarer_change: int
    switch_declarer: bool


# Spec section 9, lookup table:
# 闲家得分     庄家级别变化  闲家(新庄)级别变化  换庄
# 0            +3           0                   否
# 1~39         +2           0                   否
# 40~79        +1           0                   否
# ≥80          0            (得分-80)//40       是
#
# 级别永不倒退。闲家≥80分时，庄家换人，新庄家（原闲家）升级数
# = max(0, (闲家得分 - 80) // 40)，上不封顶。
SCORE_THRESHOLDS: tuple[ScoreThreshold, ...] = (
    ScoreThreshold(
        max_points=0, declarer_change=3, switch_declarer=False
    ),
    ScoreThreshold(
        max_points=39, declarer_change=2, switch_declarer=False
    ),
    ScoreThreshold(
        max_points=79, declarer_change=1, switch_declarer=False
    ),
)
