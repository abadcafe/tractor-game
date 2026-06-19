"""Game state machine for 升级 (Shengji/Tractor).

Top-level state machine that manages the full game lifecycle:
team levels, round results, and game-over determination.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from server.sm.card_model import Rank
from server.sm.result import Ok, Rejected, StateResult
from server.sm.scoring import RoundResult


class GameOverResult(BaseModel):
    """Result when the game is over."""

    model_config = ConfigDict(frozen=True)

    winning_team: int
    final_levels: list[Rank]  # 2 elements: [team0_level, team1_level]


class GameState(BaseModel):
    """State of the full game."""

    model_config = ConfigDict(frozen=True)

    phase: Literal["IDLE", "IN_ROUND", "GAME_OVER"]
    team0_level: Rank
    team1_level: Rank
    declarer_team: int | None
    next_declarer_player: int | None
    winning_team: int | None
    round_number: int


def create_game() -> GameState:
    """Create a new game in IDLE phase with both teams at level TWO."""
    return GameState(
        phase="IDLE",
        team0_level=Rank.TWO,
        team1_level=Rank.TWO,
        declarer_team=None,
        next_declarer_player=None,
        winning_team=None,
        round_number=0,
    )


def start_game(state: GameState) -> StateResult[GameState]:
    """Start the game, transitioning IDLE -> IN_ROUND.

    Sets both team levels to TWO and round_number to 1.

    Returns Ok(new_state) on success, Rejected(reason) if not in IDLE phase.
    """
    if state.phase != "IDLE":
        return Rejected(
            f"不能在 {state.phase} 阶段开始游戏，需要 IDLE 阶段"
        )
    return Ok(state.model_copy(update={
        "phase": "IN_ROUND",
        "team0_level": Rank.TWO,
        "team1_level": Rank.TWO,
        "round_number": 1,
    }))


def process_round_result(state: GameState, result: RoundResult) -> StateResult[GameState]:
    """Process a round result and update game state.

    Updates team levels from the result. If either team reaches ACE,
    transitions to GAME_OVER. Otherwise stays in IN_ROUND with updated
    declarer info and incremented round_number.

    Returns Ok(new_state) on success, Rejected(reason) if not in IN_ROUND phase.
    """
    if state.phase != "IN_ROUND":
        return Rejected(
            f"不能在 {state.phase} 阶段处理回合结果，需要 IN_ROUND 阶段"
        )

    new_team0 = result.team0_new_level
    new_team1 = result.team1_new_level

    team0_gain = _level_gain_for_team(result, 0)
    team1_gain = _level_gain_for_team(result, 1)

    # Check game over: a team must already be playing ACE and then gain again.
    # Reaching ACE only schedules an ACE round; it is not a win yet.
    team0_passed_ace = state.team0_level == Rank.ACE and team0_gain > 0
    team1_passed_ace = state.team1_level == Rank.ACE and team1_gain > 0
    if team0_passed_ace or team1_passed_ace:
        winning = 0 if team0_passed_ace else 1
        return Ok(state.model_copy(update={
            "phase": "GAME_OVER",
            "team0_level": new_team0,
            "team1_level": new_team1,
            "winning_team": winning,
            "declarer_team": None,
            "next_declarer_player": None,
        }))

    # Game continues
    return Ok(state.model_copy(update={
        "team0_level": new_team0,
        "team1_level": new_team1,
        "declarer_team": result.next_declarer_team,
        "next_declarer_player": result.next_declarer_player,
        "round_number": state.round_number + 1,
    }))


def _level_gain_for_team(result: RoundResult, team: int) -> int:
    """Return the positive level gain awarded to *team* by a round result."""
    if result.next_declarer_team != team:
        return 0
    if result.switch_declarer:
        return result.defender_level_change
    return result.declarer_level_change
