"""Game state machine for 升级 (Shengji/Tractor).

Top-level state machine that manages the full game lifecycle:
team levels, round results, and game-over determination.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from server.result import Ok, Rejected
from server.rules.cards import Rank
from server.rules.required_progress import (
    TeamAdvance,
    advance_team_progress,
)

from .rejections.game import (
    CannotProcessRoundResultRejected,
    CannotStartGameRejected,
)
from .scoring import RoundResult, assert_round_result_invariants


class GameOverResult(BaseModel):
    """Result when the game is over."""

    model_config = ConfigDict(frozen=True)

    winning_team: int
    final_levels: list[Rank]  # 2 elements: [team0_level, team1_level]


class GameState(BaseModel):
    """State of the full game."""

    model_config = ConfigDict(frozen=True)

    team0_level: Rank
    team1_level: Rank
    declarer_team: int | None
    next_declarer_player: int | None
    winning_team: int | None
    round_number: int


def create_game() -> GameState:
    """Create a new game waiting for the first round to start."""
    return GameState(
        team0_level=Rank.TWO,
        team1_level=Rank.TWO,
        declarer_team=None,
        next_declarer_player=None,
        winning_team=None,
        round_number=0,
    )


def start_game(state: GameState) -> Ok[GameState] | Rejected:
    """Start the game.

    Sets both team levels to TWO and round_number to 1.

    Returns Ok(new_state) on success, Rejected(reason) if the game
    already
    started or has ended.
    """
    if state.round_number != 0 or state.winning_team is not None:
        return CannotStartGameRejected()
    return Ok(
        state.model_copy(
            update={
                "team0_level": Rank.TWO,
                "team1_level": Rank.TWO,
                "round_number": 1,
            }
        )
    )


def process_round_result(
    state: GameState, result: RoundResult
) -> Ok[GameState] | Rejected:
    """Process a round result and update game state.

    Applies raw scoring gains through the rules-level required progress
    plan. Only the team that started the round as declarer can pass the
    virtual WIN target.

    Returns Ok(new_state) on success, Rejected(reason) if the game has
    not
    started or has already ended.
    """
    if state.round_number <= 0 or state.winning_team is not None:
        return CannotProcessRoundResultRejected()
    assert_round_result_invariants(result)
    assert (
        state.declarer_team is None
        or state.declarer_team == result.declarer_team
    )

    team0_gain = _level_gain_for_team(result, 0)
    team1_gain = _level_gain_for_team(result, 1)
    team0_advance = advance_team_progress(
        level=state.team0_level,
        raw_gain=team0_gain,
        was_declarer=result.declarer_team == 0,
    )
    team1_advance = advance_team_progress(
        level=state.team1_level,
        raw_gain=team1_gain,
        was_declarer=result.declarer_team == 1,
    )
    winning = _winning_team(team0_advance, team1_advance)

    if winning is not None:
        return Ok(
            state.model_copy(
                update={
                    "team0_level": team0_advance.level,
                    "team1_level": team1_advance.level,
                    "winning_team": winning,
                    "declarer_team": None,
                    "next_declarer_player": None,
                }
            )
        )

    # Game continues
    return Ok(
        state.model_copy(
            update={
                "team0_level": team0_advance.level,
                "team1_level": team1_advance.level,
                "declarer_team": result.round_winning_team,
                "next_declarer_player": result.next_declarer_player,
                "round_number": state.round_number + 1,
            }
        )
    )


def _level_gain_for_team(result: RoundResult, team: int) -> int:
    """
    Return the positive level gain awarded to *team* by a round result.
    """
    if result.round_winning_team != team:
        return 0
    if result.switch_declarer:
        return result.defender_level_gain
    return result.declarer_level_gain


def _winning_team(
    team0_advance: TeamAdvance,
    team1_advance: TeamAdvance,
) -> int | None:
    if team0_advance.won_game:
        assert not team1_advance.won_game
        return 0
    if team1_advance.won_game:
        return 1
    return None
