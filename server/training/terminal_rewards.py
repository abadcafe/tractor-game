"""Terminal round reward assignment for self-play trajectories."""

from __future__ import annotations

import math

from server.sm.constants import get_team_index
from server.training.trajectory import (
    DecisionStep,
    DecisionTransition,
    RolloutBatch,
    TeamTrajectory,
)


def terminal_reward_rollout(
    *,
    steps: tuple[DecisionStep, ...],
    team0_reward: float,
    team1_reward: float,
) -> RolloutBatch:
    """Build a terminal-only rollout batch from one completed round."""
    assert math.isfinite(team0_reward)
    assert math.isfinite(team1_reward)
    assert team0_reward + team1_reward == 0.0
    trajectories: list[TeamTrajectory] = []
    for team_index, reward in ((0, team0_reward), (1, team1_reward)):
        team_steps = tuple(
            step
            for step in steps
            if get_team_index(step.player_index) == team_index
        )
        if team_steps:
            trajectories.append(
                TeamTrajectory(
                    team_index=team_index,
                    terminal_reward=reward,
                    transitions=_terminal_transitions(
                        steps=team_steps,
                    ),
                )
            )
    return RolloutBatch(trajectories=tuple(trajectories))


def _terminal_transitions(
    *,
    steps: tuple[DecisionStep, ...],
) -> tuple[DecisionTransition, ...]:
    return tuple(
        DecisionTransition(
            decision=step,
            reward_after_step=0.0,
        )
        for step in steps
    )
