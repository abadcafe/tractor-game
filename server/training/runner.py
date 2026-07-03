"""Long-lived self-play session runner for training."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from server.game import Game
from server.protocol import StateSnapshot
from server.sm.constants import get_team_index
from server.sm.required_progress import (
    DEFAULT_REQUIRED_LEVEL_PLAN,
    RequiredLevelPlan,
    TerminalProgress,
)
from server.training.player import TrainingPlayer
from server.training.policy import TrainingPolicy
from server.training.progress import (
    TeamProgress,
    zero_sum_rewards,
)
from server.training.trajectory import (
    RewardedDecisionStep,
    TrajectoryRecorder,
)


@dataclass(frozen=True, slots=True)
class TrainingRoundResult:
    """One completed self-play round."""

    rewarded_steps: tuple[RewardedDecisionStep, ...]
    team0_reward: float
    team1_reward: float
    generated_action_count: int
    accepted_action_count: int
    average_action_choices: float
    elapsed_seconds: float
    game_over: bool


class SelfPlaySession:
    """A long-lived Game used for many per-round training episodes."""

    def __init__(
        self,
        *,
        policy: TrainingPolicy,
        required_level_plan: RequiredLevelPlan = (
            DEFAULT_REQUIRED_LEVEL_PLAN
        ),
    ) -> None:
        self._recorder = TrajectoryRecorder()
        self._players = [
            TrainingPlayer(
                index=index,
                policy=policy,
                required_level_plan=required_level_plan,
                recorder=self._recorder,
            )
            for index in range(4)
        ]
        self._game = Game(
            players=self._players,
            required_level_plan=required_level_plan,
        )
        self._required_level_plan = required_level_plan
        self._started = False

    async def play_round(
        self, *, max_seconds: float
    ) -> TrainingRoundResult:
        """Play exactly one full round from the current Game state."""
        self._recorder.clear()
        for player in self._players:
            player.reset_round_tracking()
        start = time.monotonic()
        if self._started:
            await self._confirm_next_round()
            before = self._game.snapshot(for_player=0)
        else:
            before = self._game.snapshot(for_player=0)
            self._started = True
            await asyncio.gather(
                *(player.run(self._game) for player in self._players)
            )
        final_snapshot = await _wait_for_round_scoring(
            game=self._game,
            players=tuple(self._players),
            start=start,
            max_seconds=max_seconds,
        )
        reward0, reward1 = round_rewards(
            before=before,
            after=final_snapshot,
            required_level_plan=self._required_level_plan,
        )
        rewarded_steps = tuple(
            RewardedDecisionStep(
                step=step,
                reward=reward0
                if get_team_index(step.player_index) == 0
                else reward1,
            )
            for step in self._recorder.steps()
        )
        generated_count = sum(
            player.stats().generated_action_count
            for player in self._players
        )
        accepted_count = sum(
            player.stats().accepted_action_count
            for player in self._players
        )
        choice_count = sum(
            player.stats().action_choice_count
            for player in self._players
        )
        return TrainingRoundResult(
            rewarded_steps=rewarded_steps,
            team0_reward=reward0,
            team1_reward=reward1,
            generated_action_count=generated_count,
            accepted_action_count=accepted_count,
            average_action_choices=0.0
            if generated_count == 0
            else choice_count / generated_count,
            elapsed_seconds=max(time.monotonic() - start, 0.000001),
            game_over=final_snapshot.winning_team is not None,
        )

    async def _confirm_next_round(self) -> None:
        for player in self._players:
            await player.confirm_held_scoring_next_round(self._game)


async def play_training_round(
    *,
    policy: TrainingPolicy,
    max_seconds: float,
    required_level_plan: RequiredLevelPlan = (
        DEFAULT_REQUIRED_LEVEL_PLAN
    ),
) -> TrainingRoundResult:
    """Run one self-play round in a fresh session."""
    session = SelfPlaySession(
        policy=policy,
        required_level_plan=required_level_plan,
    )
    return await session.play_round(max_seconds=max_seconds)


async def _wait_for_round_scoring(
    *,
    game: Game,
    players: tuple[TrainingPlayer, ...],
    start: float,
    max_seconds: float,
) -> StateSnapshot:
    while time.monotonic() - start < max_seconds:
        for player in players:
            player.raise_background_errors()
        snapshot = game.snapshot(for_player=0)
        if snapshot.phase == "WAITING" and snapshot.scoring is not None:
            return snapshot
        await asyncio.sleep(0.001)
    raise AssertionError("training round timed out")


def round_rewards(
    *,
    before: StateSnapshot,
    after: StateSnapshot,
    required_level_plan: RequiredLevelPlan,
) -> tuple[float, float]:
    """Return zero-sum team rewards for one completed round."""
    round_winning_team = _round_winning_team(after)
    team0_before = TeamProgress(
        level=before.team0_level,
        is_declarer=before.declarer_team == 0,
    )
    team1_before = TeamProgress(
        level=before.team1_level,
        is_declarer=before.declarer_team == 1,
    )
    team0_after = TeamProgress(
        level=TerminalProgress.WIN
        if after.winning_team == 0
        else after.team0_level,
        is_declarer=round_winning_team == 0,
    )
    team1_after = TeamProgress(
        level=TerminalProgress.WIN
        if after.winning_team == 1
        else after.team1_level,
        is_declarer=round_winning_team == 1,
    )
    reward = zero_sum_rewards(
        team0_before=team0_before,
        team1_before=team1_before,
        team0_after=team0_after,
        team1_after=team1_after,
        required_level_plan=required_level_plan,
    )
    return reward.team0, reward.team1


def _round_winning_team(snapshot: StateSnapshot) -> int | None:
    scoring = snapshot.scoring
    if scoring is not None:
        return scoring.round_winning_team
    return snapshot.declarer_team
