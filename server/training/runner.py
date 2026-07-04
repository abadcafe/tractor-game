"""Long-lived self-play session runner for training."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from server import result as _result
from server.game import Game
from server.protocol import StateSnapshot
from server.rules.required_progress import TerminalProgress
from server.training.player import TrainingPlayer
from server.training.policy import TrainingPolicy
from server.training.progress import (
    RoundScore,
    TeamProgress,
    zero_sum_rewards,
)
from server.training.terminal_rewards import terminal_reward_rollout
from server.training.trajectory import (
    RolloutBatch,
    TrajectoryRecorder,
)


@dataclass(frozen=True, slots=True)
class TrainingRoundResult:
    """One completed self-play round."""

    rollout: RolloutBatch
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
    ) -> None:
        self._recorder = TrajectoryRecorder()
        self._players = [
            TrainingPlayer(
                index=index,
                policy=policy,
                recorder=self._recorder,
            )
            for index in range(4)
        ]
        self._game = Game(players=self._players)
        self._started = False

    async def play_round(
        self, *, max_seconds: float
    ) -> _result.Ok[TrainingRoundResult] | _result.Rejected:
        """Play exactly one full round from the current Game state."""
        self._recorder.clear()
        for player in self._players:
            reset_result = player.reset_round_tracking()
            if isinstance(reset_result, _result.Rejected):
                return reset_result
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
        final_snapshot_result = await _wait_for_round_scoring(
            game=self._game,
            players=tuple(self._players),
            start=start,
            max_seconds=max_seconds,
        )
        if isinstance(final_snapshot_result, _result.Rejected):
            return final_snapshot_result
        final_snapshot = final_snapshot_result.value
        reward0, reward1 = round_rewards(
            before=before,
            after=final_snapshot,
        )
        rollout = terminal_reward_rollout(
            steps=self._recorder.steps(),
            team0_reward=reward0,
            team1_reward=reward1,
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
        return _result.Ok(
            value=TrainingRoundResult(
                rollout=rollout,
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
        )

    async def _confirm_next_round(self) -> None:
        for player in self._players:
            await player.confirm_held_scoring_next_round(self._game)


async def play_training_round(
    *,
    policy: TrainingPolicy,
    max_seconds: float,
) -> _result.Ok[TrainingRoundResult] | _result.Rejected:
    """Run one self-play round in a fresh session."""
    session = SelfPlaySession(policy=policy)
    return await session.play_round(max_seconds=max_seconds)


async def _wait_for_round_scoring(
    *,
    game: Game,
    players: tuple[TrainingPlayer, ...],
    start: float,
    max_seconds: float,
) -> _result.Ok[StateSnapshot] | _result.Rejected:
    while time.monotonic() - start < max_seconds:
        for player in players:
            background_result = player.raise_background_errors()
            if isinstance(background_result, _result.Rejected):
                return background_result
        snapshot = game.snapshot(for_player=0)
        if snapshot.phase == "WAITING" and snapshot.scoring is not None:
            return _result.Ok(value=snapshot)
        await asyncio.sleep(0.001)
    return _result.Rejected(
        reason=f"training round timed out after {max_seconds:g} seconds"
    )


def round_rewards(
    *,
    before: StateSnapshot,
    after: StateSnapshot,
) -> tuple[float, float]:
    """Return zero-sum team rewards for one completed round."""
    scoring = after.scoring
    assert scoring is not None
    round_winning_team = scoring.round_winning_team
    round_declarer_team = after.declarer_team
    assert round_declarer_team is not None
    if before.declarer_team is not None:
        assert before.declarer_team == round_declarer_team
    team0_before = TeamProgress(
        level=before.team0_level,
        is_declarer=round_declarer_team == 0,
    )
    team1_before = TeamProgress(
        level=before.team1_level,
        is_declarer=round_declarer_team == 1,
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
        score=RoundScore(
            declarer_team=round_declarer_team,
            total_defender_points=scoring.total_defender_points,
        ),
    )
    return reward.team0, reward.team1
