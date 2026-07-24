"""Direct pure-game self-play runner for training."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from server.foundation import result as _result
from server.game import (
    CommandRejected,
    GameConfig,
    GameSeed,
    GameState,
    Seat,
    apply,
    commands,
    create,
    observe,
)
from server.game.rules.progression import TerminalProgress
from server.game.snapshots import PlayerSnapshot
from server.training.player import TrainingDecision, TrainingPlayer
from server.training.policy import TrainingPolicy
from server.training.progress import (
    RoundScore,
    TeamProgress,
    zero_sum_rewards,
)
from server.training.returns import ReturnCommit, terminal_return_commit
from server.training.trajectory import TrajectoryRecorder

_SEATS: tuple[Seat, Seat, Seat, Seat] = (
    Seat.NORTH,
    Seat.WEST,
    Seat.SOUTH,
    Seat.EAST,
)
_MODEL_ACTIONS = frozenset(("bid", "stir", "discard", "play"))


@dataclass(frozen=True, slots=True)
class TrainingRoundResult:
    """One completed self-play round."""

    returns: ReturnCommit
    team0_reward: float
    team1_reward: float
    generated_action_count: int
    accepted_action_count: int
    action_choice_count: int
    average_action_choices: float
    elapsed_seconds: float
    game_over: bool


class SelfPlaySession:
    """A long-lived pure game used for consecutive training rounds."""

    def __init__(self, *, policy: TrainingPolicy) -> None:
        self._recorder = TrajectoryRecorder()
        self._players = tuple(
            TrainingPlayer(
                index=int(seat),
                policy=policy,
                recorder=self._recorder,
            )
            for seat in _SEATS
        )
        self._state: GameState | None = None
        self._seed: int | None = None
        self._seq = 0

    async def play_round(
        self,
        *,
        base_seed: int,
        policy_version: int,
        rollout_id: str,
        episode_id: int,
        max_seconds: float,
    ) -> _result.Ok[TrainingRoundResult] | _result.Rejected:
        """Play exactly one round from the current game."""
        assert base_seed >= 0
        assert policy_version >= 0
        assert rollout_id
        assert episode_id >= 0
        assert max_seconds > 0.0
        self._initialize_game(base_seed)
        state = self._required_state()
        assert observe(state, Seat.NORTH).winning_team is None

        self._recorder.clear()
        for player in self._players:
            player.reset_round_tracking(
                base_seed=base_seed,
                policy_version=policy_version,
                rollout_id=rollout_id,
                episode_id=episode_id,
            )
        observed = self._broadcast()
        if isinstance(observed, _result.Rejected):
            return observed

        start = time.monotonic()
        confirmed = self._confirm_round()
        if isinstance(confirmed, _result.Rejected):
            return confirmed
        before = observe(self._required_state(), Seat.NORTH)

        final_result = await self._play_until_scoring(
            start=start,
            max_seconds=max_seconds,
        )
        if isinstance(final_result, _result.Rejected):
            return final_result
        final_snapshot = final_result.value
        reward0, reward1 = round_rewards(
            before=before,
            after=final_snapshot,
        )
        returns = terminal_return_commit(
            policy_version=policy_version,
            episode_id=episode_id,
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
                returns=returns,
                team0_reward=reward0,
                team1_reward=reward1,
                generated_action_count=generated_count,
                accepted_action_count=accepted_count,
                action_choice_count=choice_count,
                average_action_choices=(
                    0.0
                    if generated_count == 0
                    else choice_count / generated_count
                ),
                elapsed_seconds=max(
                    time.monotonic() - start,
                    0.000001,
                ),
                game_over=final_snapshot.winning_team is not None,
            )
        )

    async def close(self) -> None:
        """Close a session that owns no background work."""

    def _initialize_game(self, base_seed: int) -> None:
        if self._state is not None:
            assert self._seed == base_seed
            return
        self._seed = base_seed
        self._state = create(
            GameConfig(),
            GameSeed(base_seed),
        )

    def _required_state(self) -> GameState:
        state = self._state
        assert state is not None
        return state

    def _confirm_round(self) -> _result.Ok[None] | _result.Rejected:
        for seat in _SEATS:
            snapshot = observe(self._required_state(), seat)
            assert snapshot.awaiting_action == "next_round"
            accepted = self._apply(
                actor=seat,
                command=commands.ConfirmRound(),
            )
            if isinstance(accepted, _result.Rejected):
                return accepted
        return _result.Ok(value=None)

    async def _play_until_scoring(
        self,
        *,
        start: float,
        max_seconds: float,
    ) -> _result.Ok[PlayerSnapshot] | _result.Rejected:
        while True:
            snapshot = observe(self._required_state(), Seat.NORTH)
            if (
                snapshot.phase == "WAITING"
                and snapshot.scoring is not None
            ):
                return _result.Ok(value=snapshot)
            remaining = max_seconds - (time.monotonic() - start)
            if remaining <= 0.0:
                return _timed_out(max_seconds)
            actor = self._awaited_actor()
            player = self._players[int(actor)]
            decision_result = await _decide_before(
                player=player,
                seq=self._seq,
                snapshot=observe(self._required_state(), actor),
                seconds=remaining,
            )
            if isinstance(decision_result, _result.Rejected):
                return decision_result
            decision = decision_result.value
            accepted = self._apply(
                actor=actor,
                command=decision.command,
            )
            if isinstance(accepted, _result.Rejected):
                raise AssertionError(
                    "training legal action was rejected: "
                    f"player={int(actor)}, seq={self._seq}, "
                    f"error={accepted.reason}, "
                    f"action={decision.step.action!r}"
                )
            player.accept(decision)

    def _awaited_actor(self) -> Seat:
        awaited = [
            seat
            for seat in _SEATS
            if observe(
                self._required_state(),
                seat,
            ).awaiting_action
            in _MODEL_ACTIONS
        ]
        assert len(awaited) == 1
        return awaited[0]

    def _apply(
        self,
        *,
        actor: Seat,
        command: commands.Command,
    ) -> _result.Ok[None] | _result.Rejected:
        result = apply(self._required_state(), actor, command)
        if isinstance(result, CommandRejected):
            return _result.Rejected(reason=result.reason)
        self._state = result.value
        self._seq += 1
        return self._broadcast()

    def _broadcast(self) -> _result.Ok[None] | _result.Rejected:
        state = self._required_state()
        for seat in _SEATS:
            result = self._players[int(seat)].observe(
                seq=self._seq,
                snapshot=observe(state, seat),
            )
            if isinstance(result, _result.Rejected):
                return result
        return _result.Ok(value=None)


async def _decide_before(
    *,
    player: TrainingPlayer,
    seq: int,
    snapshot: PlayerSnapshot,
    seconds: float,
) -> _result.Ok[TrainingDecision] | _result.Rejected:
    task = asyncio.create_task(
        player.decide(seq=seq, snapshot=snapshot)
    )
    done, _ = await asyncio.wait((task,), timeout=seconds)
    if done:
        return task.result()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return _timed_out(seconds)


def _timed_out(seconds: float) -> _result.Rejected:
    return _result.Rejected(
        reason=f"training round timed out after {seconds:g} seconds"
    )


async def play_training_round(
    *,
    policy: TrainingPolicy,
    base_seed: int,
    policy_version: int,
    rollout_id: str,
    episode_id: int,
    max_seconds: float,
) -> _result.Ok[TrainingRoundResult] | _result.Rejected:
    """Run one self-play round in a fresh session."""
    session = SelfPlaySession(policy=policy)
    return await session.play_round(
        base_seed=base_seed,
        policy_version=policy_version,
        rollout_id=rollout_id,
        episode_id=episode_id,
        max_seconds=max_seconds,
    )


def round_rewards(
    *,
    before: PlayerSnapshot,
    after: PlayerSnapshot,
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
        level=(
            TerminalProgress.WIN
            if after.winning_team == 0
            else after.team0_level
        ),
        is_declarer=round_winning_team == 0,
    )
    team1_after = TeamProgress(
        level=(
            TerminalProgress.WIN
            if after.winning_team == 1
            else after.team1_level
        ),
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
            mandatory_levels=before.mandatory_levels,
        ),
    )
    return reward.team0, reward.team1


__all__ = (
    "SelfPlaySession",
    "TrainingRoundResult",
    "play_training_round",
    "round_rewards",
)
