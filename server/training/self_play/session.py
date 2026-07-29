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
    SeatMap,
    apply,
    commands,
    create,
    observe,
    seats,
)
from server.game.snapshots import PlayerSnapshot
from server.policy_model.return_target import round_return_targets
from server.training.self_play.actor import (
    SelfPlayActor,
    TrainingDecision,
)
from server.training.self_play.policy import TrainingPolicy
from server.training.self_play.returns import (
    ReturnCommit,
    terminal_return_commit,
)
from server.training.self_play.trajectory import TrajectoryRecorder

_MODEL_ACTIONS = frozenset(("bid", "stir", "discard", "play"))


@dataclass(frozen=True, slots=True)
class TrainingRoundResult:
    """One completed self-play round."""

    returns: ReturnCommit
    first_partnership_reward: float
    second_partnership_reward: float
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
        self._actors = SeatMap(
            a=SelfPlayActor(
                Seat.A, policy=policy, recorder=self._recorder
            ),
            b=SelfPlayActor(
                Seat.B, policy=policy, recorder=self._recorder
            ),
            c=SelfPlayActor(
                Seat.C, policy=policy, recorder=self._recorder
            ),
            d=SelfPlayActor(
                Seat.D, policy=policy, recorder=self._recorder
            ),
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
        assert observe(state, Seat.A).winning_partnership is None

        self._recorder.clear()
        for actor in self._actors.values():
            actor.reset_round_tracking(
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
        before = observe(self._required_state(), Seat.A)

        final_result = await self._play_until_scoring(
            start=start,
            max_seconds=max_seconds,
        )
        if isinstance(final_result, _result.Rejected):
            return final_result
        final_snapshot = final_result.value
        first_reward, second_reward = round_return_targets(
            before=before,
            after=final_snapshot,
        )
        returns = terminal_return_commit(
            policy_version=policy_version,
            episode_id=episode_id,
            steps=self._recorder.steps(),
            first_partnership_reward=first_reward,
            second_partnership_reward=second_reward,
        )
        generated_count = sum(
            actor.stats().generated_action_count
            for actor in self._actors.values()
        )
        accepted_count = sum(
            actor.stats().accepted_action_count
            for actor in self._actors.values()
        )
        choice_count = sum(
            actor.stats().action_choice_count
            for actor in self._actors.values()
        )
        return _result.Ok(
            value=TrainingRoundResult(
                returns=returns,
                first_partnership_reward=first_reward,
                second_partnership_reward=second_reward,
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
                game_over=final_snapshot.winning_partnership
                is not None,
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
        for seat in seats():
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
            snapshot = observe(self._required_state(), Seat.A)
            if (
                snapshot.phase == "WAITING"
                and snapshot.scoring is not None
            ):
                return _result.Ok(value=snapshot)
            remaining = max_seconds - (time.monotonic() - start)
            if remaining <= 0.0:
                return _timed_out(max_seconds)
            actor = self._awaited_actor()
            self_play_actor = self._actors.at(actor)
            decision_result = await _decide_before(
                actor=self_play_actor,
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
                    f"seat={actor.value}, seq={self._seq}, "
                    f"error={accepted.reason}, "
                    f"action={decision.step.action!r}"
                )
            self_play_actor.accept(decision)

    def _awaited_actor(self) -> Seat:
        awaited = [
            seat
            for seat in seats()
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
        for seat in seats():
            result = self._actors.at(seat).observe(
                seq=self._seq,
                snapshot=observe(state, seat),
            )
            if isinstance(result, _result.Rejected):
                return result
        return _result.Ok(value=None)


async def _decide_before(
    *,
    actor: SelfPlayActor,
    seq: int,
    snapshot: PlayerSnapshot,
    seconds: float,
) -> _result.Ok[TrainingDecision] | _result.Rejected:
    task = asyncio.create_task(actor.decide(seq=seq, snapshot=snapshot))
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


__all__ = (
    "SelfPlaySession",
    "TrainingRoundResult",
    "play_training_round",
)
