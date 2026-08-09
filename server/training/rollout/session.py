"""Pure-game rollout session for model partnership versus Auto."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import final

from server.foundation import result as _result
from server.game import (
    CommandRejected,
    GameConfig,
    GameSeed,
    Partnership,
    Seat,
    apply,
    commands,
    create,
    observe,
    partnership_of,
    seats,
)
from server.game.snapshots import PlayerSnapshot
from server.game_auto import choose_auto_command
from server.policy_model.return_target import round_return_targets
from server.training.rollout.actor import ActorDecision, ModelActor
from server.training.rollout.identity import GameIdentity
from server.training.rollout.policy import RolloutPolicy
from server.training.rollout.trajectory import CompletedRoundTrajectory

_STRATEGIC_ACTIONS = frozenset(("bid", "stir", "discard", "play"))


@dataclass(frozen=True, slots=True)
class RolloutRoundResult:
    """One complete round and its atomic model trajectory."""

    trajectory: CompletedRoundTrajectory
    model_reward: float
    auto_reward: float
    model_action_count: int
    auto_action_count: int
    forced_action_count: int
    trainable_decision_count: int
    legal_choice_count: int
    scored_choice_step_count: int
    model_declarer: bool
    elapsed_seconds: float
    game_over: bool


@final
class RolloutSession:
    """One long-lived game with fixed model and Auto partnerships."""

    def __init__(
        self, *, policy: RolloutPolicy, identity: GameIdentity
    ) -> None:
        self.identity = identity
        self.model_partnership = identity.model_partnership()
        self._model_actors = tuple(
            ModelActor(seat, policy=policy)
            for seat in seats()
            if partnership_of(seat) == self.model_partnership
        )
        assert len(self._model_actors) == 2
        self._auto_random_sources = tuple(
            (
                seat,
                random.Random(identity.auto_seed(seat)),
            )
            for seat in seats()
            if partnership_of(seat) != self.model_partnership
        )
        assert len(self._auto_random_sources) == 2
        self._state = create(
            GameConfig(), GameSeed(identity.game_seed())
        )
        self._seq = 0
        self._auto_action_count = 0

    async def play_round(
        self,
        *,
        policy_version: int,
        rollout_id: str,
        round_id: int,
        max_seconds: float,
    ) -> _result.Ok[RolloutRoundResult] | _result.Rejected:
        """Play exactly one completed round from the current game."""
        assert policy_version >= 0
        assert rollout_id
        assert round_id >= 0
        assert max_seconds > 0.0
        assert observe(self._state, Seat.A).winning_partnership is None
        self._auto_action_count = 0
        for actor in self._model_actors:
            actor.begin_round(
                base_seed=self.identity.base_seed,
                policy_version=policy_version,
                rollout_id=rollout_id,
                round_id=round_id,
            )
        observed = self._broadcast()
        if isinstance(observed, _result.Rejected):
            return observed
        confirmed = self._confirm_round()
        if isinstance(confirmed, _result.Rejected):
            return confirmed
        before = observe(self._state, Seat.A)
        started = time.monotonic()
        completed = await self._play_until_scoring(
            started=started,
            max_seconds=max_seconds,
        )
        if isinstance(completed, _result.Rejected):
            return completed
        final_snapshot = completed.value
        first_reward, second_reward = round_return_targets(
            before=before,
            after=final_snapshot,
        )
        model_reward = (
            first_reward
            if self.model_partnership == Partnership.FIRST
            else second_reward
        )
        first_actor, second_actor = self._model_actors
        trajectory = CompletedRoundTrajectory(
            policy_version=policy_version,
            round_id=round_id,
            model_partnership=self.model_partnership,
            seats=(
                first_actor.trajectory().finish(),
                second_actor.trajectory().finish(),
            ),
            terminal_reward=model_reward,
        )
        stats = tuple(actor.stats() for actor in self._model_actors)
        return _result.Ok(
            RolloutRoundResult(
                trajectory=trajectory,
                model_reward=model_reward,
                auto_reward=-model_reward,
                model_action_count=sum(
                    item.model_action_count for item in stats
                ),
                auto_action_count=self._auto_action_count,
                forced_action_count=sum(
                    item.forced_action_count for item in stats
                ),
                trainable_decision_count=sum(
                    item.trainable_decision_count for item in stats
                ),
                legal_choice_count=sum(
                    item.legal_choice_count for item in stats
                ),
                scored_choice_step_count=sum(
                    item.scored_choice_step_count for item in stats
                ),
                model_declarer=(
                    final_snapshot.declarer is not None
                    and partnership_of(final_snapshot.declarer)
                    == self.model_partnership
                ),
                elapsed_seconds=max(
                    time.monotonic() - started, 0.000001
                ),
                game_over=final_snapshot.winning_partnership
                is not None,
            )
        )

    async def close(self) -> None:
        """Close a session that owns no background work."""

    def next_game(self, *, policy: RolloutPolicy) -> RolloutSession:
        """Create the next balanced game for this environment."""
        assert (
            observe(self._state, Seat.A).winning_partnership is not None
        )
        return RolloutSession(
            policy=policy,
            identity=self.identity.next_game(),
        )

    def discard_game(self, *, policy: RolloutPolicy) -> RolloutSession:
        """Discard an interrupted game before a policy change."""
        return RolloutSession(
            policy=policy,
            identity=self.identity.next_game(),
        )

    def _confirm_round(self) -> _result.Ok[None] | _result.Rejected:
        for seat in seats():
            snapshot = observe(self._state, seat)
            assert snapshot.awaiting_action == "next_round"
            accepted = self._apply(
                actor=seat,
                command=commands.ConfirmRound(),
            )
            if isinstance(accepted, _result.Rejected):
                return accepted
        return _result.Ok(None)

    async def _play_until_scoring(
        self, *, started: float, max_seconds: float
    ) -> _result.Ok[PlayerSnapshot] | _result.Rejected:
        while True:
            snapshot = observe(self._state, Seat.A)
            if (
                snapshot.phase == "WAITING"
                and snapshot.scoring is not None
            ):
                return _result.Ok(snapshot)
            remaining = max_seconds - (time.monotonic() - started)
            if remaining <= 0.0:
                return _timed_out(max_seconds)
            seat = self._awaited_seat()
            if partnership_of(seat) == self.model_partnership:
                decision = await _model_decide_before(
                    actor=self._model_actor(seat),
                    seq=self._seq,
                    snapshot=observe(self._state, seat),
                    seconds=remaining,
                )
                if isinstance(decision, _result.Rejected):
                    return decision
                accepted = self._apply(
                    actor=seat,
                    command=decision.value.command,
                )
                if isinstance(accepted, _result.Rejected):
                    raise AssertionError(
                        "rollout policy produced rejected action: "
                        + accepted.reason
                    )
                self._model_actor(seat).accept(decision.value)
                continue
            auto_command = choose_auto_command(
                actor=seat,
                snapshot=observe(self._state, seat),
                random_source=self._auto_random_source(seat),
            )
            accepted = self._apply(actor=seat, command=auto_command)
            if isinstance(accepted, _result.Rejected):
                raise AssertionError(
                    "Auto policy produced rejected legal action: "
                    + accepted.reason
                )
            self._auto_action_count += 1

    def _awaited_seat(self) -> Seat:
        awaited = tuple(
            seat
            for seat in seats()
            if observe(self._state, seat).awaiting_action
            in _STRATEGIC_ACTIONS
        )
        assert len(awaited) == 1
        return awaited[0]

    def _apply(
        self, *, actor: Seat, command: commands.Command
    ) -> _result.Ok[None] | _result.Rejected:
        result = apply(self._state, actor, command)
        if isinstance(result, CommandRejected):
            return _result.Rejected(result.reason)
        self._state = result.value
        self._seq += 1
        return self._broadcast()

    def _broadcast(self) -> _result.Ok[None] | _result.Rejected:
        for actor in self._model_actors:
            observed = actor.observe(
                seq=self._seq,
                snapshot=observe(self._state, actor.seat),
            )
            if isinstance(observed, _result.Rejected):
                return observed
        return _result.Ok(None)

    def _model_actor(self, seat: Seat) -> ModelActor:
        for actor in self._model_actors:
            if actor.seat == seat:
                return actor
        raise AssertionError("seat is not controlled by rollout model")

    def _auto_random_source(self, seat: Seat) -> random.Random:
        for policy_seat, random_source in self._auto_random_sources:
            if policy_seat == seat:
                return random_source
        raise AssertionError("seat is not controlled by Auto")


async def _model_decide_before(
    *,
    actor: ModelActor,
    seq: int,
    snapshot: PlayerSnapshot,
    seconds: float,
) -> _result.Ok[ActorDecision] | _result.Rejected:
    task = asyncio.create_task(actor.decide(seq=seq, snapshot=snapshot))
    return await _task_before(task=task, seconds=seconds)


async def _task_before[T](
    *,
    task: asyncio.Task[_result.Ok[T] | _result.Rejected],
    seconds: float,
) -> _result.Ok[T] | _result.Rejected:
    done, _ = await asyncio.wait((task,), timeout=seconds)
    if done:
        return task.result()
    _ = task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return _timed_out(seconds)


def _timed_out(seconds: float) -> _result.Rejected:
    return _result.Rejected(
        f"training round timed out after {seconds:g} seconds"
    )


__all__ = ("RolloutRoundResult", "RolloutSession")
