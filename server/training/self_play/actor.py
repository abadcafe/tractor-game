"""Policy decision engine used by direct self-play."""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from server.foundation.result import Ok, Rejected
from server.game import Seat, commands
from server.game.snapshots import PlayerSnapshot
from server.policy_model.actions import (
    bind_generated_action,
    build_legal_action_space,
)
from server.policy_model.observation import (
    ObservationMemory,
    build_observation,
)
from server.training.rollout_inference.randomness import (
    TrainingDecisionKey,
)
from server.training.self_play.policy import TrainingPolicy
from server.training.self_play.trajectory import (
    DecisionStep,
    TrajectoryRecorder,
)


@dataclass(frozen=True, slots=True)
class SelfPlayActorStats:
    """Per-round model action statistics."""

    generated_action_count: int = 0
    accepted_action_count: int = 0
    action_choice_count: int = 0


@dataclass(frozen=True, slots=True)
class TrainingDecision:
    """A typed game command and its trajectory record."""

    command: commands.Command
    step: DecisionStep


@final
class SelfPlayActor:
    """Create policy decisions from contiguous player snapshots."""

    def __init__(
        self,
        seat: Seat,
        *,
        policy: TrainingPolicy,
        recorder: TrajectoryRecorder | None = None,
        observation_memory: ObservationMemory | None = None,
    ) -> None:
        self.seat = seat
        self._policy = policy
        self._recorder = recorder or TrajectoryRecorder()
        self._observation_memory = (
            observation_memory or ObservationMemory()
        )
        self._stats = SelfPlayActorStats()
        self._base_seed = 0
        self._policy_version = 0
        self._rollout_id = "uninitialized"
        self._episode_id = 0
        self._decision_index = 0

    def recorder(self) -> TrajectoryRecorder:
        """Return the accepted trajectory recorder."""
        return self._recorder

    def stats(self) -> SelfPlayActorStats:
        """Return per-round action statistics."""
        return self._stats

    def reset_round_tracking(
        self,
        *,
        base_seed: int,
        policy_version: int,
        rollout_id: str,
        episode_id: int,
    ) -> None:
        """Clear history and counters before one training round."""
        assert base_seed >= 0
        assert policy_version >= 0
        assert rollout_id
        assert episode_id >= 0
        self._observation_memory.reset_episode()
        self._stats = SelfPlayActorStats()
        self._base_seed = base_seed
        self._policy_version = policy_version
        self._rollout_id = rollout_id
        self._episode_id = episode_id
        self._decision_index = 0

    def observe(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
    ) -> Ok[None] | Rejected:
        """Record one accepted state transition."""
        result = self._observation_memory.observe(
            seq=seq,
            snapshot=snapshot,
        )
        if isinstance(result, Rejected):
            return result
        return Ok(value=None)

    async def decide(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
    ) -> Ok[TrainingDecision] | Rejected:
        """Generate one rule-indexed model decision."""
        assert snapshot.awaiting_action in (
            "bid",
            "stir",
            "discard",
            "play",
        )
        observation = build_observation(
            viewer=self.seat,
            snapshot=snapshot,
            memory=self._observation_memory.view(),
        )
        legal_actions = build_legal_action_space(
            viewer=self.seat,
            snapshot=snapshot,
            query=observation.action_query,
        )
        decision_result = await self._policy.decide(
            observation,
            legal_actions,
            TrainingDecisionKey(
                base_seed=self._base_seed,
                policy_version=self._policy_version,
                rollout_id=self._rollout_id,
                episode_id=self._episode_id,
                seat=self.seat,
                decision_index=self._decision_index,
            ),
        )
        if isinstance(decision_result, Rejected):
            return decision_result
        decision = decision_result.value
        command_result = bind_generated_action(
            decision.action,
            snapshot.hand,
        )
        if isinstance(command_result, Rejected):
            return command_result
        self._decision_index += 1
        self._stats = _add_generated(
            self._stats,
            choice_count=decision.choice_count,
        )
        return Ok(
            value=TrainingDecision(
                command=command_result.value,
                step=DecisionStep(
                    seat=self.seat,
                    seq=seq,
                    action=decision.action,
                    decision_handle=decision.decision_handle,
                    choice_count=decision.choice_count,
                ),
            )
        )

    def accept(self, decision: TrainingDecision) -> None:
        """Commit a decision after the pure game accepts it."""
        self._recorder.append(decision.step)
        self._stats = _add_accepted(self._stats)


def _add_generated(
    stats: SelfPlayActorStats,
    *,
    choice_count: int,
) -> SelfPlayActorStats:
    return SelfPlayActorStats(
        generated_action_count=stats.generated_action_count + 1,
        accepted_action_count=stats.accepted_action_count,
        action_choice_count=stats.action_choice_count + choice_count,
    )


def _add_accepted(
    stats: SelfPlayActorStats,
) -> SelfPlayActorStats:
    return SelfPlayActorStats(
        generated_action_count=stats.generated_action_count,
        accepted_action_count=stats.accepted_action_count + 1,
        action_choice_count=stats.action_choice_count,
    )


__all__ = (
    "SelfPlayActor",
    "SelfPlayActorStats",
    "TrainingDecision",
)
