"""Seat-local model actor for rollout collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch

from server.foundation.result import Ok, Rejected
from server.game import Seat, commands
from server.game.snapshots import PlayerSnapshot
from server.policy_model.actions import (
    GeneratedAction,
    LegalActionSpace,
    bind_generated_action,
    build_legal_action_space,
)
from server.policy_model.actions.decoding import (
    ActionSampler,
    action_plan_generation_step_count,
    action_trace_from_choice_ids,
    compile_legal_action_frame,
    plan_batch_to_device,
)
from server.policy_model.observation import (
    ObservationMemory,
    build_observation,
)
from server.training.rollout.policy import RolloutPolicy
from server.training.rollout.trajectory import (
    SeatTrajectoryRecorder,
    TrainableDecision,
)
from server.training.rollout_inference.randomness import (
    TrainingDecisionKey,
)


@dataclass(frozen=True, slots=True)
class ForcedDecision:
    """A mathematically unique action resolved without the policy."""

    command: commands.Command
    action: GeneratedAction


@dataclass(frozen=True, slots=True)
class SampledDecision:
    """A genuine policy choice with replay metadata."""

    command: commands.Command
    step: TrainableDecision


type ActorDecision = ForcedDecision | SampledDecision


@dataclass(frozen=True, slots=True)
class ModelActorStats:
    """Per-round counts for one model-controlled seat."""

    model_action_count: int = 0
    forced_action_count: int = 0
    trainable_decision_count: int = 0
    legal_choice_count: int = 0
    scored_choice_step_count: int = 0


@final
class ModelActor:
    """Build lossless observations and request genuine choices."""

    def __init__(self, seat: Seat, *, policy: RolloutPolicy) -> None:
        self.seat = seat
        self._policy = policy
        self._recorder = SeatTrajectoryRecorder(seat=seat)
        self._memory = ObservationMemory()
        self._forced_sampler = ActionSampler.create(
            batch_capacity=1,
            device=torch.device("cpu"),
        )
        self._stats = ModelActorStats()
        self._base_seed = 0
        self._policy_version = 0
        self._rollout_id = "uninitialized"
        self._round_id = 0
        self._decision_index = 0

    def begin_round(
        self,
        *,
        base_seed: int,
        policy_version: int,
        rollout_id: str,
        round_id: int,
    ) -> None:
        """Reset only round-scoped observation and trajectory state."""
        assert base_seed >= 0
        assert policy_version >= 0
        assert rollout_id
        assert round_id >= 0
        self._memory.reset_round()
        self._recorder.clear()
        self._stats = ModelActorStats()
        self._base_seed = base_seed
        self._policy_version = policy_version
        self._rollout_id = rollout_id
        self._round_id = round_id
        self._decision_index = 0

    def observe(
        self, *, seq: int, snapshot: PlayerSnapshot
    ) -> Ok[None] | Rejected:
        """Consume one contiguous accepted player-visible state."""
        observed = self._memory.observe(seq=seq, snapshot=snapshot)
        if isinstance(observed, Rejected):
            return observed
        return Ok(None)

    async def decide(
        self, *, seq: int, snapshot: PlayerSnapshot
    ) -> Ok[ActorDecision] | Rejected:
        """Resolve forced actions locally or sample a policy choice."""
        observation = build_observation(
            viewer=self.seat,
            snapshot=snapshot,
            memory=self._memory.view(),
        )
        legal_actions = build_legal_action_space(
            viewer=self.seat,
            snapshot=snapshot,
            query=observation.action_query,
        )
        forced_result = _resolve_forced_action(
            legal_actions=legal_actions,
            sampler=self._forced_sampler,
        )
        if isinstance(forced_result, Rejected):
            return forced_result
        if forced_result.value is not None:
            command_result = bind_generated_action(
                forced_result.value,
                snapshot.hand,
            )
            if isinstance(command_result, Rejected):
                return command_result
            self._stats = ModelActorStats(
                model_action_count=self._stats.model_action_count + 1,
                forced_action_count=self._stats.forced_action_count + 1,
                trainable_decision_count=(
                    self._stats.trainable_decision_count
                ),
                legal_choice_count=self._stats.legal_choice_count + 1,
                scored_choice_step_count=(
                    self._stats.scored_choice_step_count
                ),
            )
            return Ok(
                ForcedDecision(
                    command=command_result.value,
                    action=forced_result.value,
                )
            )
        policy_result = await self._policy.decide(
            observation,
            legal_actions,
            TrainingDecisionKey(
                base_seed=self._base_seed,
                policy_version=self._policy_version,
                rollout_id=self._rollout_id,
                round_id=self._round_id,
                seat=self.seat,
                decision_index=self._decision_index,
            ),
        )
        if isinstance(policy_result, Rejected):
            return policy_result
        policy_decision = policy_result.value
        command_result = bind_generated_action(
            policy_decision.action,
            snapshot.hand,
        )
        if isinstance(command_result, Rejected):
            return command_result
        self._decision_index += 1
        step = TrainableDecision(
            seat=self.seat,
            seq=seq,
            action=policy_decision.action,
            replay=policy_decision.decision_handle,
            trace_step_count=len(policy_decision.action.trace.choices),
            legal_choice_count=policy_decision.legal_choice_count,
            scored_choice_step_count=(
                policy_decision.scored_choice_step_count
            ),
        )
        self._stats = ModelActorStats(
            model_action_count=self._stats.model_action_count + 1,
            forced_action_count=self._stats.forced_action_count,
            trainable_decision_count=(
                self._stats.trainable_decision_count + 1
            ),
            legal_choice_count=(
                self._stats.legal_choice_count
                + policy_decision.legal_choice_count
            ),
            scored_choice_step_count=(
                self._stats.scored_choice_step_count
                + policy_decision.scored_choice_step_count
            ),
        )
        return Ok(
            SampledDecision(command=command_result.value, step=step)
        )

    def accept(self, decision: ActorDecision) -> None:
        """Commit replay only after the game accepts the command."""
        if isinstance(decision, SampledDecision):
            self._recorder.append(decision.step)

    def trajectory(self) -> SeatTrajectoryRecorder:
        """Return the seat-owned trajectory recorder."""
        return self._recorder

    def stats(self) -> ModelActorStats:
        """Return round-local action counts."""
        return self._stats


def _resolve_forced_action(
    *,
    legal_actions: LegalActionSpace,
    sampler: ActionSampler,
) -> Ok[GeneratedAction | None] | Rejected:
    frame = compile_legal_action_frame(legal_actions)
    step_count = action_plan_generation_step_count(frame)
    resolved = sampler.resolve_forced(
        action_batch=plan_batch_to_device(
            (frame,), device=torch.device("cpu")
        ),
        generation_step_counts=torch.tensor(
            (step_count,), dtype=torch.long
        ),
        padded_generation_steps=step_count,
    )
    if isinstance(resolved, Rejected):
        return resolved
    batch = resolved.value
    if not bool(batch.forced_mask[0].item()):
        return Ok(None)
    actual_steps = int(batch.step_counts[0].item())
    choice_ids = tuple(
        int(batch.choice_ids_padded[0, index].item())
        for index in range(actual_steps)
    )
    trace_result = action_trace_from_choice_ids(choice_ids)
    if isinstance(trace_result, Rejected):
        return trace_result
    decoded = legal_actions.decode(trace_result.value)
    if isinstance(decoded, Rejected):
        return decoded
    return Ok(decoded.value)


__all__ = (
    "ActorDecision",
    "ForcedDecision",
    "ModelActor",
    "ModelActorStats",
    "SampledDecision",
)
