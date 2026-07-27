"""Particle-weighted model rollout search for live decisions."""

from __future__ import annotations

from dataclasses import dataclass

from server.foundation.result import Ok, Rejected
from server.game import (
    Partnership,
    Seat,
    commands,
    partnership_of,
    seats,
)
from server.game.snapshots import PlayerSnapshot
from server.training.observation import Observation
from server.training.rewards import round_rewards
from server.training.semantic_actions import GeneratedAction
from server.training.semantic_actions.choices import (
    action_trace_choice_ids,
)

from .actions import physical_command
from .belief import Particle
from .model import ActionEvaluation, ModelEvaluator, ModelQuery
from .runtime import EngineBranch


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Finite compute budget for one complete rollout decision."""

    candidate_samples: int = 24
    rollouts_per_particle: int = 2
    rollout_depth: int = 24

    def __post_init__(self) -> None:
        assert self.candidate_samples > 0
        assert self.rollouts_per_particle > 0
        assert self.rollout_depth > 0


@dataclass(frozen=True, slots=True)
class _Candidate:
    action: GeneratedAction
    log_probability: float


@dataclass(frozen=True, slots=True)
class _Trajectory:
    candidate_index: int
    particle_weight: float
    branch: EngineBranch


class ParticleSearch:
    """Select actions by expected model rollout value across worlds."""

    def __init__(
        self,
        *,
        model: ModelEvaluator,
        config: SearchConfig,
    ) -> None:
        self._model = model
        self._config = config

    def decide(
        self,
        *,
        particles: tuple[Particle, ...],
        viewer: Seat,
        decision_seed: int,
    ) -> Ok[commands.Command] | Rejected:
        """Search legal candidates instead of using policy argmax."""
        assert particles
        root = particles[0].branch
        root_snapshot = root.snapshot(viewer)
        if root_snapshot.awaiting_action != "play":
            return Rejected(
                reason="particle search requires a play decision"
            )
        observation, legal_actions = root.model_input(viewer)
        query = ModelQuery(
            observation=observation,
            legal_actions=legal_actions,
            seat=viewer,
        )
        sampled = self._model.sample(
            queries=tuple(
                query for _ in range(self._config.candidate_samples)
            ),
            decision_seed=decision_seed,
        )
        if isinstance(sampled, Rejected):
            return sampled
        candidates = _unique_candidates(sampled.value)
        trajectories_result = self._root_trajectories(
            candidates=candidates,
            particles=particles,
            viewer=viewer,
        )
        if isinstance(trajectories_result, Rejected):
            return trajectories_result
        evaluated = self._rollout(
            trajectories=trajectories_result.value,
            candidates=candidates,
            viewer=viewer,
            root_snapshot=root_snapshot,
            decision_seed=decision_seed,
        )
        if isinstance(evaluated, Rejected):
            return evaluated
        candidate_index = max(
            range(len(candidates)),
            key=lambda index: (
                evaluated.value[index],
                candidates[index].log_probability,
            ),
        )
        return physical_command(
            action=candidates[candidate_index].action,
            hand=root_snapshot.hand,
        )

    def _root_trajectories(
        self,
        *,
        candidates: tuple[_Candidate, ...],
        particles: tuple[Particle, ...],
        viewer: Seat,
    ) -> Ok[tuple[_Trajectory, ...]] | Rejected:
        result: list[_Trajectory] = []
        rollout_count = self._config.rollouts_per_particle
        for candidate_index, candidate in enumerate(candidates):
            for particle in particles:
                command = physical_command(
                    action=candidate.action,
                    hand=particle.branch.snapshot(viewer).hand,
                )
                if isinstance(command, Rejected):
                    return command
                advanced = particle.branch.advance(
                    actor=viewer,
                    command=command.value,
                )
                if isinstance(advanced, Rejected):
                    return Rejected(
                        reason=(
                            "AI root action disagrees with game "
                            "engine: "
                            f"{advanced.reason}"
                        )
                    )
                for _ in range(rollout_count):
                    result.append(
                        _Trajectory(
                            candidate_index=candidate_index,
                            particle_weight=(
                                particle.weight / float(rollout_count)
                            ),
                            branch=advanced.value.fork(),
                        )
                    )
        if not result:
            return Rejected(reason="AI search has no legal trajectory")
        return Ok(tuple(result))

    def _rollout(
        self,
        *,
        trajectories: tuple[_Trajectory, ...],
        candidates: tuple[_Candidate, ...],
        viewer: Seat,
        root_snapshot: PlayerSnapshot,
        decision_seed: int,
    ) -> Ok[tuple[float, ...]] | Rejected:
        active = list(trajectories)
        totals = [0.0 for _ in candidates]
        for depth in range(self._config.rollout_depth):
            pending: list[tuple[_Trajectory, Seat, ModelQuery]] = []
            next_active: list[_Trajectory] = []
            for trajectory in active:
                terminal = _terminal_value(
                    branch=trajectory.branch,
                    viewer=viewer,
                    root_snapshot=root_snapshot,
                )
                if terminal is not None:
                    totals[trajectory.candidate_index] += (
                        trajectory.particle_weight * terminal
                    )
                    continue
                actor = _awaited_actor(trajectory.branch)
                if actor is None:
                    return Rejected(
                        reason=(
                            "AI rollout reached a state without actor"
                        )
                    )
                observation, legal_actions = (
                    trajectory.branch.model_input(actor)
                )
                pending.append(
                    (
                        trajectory,
                        actor,
                        ModelQuery(
                            observation=observation,
                            legal_actions=legal_actions,
                            seat=actor,
                        ),
                    )
                )
            if not pending:
                return Ok(tuple(totals))
            sampled = self._model.sample(
                queries=tuple(item[2] for item in pending),
                decision_seed=decision_seed + depth + 1,
            )
            if isinstance(sampled, Rejected):
                return sampled
            for (
                trajectory,
                actor,
                _query,
            ), evaluation in zip(
                pending,
                sampled.value,
                strict=True,
            ):
                command = physical_command(
                    action=evaluation.action,
                    hand=trajectory.branch.snapshot(actor).hand,
                )
                if isinstance(command, Rejected):
                    return command
                advanced = trajectory.branch.advance(
                    actor=actor,
                    command=command.value,
                )
                if isinstance(advanced, Rejected):
                    return Rejected(
                        reason=(
                            "AI rollout action disagrees with game "
                            f"engine: {advanced.reason}"
                        )
                    )
                next_active.append(
                    _Trajectory(
                        candidate_index=trajectory.candidate_index,
                        particle_weight=trajectory.particle_weight,
                        branch=advanced.value,
                    )
                )
            active = next_active
            if not active:
                return Ok(tuple(totals))
        leaf_result = self._leaf_values(
            trajectories=tuple(active),
            viewer=viewer,
        )
        if isinstance(leaf_result, Rejected):
            return leaf_result
        for trajectory, value in zip(
            active,
            leaf_result.value,
            strict=True,
        ):
            totals[trajectory.candidate_index] += (
                trajectory.particle_weight * value
            )
        return Ok(tuple(totals))

    def _leaf_values(
        self,
        *,
        trajectories: tuple[_Trajectory, ...],
        viewer: Seat,
    ) -> Ok[tuple[float, ...]] | Rejected:
        if not trajectories:
            return Ok(())
        actors: list[Seat] = []
        observations: list[Observation] = []
        for trajectory in trajectories:
            actor = _awaited_actor(trajectory.branch)
            if actor is None:
                return Rejected(
                    reason="AI rollout leaf has no current actor"
                )
            actors.append(actor)
            observation, _legal_actions = trajectory.branch.model_input(
                actor
            )
            observations.append(observation)
        values = self._model.values(observations=tuple(observations))
        if isinstance(values, Rejected):
            return values
        viewer_partnership = partnership_of(viewer)
        return Ok(
            tuple(
                value
                if partnership_of(actor) == viewer_partnership
                else -value
                for actor, value in zip(
                    actors,
                    values.value,
                    strict=True,
                )
            )
        )


def _unique_candidates(
    evaluations: tuple[ActionEvaluation, ...],
) -> tuple[_Candidate, ...]:
    unique: dict[tuple[int, ...], _Candidate] = {}
    for evaluation in evaluations:
        key = action_trace_choice_ids(evaluation.action.trace)
        candidate = _Candidate(
            action=evaluation.action,
            log_probability=evaluation.log_probability,
        )
        previous = unique.get(key)
        if (
            previous is None
            or candidate.log_probability > previous.log_probability
        ):
            unique[key] = candidate
    assert unique
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: item.log_probability,
            reverse=True,
        )
    )


def _awaited_actor(branch: EngineBranch) -> Seat | None:
    actors = [
        seat
        for seat in seats()
        if branch.snapshot(seat).awaiting_action
        in ("bid", "discard", "stir", "play")
    ]
    if not actors:
        return None
    assert len(actors) == 1
    return actors[0]


def _terminal_value(
    *,
    branch: EngineBranch,
    viewer: Seat,
    root_snapshot: PlayerSnapshot,
) -> float | None:
    after = branch.snapshot(viewer)
    if after.scoring is None:
        return None
    first, second = round_rewards(
        before=root_snapshot,
        after=after,
    )
    return (
        first if partnership_of(viewer) == Partnership.FIRST else second
    )


__all__ = ("ParticleSearch", "SearchConfig")
