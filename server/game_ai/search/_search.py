"""Particle-weighted model rollout search over shared engine cohorts."""

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

from ..actions import physical_command
from ..belief import Particle
from ..branch import EngineBranch
from ..model import (
    ActionDrawKey,
    ActionSampleRequest,
    ModelEvaluator,
    ModelQuery,
    SampledAction,
)
from ._cohorts import (
    RolloutCohort,
    RolloutLane,
    group_sampled_lanes,
)


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
class SearchStatistics:
    """Execution counts for one completed particle search."""

    candidate_count: int
    rollout_lane_count: int
    maximum_cohort_count: int
    engine_advance_count: int
    leaf_value_count: int


@dataclass(frozen=True, slots=True)
class SearchDecision:
    """Selected command and observable execution statistics."""

    command: commands.Command
    statistics: SearchStatistics


@dataclass(frozen=True, slots=True)
class _Candidate:
    action: GeneratedAction
    log_probability: float


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

    async def decide(
        self,
        *,
        particles: tuple[Particle, ...],
        viewer: Seat,
        decision_seed: int,
    ) -> Ok[SearchDecision] | Rejected:
        """Search legal candidates instead of using policy argmax."""
        assert particles
        root = particles[0].branch
        root_snapshot = root.snapshot(viewer)
        if root_snapshot.awaiting_action != "play":
            return Rejected(
                reason="particle search requires a play decision"
            )
        observation, legal_actions = root.model_input(viewer)
        root_samples = await self._model.sample(
            requests=(
                ActionSampleRequest(
                    query=ModelQuery(
                        observation=observation,
                        legal_actions=legal_actions,
                        seat=viewer,
                    ),
                    draws=tuple(
                        ActionDrawKey(
                            seed=decision_seed,
                            ordinal=index,
                        )
                        for index in range(
                            self._config.candidate_samples
                        )
                    ),
                ),
            ),
        )
        if isinstance(root_samples, Rejected):
            return root_samples
        assert len(root_samples.value) == 1
        candidates = _unique_candidates(root_samples.value[0].samples)
        rooted = self._root_cohorts(
            candidates=candidates,
            particles=particles,
            viewer=viewer,
        )
        if isinstance(rooted, Rejected):
            return rooted
        cohorts, engine_advance_count = rooted.value
        rollout_lane_count = sum(
            len(cohort.lanes) for cohort in cohorts
        )
        evaluated = await self._rollout(
            cohorts=cohorts,
            candidates=candidates,
            viewer=viewer,
            root_snapshot=root_snapshot,
            decision_seed=decision_seed,
            initial_engine_advance_count=engine_advance_count,
        )
        if isinstance(evaluated, Rejected):
            return evaluated
        values, maximum_cohort_count, advances, leaf_count = (
            evaluated.value
        )
        candidate_index = max(
            range(len(candidates)),
            key=lambda index: (
                values[index],
                candidates[index].log_probability,
            ),
        )
        selected = physical_command(
            action=candidates[candidate_index].action,
            hand=root_snapshot.hand,
        )
        if isinstance(selected, Rejected):
            return selected
        return Ok(
            SearchDecision(
                command=selected.value,
                statistics=SearchStatistics(
                    candidate_count=len(candidates),
                    rollout_lane_count=rollout_lane_count,
                    maximum_cohort_count=maximum_cohort_count,
                    engine_advance_count=advances,
                    leaf_value_count=leaf_count,
                ),
            )
        )

    def _root_cohorts(
        self,
        *,
        candidates: tuple[_Candidate, ...],
        particles: tuple[Particle, ...],
        viewer: Seat,
    ) -> Ok[tuple[tuple[RolloutCohort, ...], int]] | Rejected:
        cohorts: list[RolloutCohort] = []
        rollout_count = self._config.rollouts_per_particle
        advance_count = 0
        for candidate_index, candidate in enumerate(candidates):
            for particle_index, particle in enumerate(particles):
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
                advance_count += 1
                if isinstance(advanced, Rejected):
                    return Rejected(
                        reason=(
                            "AI root action disagrees with game "
                            f"engine: {advanced.reason}"
                        )
                    )
                cohorts.append(
                    RolloutCohort(
                        branch=advanced.value,
                        lanes=tuple(
                            RolloutLane(
                                candidate_index=candidate_index,
                                particle_weight=(
                                    particle.weight
                                    / float(rollout_count)
                                ),
                                draw_ordinal=(
                                    (
                                        candidate_index * len(particles)
                                        + particle_index
                                    )
                                    * rollout_count
                                    + replica_index
                                ),
                            )
                            for replica_index in range(rollout_count)
                        ),
                    )
                )
        if not cohorts:
            return Rejected(reason="AI search has no legal cohort")
        return Ok((tuple(cohorts), advance_count))

    async def _rollout(
        self,
        *,
        cohorts: tuple[RolloutCohort, ...],
        candidates: tuple[_Candidate, ...],
        viewer: Seat,
        root_snapshot: PlayerSnapshot,
        decision_seed: int,
        initial_engine_advance_count: int,
    ) -> Ok[tuple[tuple[float, ...], int, int, int]] | Rejected:
        active = cohorts
        totals = [0.0 for _ in candidates]
        maximum_cohort_count = len(active)
        engine_advance_count = initial_engine_advance_count
        for depth in range(self._config.rollout_depth):
            pending_cohorts: list[RolloutCohort] = []
            pending_actors: list[Seat] = []
            requests: list[ActionSampleRequest] = []
            for cohort in active:
                terminal = _terminal_value(
                    branch=cohort.branch,
                    viewer=viewer,
                    root_snapshot=root_snapshot,
                )
                if terminal is not None:
                    _accumulate_terminal(
                        totals=totals,
                        lanes=cohort.lanes,
                        value=terminal,
                    )
                    continue
                actor = _awaited_actor(cohort.branch)
                if actor is None:
                    return Rejected(
                        reason=(
                            "AI rollout reached a state without actor"
                        )
                    )
                observation, legal_actions = cohort.branch.model_input(
                    actor
                )
                requests.append(
                    ActionSampleRequest(
                        query=ModelQuery(
                            observation=observation,
                            legal_actions=legal_actions,
                            seat=actor,
                        ),
                        draws=tuple(
                            ActionDrawKey(
                                seed=decision_seed + depth + 1,
                                ordinal=lane.draw_ordinal,
                            )
                            for lane in cohort.lanes
                        ),
                    )
                )
                pending_cohorts.append(cohort)
                pending_actors.append(actor)
            if not requests:
                return Ok(
                    (
                        tuple(totals),
                        maximum_cohort_count,
                        engine_advance_count,
                        0,
                    )
                )
            sampled = await self._model.sample(requests=tuple(requests))
            if isinstance(sampled, Rejected):
                return sampled
            assert len(sampled.value) == len(pending_cohorts)
            next_active: list[RolloutCohort] = []
            for cohort, actor, samples in zip(
                pending_cohorts,
                pending_actors,
                sampled.value,
                strict=True,
            ):
                for group in group_sampled_lanes(
                    cohort=cohort,
                    sampled=samples.samples,
                ):
                    command = physical_command(
                        action=group.action,
                        hand=cohort.branch.snapshot(actor).hand,
                    )
                    if isinstance(command, Rejected):
                        return command
                    advanced = cohort.branch.advance(
                        actor=actor,
                        command=command.value,
                    )
                    engine_advance_count += 1
                    if isinstance(advanced, Rejected):
                        return Rejected(
                            reason=(
                                "AI rollout action disagrees with "
                                f"game engine: {advanced.reason}"
                            )
                        )
                    next_active.append(
                        RolloutCohort(
                            branch=advanced.value,
                            lanes=group.lanes,
                        )
                    )
            active = tuple(next_active)
            maximum_cohort_count = max(
                maximum_cohort_count, len(active)
            )
            if not active:
                return Ok(
                    (
                        tuple(totals),
                        maximum_cohort_count,
                        engine_advance_count,
                        0,
                    )
                )
        leaf_result = await self._leaf_values(
            cohorts=active,
            viewer=viewer,
        )
        if isinstance(leaf_result, Rejected):
            return leaf_result
        for cohort, value in zip(
            active,
            leaf_result.value,
            strict=True,
        ):
            _accumulate_terminal(
                totals=totals,
                lanes=cohort.lanes,
                value=value,
            )
        return Ok(
            (
                tuple(totals),
                maximum_cohort_count,
                engine_advance_count,
                len(active),
            )
        )

    async def _leaf_values(
        self,
        *,
        cohorts: tuple[RolloutCohort, ...],
        viewer: Seat,
    ) -> Ok[tuple[float, ...]] | Rejected:
        if not cohorts:
            return Ok(())
        actors: list[Seat] = []
        observations: list[Observation] = []
        for cohort in cohorts:
            actor = _awaited_actor(cohort.branch)
            if actor is None:
                return Rejected(
                    reason="AI rollout leaf has no current actor"
                )
            actors.append(actor)
            observation, _legal_actions = cohort.branch.model_input(
                actor
            )
            observations.append(observation)
        values = await self._model.values(
            observations=tuple(observations)
        )
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
    samples: tuple[SampledAction, ...],
) -> tuple[_Candidate, ...]:
    unique: dict[tuple[int, ...], _Candidate] = {}
    for sample in samples:
        key = action_trace_choice_ids(sample.action.trace)
        candidate = _Candidate(
            action=sample.action,
            log_probability=sample.log_probability,
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


def _accumulate_terminal(
    *,
    totals: list[float],
    lanes: tuple[RolloutLane, ...],
    value: float,
) -> None:
    for lane in lanes:
        totals[lane.candidate_index] += lane.particle_weight * value


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


__all__ = (
    "ParticleSearch",
    "SearchConfig",
    "SearchDecision",
    "SearchStatistics",
)
