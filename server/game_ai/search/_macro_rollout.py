"""Paired blueprint rollouts to the root viewer's next decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from server.foundation.result import Ok, Rejected
from server.game import Partnership, Seat, partnership_of, seats
from server.game.snapshots import PlayerSnapshot
from server.policy_model.actions import physical_command
from server.policy_model.inference import (
    ActionSamples,
    PolicyQuery,
    PolicySampleRequest,
    ProposedAction,
    SamplingSeed,
)
from server.policy_model.observation import Observation
from server.policy_model.value_target import round_value_targets

from ..belief import Particle
from ..simulation import SimulationBranch


class RolloutInference(Protocol):
    """Model operations required by macro rollout execution."""

    async def sample(
        self,
        *,
        requests: tuple[PolicySampleRequest, ...],
    ) -> Ok[tuple[ActionSamples, ...]] | Rejected:
        """Sample each simulated actor's blueprint action."""
        ...

    async def values(
        self,
        *,
        observations: tuple[Observation, ...],
    ) -> Ok[tuple[float, ...]] | Rejected:
        """Estimate root-viewer continuation values."""
        ...


@dataclass(frozen=True, slots=True)
class MacroRolloutStatistics:
    """Exact work performed by one paired macro rollout batch."""

    sample_count: int
    engine_advance_count: int
    policy_row_count: int
    value_row_count: int


@dataclass(frozen=True, slots=True)
class MacroRolloutResult:
    """One value sample per candidate and replica."""

    values: tuple[tuple[float, ...], ...]
    statistics: MacroRolloutStatistics


@dataclass(frozen=True, slots=True)
class _Lane:
    candidate_position: int
    replica_index: int
    branch: SimulationBranch
    actor_step: int


@dataclass(frozen=True, slots=True)
class _Leaf:
    candidate_position: int
    replica_index: int
    observation: Observation | None
    terminal_value: float | None

    def __post_init__(self) -> None:
        assert (self.observation is None) != (
            self.terminal_value is None
        )


async def run_paired_macro_rollouts(
    *,
    model: RolloutInference,
    particles: tuple[Particle, ...],
    candidates: tuple[ProposedAction, ...],
    viewer: Seat,
    root_snapshot: PlayerSnapshot,
    decision_seed: int,
    halving_round: int,
    sample_count: int,
) -> Ok[MacroRolloutResult] | Rejected:
    """Evaluate candidates on identical weighted worlds and RNG keys."""
    assert particles
    assert candidates
    assert sample_count > 0
    particle_indices = _stratified_particle_indices(
        particles=particles,
        sample_count=sample_count,
        seed=decision_seed,
        halving_round=halving_round,
    )
    active: list[_Lane] = []
    engine_advances = 0
    for candidate_position, candidate in enumerate(candidates):
        for replica_index, particle_index in enumerate(
            particle_indices
        ):
            branch = particles[particle_index].branch
            command = physical_command(
                action=candidate.action,
                hand=branch.snapshot(viewer).hand,
            )
            if isinstance(command, Rejected):
                return command
            advanced = branch.advance(
                actor=viewer,
                command=command.value,
            )
            engine_advances += 1
            if isinstance(advanced, Rejected):
                return Rejected(
                    reason=(
                        "AI root action disagrees with game engine: "
                        f"{advanced.reason}"
                    )
                )
            active.append(
                _Lane(
                    candidate_position=candidate_position,
                    replica_index=replica_index,
                    branch=advanced.value,
                    actor_step=0,
                )
            )
    leaves: list[_Leaf] = []
    policy_rows = 0
    while active:
        pending_lanes: list[_Lane] = []
        requests: list[PolicySampleRequest] = []
        for lane in active:
            terminal = _terminal_value(
                branch=lane.branch,
                viewer=viewer,
                root_snapshot=root_snapshot,
            )
            if terminal is not None:
                leaves.append(
                    _Leaf(
                        candidate_position=lane.candidate_position,
                        replica_index=lane.replica_index,
                        observation=None,
                        terminal_value=terminal,
                    )
                )
                continue
            actor = _awaited_actor(lane.branch)
            if isinstance(actor, Rejected):
                return actor
            if actor.value == viewer:
                observation, _legal_actions = lane.branch.model_input(
                    viewer
                )
                leaves.append(
                    _Leaf(
                        candidate_position=lane.candidate_position,
                        replica_index=lane.replica_index,
                        observation=observation,
                        terminal_value=None,
                    )
                )
                continue
            observation, legal_actions = lane.branch.model_input(
                actor.value
            )
            pending_lanes.append(lane)
            requests.append(
                PolicySampleRequest(
                    query=PolicyQuery(
                        observation=observation,
                        legal_actions=legal_actions,
                        seat=actor.value,
                    ),
                    draws=(
                        SamplingSeed(
                            seed=_rollout_seed(
                                decision_seed=decision_seed,
                                halving_round=halving_round,
                                replica_index=lane.replica_index,
                                actor_step=lane.actor_step,
                            ),
                            ordinal=0,
                        ),
                    ),
                )
            )
        if not pending_lanes:
            active = []
            continue
        sampled = await model.sample(requests=tuple(requests))
        if isinstance(sampled, Rejected):
            return sampled
        assert len(sampled.value) == len(pending_lanes)
        policy_rows += len(pending_lanes)
        next_active: list[_Lane] = []
        for lane, request, samples in zip(
            pending_lanes,
            requests,
            sampled.value,
            strict=True,
        ):
            assert len(samples.samples) == 1
            command = physical_command(
                action=samples.samples[0].action,
                hand=lane.branch.snapshot(request.query.seat).hand,
            )
            if isinstance(command, Rejected):
                return command
            advanced = lane.branch.advance(
                actor=request.query.seat,
                command=command.value,
            )
            engine_advances += 1
            if isinstance(advanced, Rejected):
                return Rejected(
                    reason=(
                        "AI rollout action disagrees with game engine: "
                        f"{advanced.reason}"
                    )
                )
            next_active.append(
                _Lane(
                    candidate_position=lane.candidate_position,
                    replica_index=lane.replica_index,
                    branch=advanced.value,
                    actor_step=lane.actor_step + 1,
                )
            )
        active = next_active
    leaf_observations = tuple(
        leaf.observation
        for leaf in leaves
        if leaf.observation is not None
    )
    if leaf_observations:
        estimated = await model.values(observations=leaf_observations)
        if isinstance(estimated, Rejected):
            return estimated
        estimated_values = iter(estimated.value)
    else:
        estimated_values = iter(())
    values = [
        [0.0 for _index in range(sample_count)]
        for _candidate in candidates
    ]
    seen = [
        [False for _index in range(sample_count)]
        for _candidate in candidates
    ]
    for leaf in leaves:
        value = (
            leaf.terminal_value
            if leaf.terminal_value is not None
            else next(estimated_values)
        )
        assert value is not None
        assert not seen[leaf.candidate_position][leaf.replica_index]
        values[leaf.candidate_position][leaf.replica_index] = value
        seen[leaf.candidate_position][leaf.replica_index] = True
    assert all(all(row) for row in seen)
    return Ok(
        MacroRolloutResult(
            values=tuple(tuple(row) for row in values),
            statistics=MacroRolloutStatistics(
                sample_count=len(candidates) * sample_count,
                engine_advance_count=engine_advances,
                policy_row_count=policy_rows,
                value_row_count=len(leaf_observations),
            ),
        )
    )


def _awaited_actor(
    branch: SimulationBranch,
) -> Ok[Seat] | Rejected:
    actors = [
        seat
        for seat in seats()
        if branch.snapshot(seat).awaiting_action
        in ("bid", "discard", "stir", "play")
    ]
    if len(actors) != 1:
        return Rejected(
            reason="AI simulation does not have exactly one actor"
        )
    return Ok(actors[0])


def _terminal_value(
    *,
    branch: SimulationBranch,
    viewer: Seat,
    root_snapshot: PlayerSnapshot,
) -> float | None:
    after = branch.snapshot(viewer)
    if after.scoring is None:
        return None
    first, second = round_value_targets(
        before=root_snapshot,
        after=after,
    )
    return (
        first if partnership_of(viewer) == Partnership.FIRST else second
    )


def _stratified_particle_indices(
    *,
    particles: tuple[Particle, ...],
    sample_count: int,
    seed: int,
    halving_round: int,
) -> tuple[int, ...]:
    total = sum(particle.weight for particle in particles)
    assert total > 0.0
    offset = _unit_float(
        seed=seed,
        first=halving_round,
        second=sample_count,
    )
    thresholds = tuple(
        (float(index) + offset) / float(sample_count)
        for index in range(sample_count)
    )
    result: list[int] = []
    cumulative = 0.0
    particle_index = 0
    for threshold in thresholds:
        while particle_index < len(particles) - 1 and (
            cumulative + particles[particle_index].weight / total
            <= threshold
        ):
            cumulative += particles[particle_index].weight / total
            particle_index += 1
        result.append(particle_index)
    return tuple(result)


def _rollout_seed(
    *,
    decision_seed: int,
    halving_round: int,
    replica_index: int,
    actor_step: int,
) -> int:
    return int(
        _unit_float(
            seed=decision_seed,
            first=halving_round * 1_000_003 + replica_index,
            second=actor_step,
        )
        * float(1 << 63)
    )


def _unit_float(*, seed: int, first: int, second: int) -> float:
    value = (
        seed * 6_364_136_223_846_793_005
        + first * 1_442_695_040_888_963_407
        + second * 2_862_933_555_777_941_757
    ) & ((1 << 64) - 1)
    value ^= value >> 30
    value = (value * 13_785_493_119_561_298_705) & ((1 << 64) - 1)
    value ^= value >> 27
    return float(value >> 11) / float(1 << 53)


__all__ = (
    "MacroRolloutResult",
    "MacroRolloutStatistics",
    "RolloutInference",
    "run_paired_macro_rollouts",
)
