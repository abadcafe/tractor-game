"""Particle belief over hidden deals, exchanges, and remaining cards."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

from server.foundation.result import Ok, Rejected
from server.game import Seat, commands, instantiate
from server.game.snapshots import PlayerSnapshot
from server.training.observation_memory import ObservationMemoryView

from ..actions import physical_command, semantic_action
from ..queries import (
    ActionSamples,
    InferenceDrawKey,
    ModelQuery,
    PolicySampleRequest,
    PolicyScoreRequest,
)
from ..simulation import SimulationBranch
from ._history import (
    ObservedFrame,
    awaited_actor,
    bid_command,
    play_command,
    stir_command,
)
from ._worlds import deal_constraints, sample_round_deal


@dataclass(frozen=True, slots=True)
class Particle:
    """One complete legal hidden world and its posterior weight."""

    branch: SimulationBranch
    weight: float

    def __post_init__(self) -> None:
        assert math.isfinite(self.weight)
        assert self.weight > 0.0


@dataclass(frozen=True, slots=True)
class _WeightedBranch:
    branch: SimulationBranch
    log_weight: float


@dataclass(frozen=True, slots=True)
class _ReplayBranch:
    branch: SimulationBranch
    log_weight: float
    pending_scores: tuple[PolicyScoreRequest, ...]


class BeliefInference(Protocol):
    """Model operations required only by hidden-world filtering."""

    async def sample(
        self,
        *,
        requests: tuple[PolicySampleRequest, ...],
    ) -> Ok[tuple[ActionSamples, ...]] | Rejected:
        """Sample hidden actions under their actor's policy."""
        ...

    async def score(
        self,
        *,
        requests: tuple[PolicyScoreRequest, ...],
    ) -> Ok[tuple[float, ...]] | Rejected:
        """Score observed actions under their actor's policy."""
        ...


class _NoParticleSurvived(Rejected):
    """All proposed worlds conflicted with one public transition."""

    def __init__(self) -> None:
        super().__init__("all AI hidden worlds were eliminated")


class ParticleBelief:
    """Reconstruct and incrementally update independent game worlds."""

    def __init__(
        self,
        *,
        viewer: Seat,
        model: BeliefInference,
        particle_count: int,
        random_source: random.Random,
    ) -> None:
        assert particle_count > 0
        self._viewer = viewer
        self._model = model
        self._particle_count = particle_count
        self._random = random_source
        self._frames: list[ObservedFrame] = []
        self._submitted: dict[int, commands.Command] = {}
        self._branches: tuple[_WeightedBranch, ...] = ()
        self._synced_frame_index = -1

    def record(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
        memory: ObservationMemoryView,
        error: str | None,
    ) -> Ok[None] | Rejected:
        """Record one contiguous real observation without inference."""
        if error is not None:
            return Ok(None)
        if self._frames and seq == self._frames[-1].seq:
            if snapshot != self._frames[-1].snapshot:
                return Rejected(
                    reason="AI received conflicting duplicate state"
                )
            return Ok(None)
        if self._frames and seq != self._frames[-1].seq + 1:
            return Rejected(reason="AI missed a game-state sequence")
        if snapshot.phase == "DEAL_BID" and (
            not self._frames
            or self._frames[-1].snapshot.round_number
            != snapshot.round_number
        ):
            self._frames.clear()
            self._submitted.clear()
            self._branches = ()
            self._synced_frame_index = -1
        if not self._frames and snapshot.phase != "DEAL_BID":
            return Ok(None)
        self._frames.append(
            ObservedFrame(
                seq=seq,
                snapshot=snapshot,
                memory=memory,
            )
        )
        return Ok(None)

    def submitted(
        self,
        *,
        seq: int,
        command: commands.Command,
    ) -> None:
        """Remember the viewer's private command for later replay."""
        assert seq not in self._submitted
        self._submitted[seq] = command

    async def synchronize(
        self,
    ) -> Ok[tuple[Particle, ...]] | Rejected:
        """Bring all particles to the latest recorded real state."""
        if not self._frames:
            return Rejected(reason="AI has no active round history")
        if not self._branches:
            built = await self._build_particles(
                target_count=self._particle_count
            )
            if isinstance(built, Rejected):
                return built
            self._branches = built.value
            self._synced_frame_index = len(self._frames) - 1
        elif self._synced_frame_index < len(self._frames) - 1:
            advanced = await self._advance_recorded_frames()
            if isinstance(advanced, _NoParticleSurvived):
                advanced = await self._build_particles(
                    target_count=self._particle_count
                )
            if isinstance(advanced, Rejected):
                return advanced
            self._branches = advanced.value
            self._synced_frame_index = len(self._frames) - 1
        normalized = _normalize(self._branches)
        if isinstance(normalized, Rejected):
            return normalized
        particles = normalized.value
        if _effective_sample_size(particles) < (
            float(self._particle_count) * 0.5
        ):
            replenished = await self._build_particles(
                target_count=max(1, self._particle_count // 2)
            )
            if isinstance(replenished, Rejected):
                return replenished
            fresh_count = len(replenished.value)
            retained_count = self._particle_count - fresh_count
            self._branches = (
                *_systematic_resample(
                    branches=self._branches,
                    count=retained_count,
                    random_source=self._random,
                ),
                *_systematic_resample(
                    branches=replenished.value,
                    count=fresh_count,
                    random_source=self._random,
                ),
            )
            refreshed = _normalize(self._branches)
            if isinstance(refreshed, Rejected):
                return refreshed
            return refreshed
        return Ok(particles)

    async def _build_particles(
        self,
        *,
        target_count: int,
    ) -> Ok[tuple[_WeightedBranch, ...]] | Rejected:
        assert target_count > 0
        constraints_result = deal_constraints(
            viewer=self._viewer,
            frames=tuple(self._frames),
        )
        if isinstance(constraints_result, Rejected):
            return constraints_result
        constraints = constraints_result.value
        built: list[_WeightedBranch] = []
        attempt_limit = target_count * 200
        attempts = 0
        while len(built) < target_count and attempts < attempt_limit:
            missing = target_count - len(built)
            initial: list[_WeightedBranch] = []
            for _index in range(missing):
                attempts += 1
                deal_result = sample_round_deal(
                    constraints=constraints,
                    random_source=self._random,
                )
                if isinstance(deal_result, Rejected):
                    return deal_result
                state_result = instantiate(deal_result.value)
                if isinstance(state_result, Rejected):
                    return state_result
                branch_result = SimulationBranch.start(
                    state_result.value
                )
                if isinstance(branch_result, Rejected):
                    return branch_result
                if (
                    branch_result.value.snapshot(self._viewer)
                    != self._frames[0].snapshot
                ):
                    continue
                initial.append(
                    _WeightedBranch(
                        branch=branch_result.value,
                        log_weight=0.0,
                    )
                )
                if attempts == attempt_limit:
                    break
            replayed = await self._replay_branches(tuple(initial))
            if isinstance(replayed, _NoParticleSurvived):
                continue
            if isinstance(replayed, Rejected):
                return replayed
            built.extend(replayed.value[:missing])
        if len(built) == target_count:
            return Ok(tuple(built))
        return Rejected(
            reason=(
                "AI could not construct enough legal hidden worlds "
                f"({len(built)}/{target_count})"
            )
        )

    async def _replay_branches(
        self,
        branches: tuple[_WeightedBranch, ...],
    ) -> Ok[tuple[_WeightedBranch, ...]] | Rejected:
        if not branches:
            return Ok(())
        return await self._advance_frames(
            branches=branches,
            frame_pairs=tuple(
                zip(
                    self._frames[:-1],
                    self._frames[1:],
                    strict=True,
                )
            ),
        )

    async def _advance_recorded_frames(
        self,
    ) -> Ok[tuple[_WeightedBranch, ...]] | Rejected:
        frame_pairs = tuple(
            (
                self._frames[frame_index - 1],
                self._frames[frame_index],
            )
            for frame_index in range(
                self._synced_frame_index + 1,
                len(self._frames),
            )
        )
        return await self._advance_frames(
            branches=self._branches,
            frame_pairs=frame_pairs,
        )

    async def _advance_frames(
        self,
        *,
        branches: tuple[_WeightedBranch, ...],
        frame_pairs: tuple[tuple[ObservedFrame, ObservedFrame], ...],
    ) -> Ok[tuple[_WeightedBranch, ...]] | Rejected:
        active = tuple(
            _ReplayBranch(
                branch=item.branch,
                log_weight=item.log_weight,
                pending_scores=(),
            )
            for item in branches
        )
        for previous, current in frame_pairs:
            advanced = await self._advance_replay_frame(
                branches=active,
                previous=previous,
                current=current,
            )
            if isinstance(advanced, Rejected):
                return advanced
            active = advanced.value
        scored = await self._score_pending(active)
        if isinstance(scored, Rejected):
            return scored
        if not scored.value:
            return _NoParticleSurvived()
        return Ok(
            tuple(
                _WeightedBranch(
                    branch=item.branch,
                    log_weight=item.log_weight,
                )
                for item in scored.value
            )
        )

    async def _advance_replay_frame(
        self,
        *,
        branches: tuple[_ReplayBranch, ...],
        previous: ObservedFrame,
        current: ObservedFrame,
    ) -> Ok[tuple[_ReplayBranch, ...]] | Rejected:
        unresolved: list[
            tuple[
                _ReplayBranch,
                Seat,
                commands.Command | None,
                bool,
            ]
        ] = []
        for particle in branches:
            command_result = self._transition_command(
                branch=particle.branch,
                previous=previous,
                current=current,
            )
            if isinstance(command_result, Rejected):
                continue
            actor, command, observed_action = command_result.value
            unresolved.append(
                (particle, actor, command, observed_action)
            )
        pending_result = await self._resolve_hidden_commands(
            unresolved=tuple(unresolved),
            seed=previous.seq,
        )
        if isinstance(pending_result, Rejected):
            return pending_result
        pending = pending_result.value
        if not pending:
            return _NoParticleSurvived()
        result: list[_ReplayBranch] = []
        for (
            particle,
            actor,
            command,
            observed_action,
        ) in pending:
            request: PolicyScoreRequest | None = None
            if observed_action:
                request_result = _score_request(
                    branch=particle.branch,
                    actor=actor,
                    command=command,
                )
                if isinstance(request_result, Rejected):
                    continue
                request = request_result.value
            advanced = particle.branch.advance(
                actor=actor,
                command=command,
            )
            if isinstance(advanced, Rejected):
                continue
            if (
                advanced.value.snapshot(self._viewer)
                != current.snapshot
            ):
                continue
            result.append(
                _ReplayBranch(
                    branch=advanced.value,
                    log_weight=particle.log_weight,
                    pending_scores=(
                        particle.pending_scores
                        if request is None
                        else (*particle.pending_scores, request)
                    ),
                )
            )
        if not result:
            return _NoParticleSurvived()
        return Ok(tuple(result))

    async def _score_pending(
        self,
        branches: tuple[_ReplayBranch, ...],
    ) -> Ok[tuple[_ReplayBranch, ...]] | Rejected:
        requests = tuple(
            request
            for particle in branches
            for request in particle.pending_scores
        )
        if not requests:
            return Ok(branches)
        scored = await self._model.score(requests=requests)
        if isinstance(scored, Rejected):
            return scored
        score_index = 0
        result: list[_ReplayBranch] = []
        for particle in branches:
            log_weight = particle.log_weight
            for _request in particle.pending_scores:
                log_weight += scored.value[score_index]
                score_index += 1
            result.append(
                _ReplayBranch(
                    branch=particle.branch,
                    log_weight=log_weight,
                    pending_scores=(),
                )
            )
        assert score_index == len(scored.value)
        return Ok(tuple(result))

    def _transition_command(
        self,
        *,
        branch: SimulationBranch,
        previous: ObservedFrame,
        current: ObservedFrame,
    ) -> Ok[tuple[Seat, commands.Command | None, bool]] | Rejected:
        actor_result = awaited_actor(branch)
        if isinstance(actor_result, Rejected):
            return actor_result
        actor = actor_result.value
        own = self._submitted.get(previous.seq)
        if actor == self._viewer and own is not None:
            return Ok((actor, own, False))
        action = branch.snapshot(actor).awaiting_action
        if action == "bid":
            command = bid_command(previous.snapshot, current.snapshot)
            return Ok((actor, command, True))
        if action == "stir":
            command = stir_command(
                previous.snapshot,
                current.snapshot,
            )
            return Ok((actor, command, True))
        if action == "play":
            command = play_command(previous, current)
            if isinstance(command, Rejected):
                return command
            return Ok((actor, command.value, True))
        if action == "discard":
            return Ok((actor, None, False))
        return Rejected(
            reason="particle transition has no model action"
        )

    async def _resolve_hidden_commands(
        self,
        *,
        unresolved: tuple[
            tuple[
                _ReplayBranch,
                Seat,
                commands.Command | None,
                bool,
            ],
            ...,
        ],
        seed: int,
    ) -> (
        Ok[
            tuple[
                tuple[
                    _ReplayBranch,
                    Seat,
                    commands.Command,
                    bool,
                ],
                ...,
            ]
        ]
        | Rejected
    ):
        hidden = tuple(
            (index, particle, actor)
            for index, (
                particle,
                actor,
                command,
                _observed,
            ) in enumerate(unresolved)
            if command is None
        )
        if not hidden:
            return Ok(
                tuple(
                    (particle, actor, command, observed)
                    for particle, actor, command, observed in unresolved
                    if command is not None
                )
            )
        queries: list[ModelQuery] = []
        for _index, particle, actor in hidden:
            observation, legal_actions = particle.branch.model_input(
                actor
            )
            queries.append(
                ModelQuery(
                    observation=observation,
                    legal_actions=legal_actions,
                    seat=actor,
                )
            )
        sampled = await self._model.sample(
            requests=tuple(
                PolicySampleRequest(
                    query=query,
                    draws=(
                        InferenceDrawKey(
                            seed=seed,
                            ordinal=index,
                        ),
                    ),
                )
                for index, query in enumerate(queries)
            ),
        )
        if isinstance(sampled, Rejected):
            return sampled
        commands_by_index: dict[int, commands.Command] = {}
        for (
            index,
            particle,
            actor,
        ), samples in zip(
            hidden,
            sampled.value,
            strict=True,
        ):
            assert len(samples.samples) == 1
            evaluation = samples.samples[0]
            command = physical_command(
                action=evaluation.action,
                hand=particle.branch.snapshot(actor).hand,
            )
            if isinstance(command, Rejected):
                return command
            commands_by_index[index] = command.value
        return Ok(
            tuple(
                (
                    particle,
                    actor,
                    (
                        command
                        if command is not None
                        else commands_by_index[index]
                    ),
                    observed,
                )
                for index, (
                    particle,
                    actor,
                    command,
                    observed,
                ) in enumerate(unresolved)
                if command is not None or index in commands_by_index
            )
        )


def _score_request(
    *,
    branch: SimulationBranch,
    actor: Seat,
    command: commands.Command,
) -> Ok[PolicyScoreRequest] | Rejected:
    observation, legal_actions = branch.model_input(actor)
    action = semantic_action(
        command=command,
        hand=branch.snapshot(actor).hand,
        legal_actions=legal_actions,
    )
    if isinstance(action, Rejected):
        return action
    return Ok(
        PolicyScoreRequest(
            query=ModelQuery(
                observation=observation,
                legal_actions=legal_actions,
                seat=actor,
            ),
            action=action.value,
        )
    )


def _normalize(
    branches: tuple[_WeightedBranch, ...],
) -> Ok[tuple[Particle, ...]] | Rejected:
    if not branches:
        return Rejected(reason="AI has no legal hidden world")
    maximum = max(item.log_weight for item in branches)
    unscaled = tuple(
        math.exp(item.log_weight - maximum) for item in branches
    )
    total = sum(unscaled)
    if not math.isfinite(total) or total <= 0.0:
        return Rejected(reason="AI particle weights are invalid")
    return Ok(
        tuple(
            Particle(
                branch=item.branch,
                weight=weight / total,
            )
            for item, weight in zip(
                branches,
                unscaled,
                strict=True,
            )
        )
    )


def _effective_sample_size(
    particles: tuple[Particle, ...],
) -> float:
    return 1.0 / sum(
        particle.weight * particle.weight for particle in particles
    )


def _systematic_resample(
    *,
    branches: tuple[_WeightedBranch, ...],
    count: int,
    random_source: random.Random,
) -> tuple[_WeightedBranch, ...]:
    assert branches
    assert count > 0
    normalized = _normalize(branches)
    assert isinstance(normalized, Ok)
    particles = normalized.value
    offset = random_source.random() / float(count)
    positions = tuple(
        offset + float(index) / float(count) for index in range(count)
    )
    result: list[_WeightedBranch] = []
    source_index = 0
    cumulative = particles[0].weight
    for position in positions:
        while (
            position > cumulative and source_index < len(particles) - 1
        ):
            source_index += 1
            cumulative += particles[source_index].weight
        result.append(
            _WeightedBranch(
                branch=particles[source_index].branch,
                log_weight=0.0,
            )
        )
    return tuple(result)


__all__ = ("BeliefInference", "Particle", "ParticleBelief")
