"""Belief-conditioned Gumbel sequential-halving policy improvement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from server.foundation.result import Ok, Rejected
from server.game import Seat, commands
from server.game.snapshots import PlayerSnapshot
from server.policy_model.actions import physical_command
from server.policy_model.inference import (
    ActionProposalRequest,
    ActionProposals,
    ActionSamples,
    PolicyQuery,
    PolicySampleRequest,
    ProposedAction,
    SamplingSeed,
)
from server.policy_model.observation import Observation

from ..belief import Particle
from ._macro_rollout import run_paired_macro_rollouts
from ._q_transform import transformed_q_values
from ._schedule import HalvingSchedule


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Finite root policy-improvement budget."""

    root_candidate_count: int
    macro_simulation_budget: int

    def __post_init__(self) -> None:
        assert self.root_candidate_count > 1
        assert self.macro_simulation_budget >= (
            2 * self.root_candidate_count - 2
        )


@dataclass(frozen=True, slots=True)
class SearchStatistics:
    """Exact work completed by one root search."""

    candidate_count: int
    halving_round_count: int
    macro_sample_count: int
    engine_advance_count: int
    policy_row_count: int
    value_row_count: int


@dataclass(frozen=True, slots=True)
class SearchDecision:
    """Selected command and observable execution statistics."""

    command: commands.Command
    statistics: SearchStatistics


class SearchInference(Protocol):
    """Model operations required by root policy improvement."""

    async def propose(
        self,
        *,
        requests: tuple[ActionProposalRequest, ...],
    ) -> Ok[tuple[ActionProposals, ...]] | Rejected:
        """Generate unique structured root candidates."""
        ...

    async def sample(
        self,
        *,
        requests: tuple[PolicySampleRequest, ...],
    ) -> Ok[tuple[ActionSamples, ...]] | Rejected:
        """Sample blueprint actions during macro rollouts."""
        ...

    async def values(
        self,
        *,
        observations: tuple[Observation, ...],
    ) -> Ok[tuple[float, ...]] | Rejected:
        """Estimate continuation values at root-viewer leaves."""
        ...


@dataclass(slots=True)
class _Estimate:
    count: int = 0
    total: float = 0.0

    def add(self, values: tuple[float, ...]) -> None:
        assert values
        assert all(math.isfinite(value) for value in values)
        self.count += len(values)
        self.total += sum(values)

    def mean(self) -> float:
        assert self.count > 0
        return self.total / float(self.count)


class SearchPlanner:
    """Improve the checkpoint policy under a weighted hidden belief."""

    def __init__(
        self,
        *,
        model: SearchInference,
        config: SearchConfig,
    ) -> None:
        self._model = model
        self._config = config

    async def propose(
        self,
        *,
        root_query: PolicyQuery,
        decision_seed: int,
    ) -> Ok[ActionProposals] | Rejected:
        """Generate root candidates before hidden-belief work."""
        proposed = await self._model.propose(
            requests=(
                ActionProposalRequest(
                    query=root_query,
                    candidate_count=self._config.root_candidate_count,
                    draw=SamplingSeed(
                        seed=decision_seed,
                        ordinal=0,
                    ),
                ),
            )
        )
        if isinstance(proposed, Rejected):
            return proposed
        assert len(proposed.value) == 1
        return Ok(proposed.value[0])

    async def decide(
        self,
        *,
        proposals: ActionProposals,
        particles: tuple[Particle, ...],
        viewer: Seat,
        root_snapshot: PlayerSnapshot,
        decision_seed: int,
    ) -> Ok[SearchDecision] | Rejected:
        """Run structured proposals and paired sequential halving."""
        assert particles
        candidates = proposals.candidates
        if len(candidates) == 1:
            return self.bind_single(
                candidate=candidates[0],
                root_snapshot=root_snapshot,
            )
        schedule = HalvingSchedule.build(
            candidate_count=len(candidates),
            sample_budget=self._config.macro_simulation_budget,
        )
        estimates = [_Estimate() for _candidate in candidates]
        survivors = tuple(range(len(candidates)))
        macro_samples = 0
        engine_advances = 0
        policy_rows = 0
        value_rows = 0
        for round_index, halving_round in enumerate(schedule.rounds):
            assert len(survivors) == halving_round.survivor_count
            active_candidates = tuple(
                candidates[index] for index in survivors
            )
            rolled = await run_paired_macro_rollouts(
                model=self._model,
                particles=particles,
                candidates=active_candidates,
                viewer=viewer,
                root_snapshot=root_snapshot,
                decision_seed=decision_seed,
                halving_round=round_index,
                sample_count=halving_round.samples_per_candidate,
            )
            if isinstance(rolled, Rejected):
                return rolled
            for survivor, values in zip(
                survivors,
                rolled.value.values,
                strict=True,
            ):
                estimates[survivor].add(values)
            stats = rolled.value.statistics
            macro_samples += stats.sample_count
            engine_advances += stats.engine_advance_count
            policy_rows += stats.policy_row_count
            value_rows += stats.value_row_count
            means = tuple(
                estimates[index].mean() for index in survivors
            )
            visits = tuple(
                estimates[index].count for index in survivors
            )
            transformed = transformed_q_values(
                means=means,
                visits=visits,
            )
            survivor_scores = tuple(
                candidates[index].perturbed_root_score
                + transformed[position]
                for position, index in enumerate(survivors)
            )
            keep_count = (len(survivors) + 1) // 2
            survivors = tuple(
                index
                for _score, index in sorted(
                    zip(survivor_scores, survivors, strict=True),
                    key=lambda item: (item[0], -item[1]),
                    reverse=True,
                )[:keep_count]
            )
        assert len(survivors) == 1
        selected = physical_command(
            action=candidates[survivors[0]].action,
            hand=root_snapshot.hand,
        )
        if isinstance(selected, Rejected):
            return selected
        return Ok(
            SearchDecision(
                command=selected.value,
                statistics=SearchStatistics(
                    candidate_count=len(candidates),
                    halving_round_count=len(schedule.rounds),
                    macro_sample_count=macro_samples,
                    engine_advance_count=engine_advances,
                    policy_row_count=policy_rows,
                    value_row_count=value_rows,
                ),
            )
        )

    def bind_single(
        self,
        *,
        candidate: ProposedAction,
        root_snapshot: PlayerSnapshot,
    ) -> Ok[SearchDecision] | Rejected:
        selected = physical_command(
            action=candidate.action,
            hand=root_snapshot.hand,
        )
        if isinstance(selected, Rejected):
            return selected
        return Ok(
            SearchDecision(
                command=selected.value,
                statistics=SearchStatistics(
                    candidate_count=1,
                    halving_round_count=0,
                    macro_sample_count=0,
                    engine_advance_count=0,
                    policy_row_count=0,
                    value_row_count=0,
                ),
            )
        )


__all__ = (
    "SearchConfig",
    "SearchDecision",
    "SearchInference",
    "SearchPlanner",
    "SearchStatistics",
)
