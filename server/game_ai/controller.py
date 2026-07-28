"""Seat-local belief and policy-improvement composition."""

from __future__ import annotations

import logging
import random
import time
from typing import Protocol

from server.foundation.result import Ok, Rejected
from server.game import Seat, commands
from server.game.snapshots import PlayerSnapshot
from server.training.legal_actions import build_legal_action_index
from server.training.observation import build_observation
from server.training.observation_memory import ObservationMemory

from .belief import BeliefInference, ParticleBelief
from .queries import ModelQuery
from .search import SearchConfig, SearchInference, SearchPlanner

_LOGGER = logging.getLogger(__name__)


class AIControllerPort(Protocol):
    """Sequenced observation and decision boundary for AI players."""

    def observe(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
        error: str | None,
    ) -> Ok[None] | Rejected:
        """Record one complete player observation."""
        ...

    async def decide(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
    ) -> Ok[commands.Command] | Rejected:
        """Return one strategic command for the current view."""
        ...


class ControllerInference(
    BeliefInference,
    SearchInference,
    Protocol,
):
    """Exact model surface required by one AI controller."""


class AIController:
    """Produce commands from one contiguous seat-local game history."""

    def __init__(
        self,
        *,
        seat: Seat,
        model: ControllerInference,
        particle_count: int,
        search_config: SearchConfig,
        random_source: random.Random,
    ) -> None:
        self._seat = seat
        self._random = random_source
        self._memory = ObservationMemory()
        self._belief = ParticleBelief(
            viewer=seat,
            model=model,
            particle_count=particle_count,
            random_source=random_source,
        )
        self._search = SearchPlanner(
            model=model,
            config=search_config,
        )

    def observe(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
        error: str | None,
    ) -> Ok[None] | Rejected:
        """Consume one contiguous real player view."""
        remembered = self._memory.observe(
            seq=seq,
            snapshot=snapshot,
            error=error,
        )
        if isinstance(remembered, Rejected):
            return remembered
        return self._belief.record(
            seq=seq,
            snapshot=snapshot,
            memory=remembered.value,
            error=error,
        )

    async def decide(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
    ) -> Ok[commands.Command] | Rejected:
        """Run the same policy-improvement pipeline for every action."""
        if snapshot.awaiting_action not in (
            "bid",
            "stir",
            "discard",
            "play",
        ):
            return Rejected(
                reason="AI has no strategic action to decide"
            )
        started = time.perf_counter()
        decision_kind = snapshot.awaiting_action
        decision_seed = self._random.getrandbits(63)
        root_query = self._root_query(snapshot)
        proposals = await self._search.propose(
            root_query=root_query,
            decision_seed=decision_seed,
        )
        if isinstance(proposals, Rejected):
            return proposals
        if len(proposals.value.candidates) == 1:
            decided = self._search.bind_single(
                candidate=proposals.value.candidates[0],
                root_snapshot=snapshot,
            )
        else:
            particles = await self._belief.synchronize()
            if isinstance(particles, Rejected):
                return particles
            decided = await self._search.decide(
                proposals=proposals.value,
                particles=particles.value,
                viewer=self._seat,
                root_snapshot=snapshot,
                decision_seed=decision_seed,
            )
        if isinstance(decided, Rejected):
            return decided
        self._belief.submitted(seq=seq, command=decided.value.command)
        statistics = decided.value.statistics
        _LOGGER.info(
            "ai.decision kind=%s candidates=%d rounds=%d "
            "simulations=%d engine_advances=%d policy_rows=%d "
            "value_rows=%d elapsed_ms=%.3f",
            decision_kind,
            statistics.candidate_count,
            statistics.halving_round_count,
            statistics.macro_sample_count,
            statistics.engine_advance_count,
            statistics.policy_row_count,
            statistics.value_row_count,
            (time.perf_counter() - started) * 1000.0,
        )
        return Ok(decided.value.command)

    def _root_query(self, snapshot: PlayerSnapshot) -> ModelQuery:
        observation = build_observation(
            viewer=self._seat,
            snapshot=snapshot,
            memory=self._memory.view(),
        )
        return ModelQuery(
            observation=observation,
            legal_actions=build_legal_action_index(
                viewer=self._seat,
                snapshot=snapshot,
                query=observation.action_query,
            ),
            seat=self._seat,
        )


__all__ = (
    "AIController",
    "AIControllerPort",
    "ControllerInference",
)
