"""Seat-local AI history, belief, and search composition."""

from __future__ import annotations

import random

from server.foundation.result import Ok, Rejected
from server.game import Seat, commands
from server.game.snapshots import PlayerSnapshot
from server.training.legal_actions import build_legal_action_index
from server.training.observation import build_observation
from server.training.observation_memory import ObservationMemory

from .actions import physical_command
from .belief import ParticleBelief
from .model import ModelEvaluator, ModelQuery
from .search import ParticleSearch, SearchConfig


class AIController:
    """Produce one command from a contiguous seat-local game history."""

    def __init__(
        self,
        *,
        seat: Seat,
        model: ModelEvaluator,
        particle_count: int,
        direct_samples: int,
        search_config: SearchConfig,
        random_source: random.Random,
    ) -> None:
        assert direct_samples > 0
        self._seat = seat
        self._model = model
        self._direct_samples = direct_samples
        self._random = random_source
        self._memory = ObservationMemory()
        self._belief = ParticleBelief(
            viewer=seat,
            model=model,
            particle_count=particle_count,
            random_source=random_source,
        )
        self._search = ParticleSearch(
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

    def decide(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
    ) -> Ok[commands.Command] | Rejected:
        """Run direct model inference or hidden-information search."""
        action = snapshot.awaiting_action
        if action not in ("bid", "stir", "discard", "play"):
            return Rejected(
                reason="AI has no strategic action to decide"
            )
        decision_seed = self._random.getrandbits(63)
        if action == "play":
            particles = self._belief.synchronize()
            if isinstance(particles, Rejected):
                return particles
            decided = self._search.decide(
                particles=particles.value,
                viewer=self._seat,
                decision_seed=decision_seed,
            )
        else:
            synchronized = self._belief.synchronize()
            if isinstance(synchronized, Rejected):
                return synchronized
            decided = self._direct_decision(
                snapshot=snapshot,
                decision_seed=decision_seed,
            )
        if isinstance(decided, Rejected):
            return decided
        self._belief.submitted(seq=seq, command=decided.value)
        return decided

    def _direct_decision(
        self,
        *,
        snapshot: PlayerSnapshot,
        decision_seed: int,
    ) -> Ok[commands.Command] | Rejected:
        observation = build_observation(
            viewer=self._seat,
            snapshot=snapshot,
            memory=self._memory.view(),
        )
        legal_actions = build_legal_action_index(
            viewer=self._seat,
            snapshot=snapshot,
            query=observation.action_query,
        )
        query = ModelQuery(
            observation=observation,
            legal_actions=legal_actions,
            seat=self._seat,
        )
        sampled = self._model.sample(
            queries=tuple(query for _ in range(self._direct_samples)),
            decision_seed=decision_seed,
        )
        if isinstance(sampled, Rejected):
            return sampled
        selected = max(
            sampled.value,
            key=lambda evaluation: evaluation.log_probability,
        )
        return physical_command(
            action=selected.action,
            hand=snapshot.hand,
        )


__all__ = ("AIController",)
