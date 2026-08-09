"""Policy contract consumed by rollout actors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from server.foundation.result import Ok, Rejected
from server.policy_model.actions import (
    GeneratedAction,
    LegalActionSpace,
)
from server.policy_model.observation import Observation
from server.training.rollout_inference.randomness import (
    TrainingDecisionKey,
)
from server.training.rollout_inference.samples.records import (
    DecisionHandle,
)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """One sampled policy action and its exact replay reference."""

    action: GeneratedAction
    decision_handle: DecisionHandle
    legal_choice_count: int
    scored_choice_step_count: int

    def __post_init__(self) -> None:
        assert self.legal_choice_count > 1
        assert self.scored_choice_step_count > 0


class RolloutPolicy(Protocol):
    """Stochastic trainable policy used only at genuine choices."""

    async def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionSpace,
        decision_key: TrainingDecisionKey,
    ) -> Ok[PolicyDecision] | Rejected: ...


__all__ = ("PolicyDecision", "RolloutPolicy")
