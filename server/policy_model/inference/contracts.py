"""Strongly typed production decision contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass

from server.policy_model.actions import GeneratedAction
from server.policy_model.actions.legality import LegalActionSpace
from server.policy_model.observation import Observation


@dataclass(frozen=True, slots=True)
class PolicyQuery:
    """One observation paired with its exact legal action grammar."""

    observation: Observation
    legal_actions: LegalActionSpace

    def __post_init__(self) -> None:
        assert self.legal_actions.query == self.observation.action_query


@dataclass(frozen=True, slots=True)
class SamplingSeed:
    """Stable random stream identity independent of runtime batching."""

    seed: int
    ordinal: int

    def __post_init__(self) -> None:
        assert self.seed >= 0
        assert self.ordinal >= 0


@dataclass(frozen=True, slots=True)
class ActionDecisionRequest:
    """One complete policy-improvement decision."""

    query: PolicyQuery
    candidate_count: int
    action_value_temperature: float
    draw: SamplingSeed

    def __post_init__(self) -> None:
        assert self.candidate_count > 0
        assert math.isfinite(self.action_value_temperature)
        assert self.action_value_temperature > 0.0


@dataclass(frozen=True, slots=True)
class ActionDecision:
    """The selected legal action and decision diagnostics."""

    action: GeneratedAction
    candidate_count: int
    selected_action_value: float | None

    def __post_init__(self) -> None:
        assert self.candidate_count > 0
        if self.selected_action_value is not None:
            assert math.isfinite(self.selected_action_value)


__all__ = (
    "ActionDecision",
    "ActionDecisionRequest",
    "PolicyQuery",
    "SamplingSeed",
)
