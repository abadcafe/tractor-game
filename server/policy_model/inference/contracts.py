"""Strongly typed production policy decision contracts."""

from __future__ import annotations

from dataclasses import dataclass

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
class PolicyDecisionRequest:
    """One complete action sampled from a policy query."""

    query: PolicyQuery
    draw: SamplingSeed


__all__ = (
    "PolicyDecisionRequest",
    "PolicyQuery",
    "SamplingSeed",
)
