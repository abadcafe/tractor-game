"""Strongly typed model queries shared by AI consumers and inference."""

from __future__ import annotations

import math
from dataclasses import dataclass

from server.game import Seat
from server.policy_model.actions import GeneratedAction
from server.policy_model.actions.legality import LegalActionSpace
from server.policy_model.observation import Observation


@dataclass(frozen=True, slots=True)
class PolicyQuery:
    """One observation paired with its exact legal action grammar."""

    observation: Observation
    legal_actions: LegalActionSpace
    seat: Seat


@dataclass(frozen=True, slots=True)
class SamplingSeed:
    """Stable random stream identity independent of runtime batching."""

    seed: int
    ordinal: int

    def __post_init__(self) -> None:
        assert self.seed >= 0
        assert self.ordinal >= 0


@dataclass(frozen=True, slots=True)
class SampledAction:
    """One sampled legal semantic action and its policy probability."""

    action: GeneratedAction
    log_probability: float

    def __post_init__(self) -> None:
        assert math.isfinite(self.log_probability)


@dataclass(frozen=True, slots=True)
class ActionSamples:
    """Samples returned for one model query in draw-key order."""

    samples: tuple[SampledAction, ...]

    def __post_init__(self) -> None:
        assert self.samples


@dataclass(frozen=True, slots=True)
class PolicySampleRequest:
    """One model query paired with explicit independent draws."""

    query: PolicyQuery
    draws: tuple[SamplingSeed, ...]

    def __post_init__(self) -> None:
        assert self.draws
        assert len(set(self.draws)) == len(self.draws)


@dataclass(frozen=True, slots=True)
class PolicyScoreRequest:
    """One supplied action to score in its originating query."""

    query: PolicyQuery
    action: GeneratedAction


@dataclass(frozen=True, slots=True)
class ProposedAction:
    """One unique root action from structured Gumbel Top-k."""

    action: GeneratedAction
    log_probability: float
    perturbed_root_score: float

    def __post_init__(self) -> None:
        assert math.isfinite(self.log_probability)
        assert math.isfinite(self.perturbed_root_score)


@dataclass(frozen=True, slots=True)
class ActionProposals:
    """Ordered unique root candidates from one proposal request."""

    candidates: tuple[ProposedAction, ...]

    def __post_init__(self) -> None:
        assert self.candidates


@dataclass(frozen=True, slots=True)
class ActionProposalRequest:
    """Structured no-replacement proposal request."""

    query: PolicyQuery
    candidate_count: int
    draw: SamplingSeed

    def __post_init__(self) -> None:
        assert self.candidate_count > 0


__all__ = (
    "ActionProposalRequest",
    "ActionProposals",
    "ActionSamples",
    "SamplingSeed",
    "PolicyQuery",
    "PolicySampleRequest",
    "PolicyScoreRequest",
    "ProposedAction",
    "SampledAction",
)
