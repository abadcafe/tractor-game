"""Public records for policy inference response wire messages."""

from __future__ import annotations

from dataclasses import dataclass

from server.training.policy_inference_batch.types import (
    PolicyRequestRoute,
)
from server.training.policy_sampling import CompactActionChoiceIds


@dataclass(frozen=True, slots=True)
class PolicyResponseBatchWire:
    """One complete binary policy inference response batch frame."""

    data: bytes

    def __post_init__(self) -> None:
        assert self.data

    def byte_count(self) -> int:
        """Return frame byte length."""
        return len(self.data)


@dataclass(frozen=True, slots=True)
class CompletedPolicyResponse:
    """Successful policy inference response."""

    route: PolicyRequestRoute
    action_choice_ids: CompactActionChoiceIds
    decision_handle_model_rank: int
    decision_handle_policy_version: int
    decision_handle_row_index: int
    choice_count: int

    def __post_init__(self) -> None:
        assert len(self.action_choice_ids) > 0
        assert self.decision_handle_model_rank >= 0
        assert self.decision_handle_policy_version >= 0
        assert self.decision_handle_row_index >= 0
        assert self.choice_count > 0


@dataclass(frozen=True, slots=True)
class RejectedPolicyResponse:
    """Rejected policy inference response."""

    route: PolicyRequestRoute
    reason: str

    def __post_init__(self) -> None:
        assert self.reason


type PolicyResponse = CompletedPolicyResponse | RejectedPolicyResponse
