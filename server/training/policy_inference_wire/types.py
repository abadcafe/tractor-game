"""Public records for policy inference wire messages."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.training.semantic_action_plan import DeviceActionPlanBatch
from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.tensorize import ObservationTensorBatch


@dataclass(frozen=True, slots=True)
class PolicyRequestRoute:
    """Worker/request route carried by every inference wire message."""

    worker_index: int
    request_id: int

    def __post_init__(self) -> None:
        assert self.worker_index >= 0
        assert self.request_id >= 0


@dataclass(frozen=True, slots=True)
class PolicyRequestWire:
    """One complete binary policy inference request."""

    data: bytes

    def __post_init__(self) -> None:
        assert self.data

    def byte_count(self) -> int:
        """Return wire byte length."""
        return len(self.data)


@dataclass(frozen=True, slots=True)
class PolicyRequestWireBatch:
    """A batch of complete wire requests."""

    requests: tuple[PolicyRequestWire, ...]

    def __post_init__(self) -> None:
        assert self.requests

    def batch_size(self) -> int:
        """Return request count."""
        return len(self.requests)

    def byte_count(self) -> int:
        """Return total wire byte length."""
        return sum(request.byte_count() for request in self.requests)


@dataclass(frozen=True, slots=True)
class PolicyRequestMetadata:
    """Small CPU metadata parsed from a request header."""

    route: PolicyRequestRoute
    byte_count: int
    token_count: int
    trace_count: int
    trace_steps: int
    pair_plan_count: int
    policy_version: int

    def __post_init__(self) -> None:
        assert self.byte_count > 0
        assert self.token_count > 0
        assert self.trace_count > 0
        assert self.trace_steps > 0
        assert self.pair_plan_count > 0
        assert self.policy_version >= 0


@dataclass(frozen=True, slots=True)
class DevicePolicyRequestBatch:
    """Policy inference request tensors resident on one torch device."""

    observation_batch: ObservationTensorBatch
    action_plan_batch: DeviceActionPlanBatch
    sampling_thresholds: Tensor
    policy_versions: tuple[int, ...]

    def __post_init__(self) -> None:
        batch_size = self.action_plan_batch.batch_size()
        assert int(self.observation_batch.component_ids.shape[0]) == (
            batch_size
        )
        assert self.sampling_thresholds.shape == (
            batch_size,
            SEMANTIC_CODEC.max_argument_tokens,
        )
        assert self.sampling_thresholds.dtype == torch.float64
        assert len(self.policy_versions) == batch_size
        assert all(version >= 0 for version in self.policy_versions)


@dataclass(frozen=True, slots=True)
class PolicyResponseWire:
    """One complete binary policy inference response."""

    data: bytes

    def __post_init__(self) -> None:
        assert self.data


@dataclass(frozen=True, slots=True)
class CompletedPolicyResponse:
    """Successful policy inference response."""

    route: PolicyRequestRoute
    trace_token_ids: tuple[int, ...]
    decision_handle_model_rank: int
    decision_handle_policy_version: int
    decision_handle_slot_index: int
    decision_handle_slot_generation: int
    choice_count: int

    def __post_init__(self) -> None:
        assert self.trace_token_ids
        assert self.decision_handle_model_rank >= 0
        assert self.decision_handle_policy_version >= 0
        assert self.decision_handle_slot_index >= 0
        assert self.decision_handle_slot_generation >= 0
        assert self.choice_count > 0


@dataclass(frozen=True, slots=True)
class RejectedPolicyResponse:
    """Rejected policy inference response."""

    route: PolicyRequestRoute
    reason: str

    def __post_init__(self) -> None:
        assert self.reason


type PolicyResponse = CompletedPolicyResponse | RejectedPolicyResponse
