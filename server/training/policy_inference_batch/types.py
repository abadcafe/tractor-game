"""Public request records for policy inference batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.training.legal_actions import LegalActionIndex
from server.training.observation import Observation
from server.training.policy_inference_batch.schema import (
    PolicyRequestBatchLayout,
)
from server.training.sampling import PolicyDecisionKey
from server.training.semantic_action_plan import DeviceActionPlanBatch
from server.training.tensorize import ObservationTensorBatch


@dataclass(frozen=True, slots=True)
class PolicyRequestRoute:
    """Worker/request route carried by every inference message."""

    worker_index: int
    request_id: int

    def __post_init__(self) -> None:
        assert self.worker_index >= 0
        assert self.request_id >= 0


@dataclass(frozen=True, slots=True)
class PolicyRequestInput:
    """One worker-side policy request before batch materialization."""

    route: PolicyRequestRoute
    observation: Observation
    legal_actions: LegalActionIndex
    decision_key: PolicyDecisionKey

    def __post_init__(self) -> None:
        assert self.decision_key.policy_version >= 0


@dataclass(frozen=True, slots=True, init=False)
class PolicyRequestWireFrame:
    """One borrowed columnar binary policy inference request frame."""

    _buffer: bytearray
    _byte_count: int

    def __init__(self, *, buffer: bytearray, byte_count: int) -> None:
        assert buffer
        assert byte_count > 0
        assert byte_count <= len(buffer)
        object.__setattr__(self, "_buffer", buffer)
        object.__setattr__(self, "_byte_count", byte_count)

    @property
    def byte_count(self) -> int:
        """Return active frame byte count."""
        return self._byte_count

    def view(self) -> memoryview:
        """Return the active frame bytes without copying."""
        return memoryview(self._buffer)[: self._byte_count]


@dataclass(frozen=True, slots=True)
class PolicyRequestFrameMetadata:
    """CPU metadata parsed from one columnar request batch frame."""

    row_count: int
    batch_capacity: int
    observation_token_capacity: int
    padded_generation_steps: int
    generation_step_counts: tuple[int, ...]
    routes: tuple[PolicyRequestRoute, ...]
    policy_versions: tuple[int, ...]
    byte_count: int
    layout: PolicyRequestBatchLayout

    def __post_init__(self) -> None:
        assert self.row_count > 0
        assert self.batch_capacity >= self.row_count
        assert self.observation_token_capacity > 0
        assert self.padded_generation_steps > 0
        assert len(self.generation_step_counts) == self.row_count
        assert all(count > 0 for count in self.generation_step_counts)
        assert all(
            count <= self.padded_generation_steps
            for count in self.generation_step_counts
        )
        assert len(self.routes) == self.row_count
        assert len(self.policy_versions) == self.row_count
        assert all(version >= 0 for version in self.policy_versions)
        assert self.byte_count > 0
        assert self.layout.batch_capacity == self.batch_capacity
        assert self.layout.observation_token_capacity == (
            self.observation_token_capacity
        )
        assert self.layout.padded_generation_steps == (
            self.padded_generation_steps
        )
        assert self.layout.total_bytes == self.byte_count


@dataclass(frozen=True, slots=True)
class BorrowedPolicyRequestBatch:
    """Transport-ready batch borrowed from compiler workspace."""

    frame: PolicyRequestWireFrame
    metadata: PolicyRequestFrameMetadata

    def __post_init__(self) -> None:
        assert self.frame.byte_count == self.metadata.byte_count

    @property
    def routes(self) -> tuple[PolicyRequestRoute, ...]:
        """Return response routes for this compiled batch."""
        return self.metadata.routes

    @property
    def policy_versions(self) -> tuple[int, ...]:
        """Return request policy versions."""
        return self.metadata.policy_versions

    @property
    def generation_step_counts(self) -> tuple[int, ...]:
        """Return semantic generation step counts."""
        return self.metadata.generation_step_counts

    @property
    def observation_token_capacity(self) -> int:
        """Return this lossless batch's padded token count."""
        return self.metadata.observation_token_capacity

    @property
    def padded_generation_steps(self) -> int:
        """Return the padded semantic generation width."""
        return self.metadata.padded_generation_steps

    def row_count(self) -> int:
        """Return request count."""
        return self.metadata.row_count


@dataclass(frozen=True, slots=True)
class DevicePolicyRequestBatch:
    """Policy inference request tensors resident on one torch device."""

    observation_batch: ObservationTensorBatch
    action_plan_batch: DeviceActionPlanBatch
    sampling_thresholds: Tensor
    generation_step_counts: Tensor
    policy_versions: tuple[int, ...]
    padded_generation_steps: int

    def __post_init__(self) -> None:
        batch_size = self.action_plan_batch.batch_size()
        assert self.padded_generation_steps > 0
        assert int(self.observation_batch.category_ids.shape[0]) == (
            batch_size
        )
        assert self.sampling_thresholds.shape == (
            batch_size,
            self.padded_generation_steps,
        )
        expected_threshold_dtype = (
            torch.float32
            if self.action_plan_batch.device.type == "mps"
            else torch.float64
        )
        assert (
            self.sampling_thresholds.dtype == expected_threshold_dtype
        )
        assert self.generation_step_counts.shape == (batch_size,)
        assert self.generation_step_counts.dtype == torch.long
        assert len(self.policy_versions) == batch_size
        assert all(version >= 0 for version in self.policy_versions)
