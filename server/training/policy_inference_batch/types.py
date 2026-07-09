"""Public request records for policy inference batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.training.legal_actions import LegalActionIndex
from server.training.observation import Observation
from server.training.packed_observation import PackedObservation
from server.training.sampling import PolicyDecisionKey
from server.training.semantic_action_plan import (
    ActionPlanFrame,
    DeviceActionPlanBatch,
)
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


@dataclass(frozen=True, slots=True)
class PolicyRequestBatchRow:
    """Tensor-ready policy request row hidden behind a batch."""

    route: PolicyRequestRoute
    policy_version: int
    packed_observation: PackedObservation
    action_plan: ActionPlanFrame
    generation_step_count: int
    sampling_thresholds: tuple[float, ...]

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert self.generation_step_count > 0
        assert (
            len(self.sampling_thresholds) == self.generation_step_count
        )


@dataclass(frozen=True, slots=True)
class PolicyRequestBatch:
    """Transport-neutral policy request batch prepared by a worker."""

    rows: tuple[PolicyRequestBatchRow, ...]
    max_observation_tokens: int
    padded_generation_steps: int

    def __post_init__(self) -> None:
        assert self.rows
        assert self.max_observation_tokens > 0
        assert self.padded_generation_steps > 0
        assert all(
            row.generation_step_count <= self.padded_generation_steps
            for row in self.rows
        )
        assert all(
            row.packed_observation.token_count()
            <= self.max_observation_tokens
            for row in self.rows
        )

    def row_count(self) -> int:
        """Return request count."""
        return len(self.rows)

    @property
    def routes(self) -> tuple[PolicyRequestRoute, ...]:
        """Return request routes in batch order."""
        return tuple(row.route for row in self.rows)

    @property
    def policy_versions(self) -> tuple[int, ...]:
        """Return policy versions in batch order."""
        return tuple(row.policy_version for row in self.rows)

    @property
    def generation_step_counts(self) -> tuple[int, ...]:
        """Return semantic generation widths in batch order."""
        return tuple(row.generation_step_count for row in self.rows)


@dataclass(frozen=True, slots=True)
class PolicyRequestWireFrame:
    """One columnar binary policy inference request batch."""

    data: bytearray
    byte_count: int

    def __post_init__(self) -> None:
        assert self.data
        assert self.byte_count > 0
        assert self.byte_count <= len(self.data)

    def view(self) -> memoryview:
        """Return the active frame bytes without copying."""
        return memoryview(self.data)[: self.byte_count]


@dataclass(frozen=True, slots=True)
class PolicyRequestFrameMetadata:
    """CPU metadata parsed from one columnar request batch frame."""

    row_count: int
    batch_capacity: int
    max_observation_tokens: int
    padded_generation_steps: int
    generation_step_counts: tuple[int, ...]
    routes: tuple[PolicyRequestRoute, ...]
    policy_versions: tuple[int, ...]
    byte_count: int

    def __post_init__(self) -> None:
        assert self.row_count > 0
        assert self.batch_capacity >= self.row_count
        assert self.max_observation_tokens > 0
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
        assert int(self.observation_batch.component_ids.shape[0]) == (
            batch_size
        )
        assert self.sampling_thresholds.shape == (
            batch_size,
            self.padded_generation_steps,
        )
        assert self.sampling_thresholds.dtype == torch.float64
        assert self.generation_step_counts.shape == (batch_size,)
        assert self.generation_step_counts.dtype == torch.long
        assert len(self.policy_versions) == batch_size
        assert all(version >= 0 for version in self.policy_versions)
