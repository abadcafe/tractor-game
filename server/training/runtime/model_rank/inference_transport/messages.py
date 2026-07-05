"""Typed inference transport envelopes."""

from __future__ import annotations

from dataclasses import dataclass

from server.training.policy_request_frame import (
    PolicyRequestBatchFrame,
    PolicyRequestFrame,
    PolicyResponseFrame,
)


@dataclass(frozen=True, slots=True)
class PolicyInferenceRequest:
    """One framed worker-to-model-rank inference request."""

    worker_index: int
    request_id: int
    frame: PolicyRequestFrame
    byte_count: int = 0

    def __post_init__(self) -> None:
        assert self.worker_index >= 0
        assert self.request_id >= 0
        assert self.byte_count >= 0


@dataclass(frozen=True, slots=True)
class PolicyInferenceRequestControl:
    """One worker-to-model-rank shared-memory request descriptor."""

    worker_index: int
    request_id: int
    byte_count: int
    slot_name: str

    def __post_init__(self) -> None:
        assert self.worker_index >= 0
        assert self.request_id >= 0
        assert self.byte_count > 0
        assert self.slot_name


@dataclass(frozen=True, slots=True)
class PolicyInferenceRequestBatch:
    """One model-rank inference batch from one or more workers."""

    requests: tuple[PolicyInferenceRequest, ...]

    def __post_init__(self) -> None:
        assert self.requests

    def request_frame_batch(self) -> PolicyRequestBatchFrame:
        """Return the model-rank compute payload for this batch."""
        return PolicyRequestBatchFrame(
            frames=tuple(request.frame for request in self.requests)
        )

    def byte_count(self) -> int:
        """Return total transport bytes represented by this batch."""
        return sum(request.byte_count for request in self.requests)


@dataclass(frozen=True, slots=True)
class PolicyInferenceResponseEnvelope:
    """One framed model-rank-to-worker inference response."""

    worker_index: int
    request_id: int
    frame: PolicyResponseFrame

    def __post_init__(self) -> None:
        assert self.worker_index >= 0
        assert self.request_id >= 0
