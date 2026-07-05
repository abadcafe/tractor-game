"""Policy inference frames independent of Python rule objects."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from server import result as _result
from server.result import Ok, Rejected
from server.training.legal_actions import LegalActionIndex
from server.training.observation import Observation
from server.training.packed_observation import (
    OBSERVATION_COMPONENT_COUNT,
    ComponentRow,
    NumericRow,
    PackedObservation,
    pack_observation,
)
from server.training.policy import PolicyDecision
from server.training.policy_sampling import DecisionHandle
from server.training.sampling import (
    PolicyDecisionKey,
    policy_choice_threshold,
)
from server.training.semantic_action_plan import (
    ActionPlanFrame,
    DeviceActionPlanBatch,
    compile_legal_action_frame,
    plan_batch_to_device,
    semantic_trace_from_token_ids,
)
from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.tensorize import (
    ObservationTensorBatch,
    tensorize_packed_observations,
)


@dataclass(frozen=True, slots=True)
class PolicyRequestFrame:
    """One compact policy request payload."""

    component_rows: tuple[ComponentRow, ...]
    numeric_value_rows: tuple[NumericRow, ...]
    numeric_mask_rows: tuple[NumericRow, ...]
    action_plan: ActionPlanFrame
    decision_key: PolicyDecisionKey

    def __post_init__(self) -> None:
        assert self.component_rows
        assert len(self.component_rows) == len(self.numeric_value_rows)
        assert len(self.component_rows) == len(self.numeric_mask_rows)
        assert all(
            len(row) == OBSERVATION_COMPONENT_COUNT
            for row in self.component_rows
        )

    def token_count(self) -> int:
        """Return observation token count."""
        return len(self.component_rows)

    def packed_observation(self) -> PackedObservation:
        """Return an internal packed observation view."""
        return PackedObservation(
            component_rows=self.component_rows,
            numeric_value_rows=self.numeric_value_rows,
            numeric_mask_rows=self.numeric_mask_rows,
        )


@dataclass(frozen=True, slots=True)
class PolicyRequestBatchFrame:
    """Batched policy request frames ready for model-rank staging."""

    frames: tuple[PolicyRequestFrame, ...]

    def __post_init__(self) -> None:
        assert self.frames

    def batch_size(self) -> int:
        """Return the number of request frames."""
        return len(self.frames)


@dataclass(frozen=True, slots=True)
class CompletedPolicyResponseFrame:
    """Successful model-rank inference response."""

    trace_token_ids: tuple[int, ...]
    decision_handle: DecisionHandle
    choice_count: int

    def __post_init__(self) -> None:
        assert self.trace_token_ids
        assert self.choice_count > 0


@dataclass(frozen=True, slots=True)
class RejectedPolicyResponseFrame:
    """Rejected model-rank inference response."""

    reason: str

    def __post_init__(self) -> None:
        assert self.reason


type PolicyResponseFrame = (
    CompletedPolicyResponseFrame | RejectedPolicyResponseFrame
)


@dataclass(frozen=True, slots=True)
class DevicePolicyRequestBatch:
    """Policy request batch tensorized on one model-rank device."""

    observation_batch: ObservationTensorBatch
    action_plan_batch: DeviceActionPlanBatch
    sampling_thresholds: torch.Tensor

    def __post_init__(self) -> None:
        batch_size = self.action_plan_batch.batch_size()
        assert self.sampling_thresholds.shape == (
            batch_size,
            SEMANTIC_CODEC.max_argument_tokens,
        )
        assert self.sampling_thresholds.dtype == torch.float64


def build_policy_request_frame(
    *,
    observation: Observation,
    legal_actions: LegalActionIndex,
    decision_key: PolicyDecisionKey,
) -> _result.Ok[PolicyRequestFrame] | _result.Rejected:
    """Build a runtime frame from CPU game/rule data."""
    packed = pack_observation(observation)
    return Ok(
        value=PolicyRequestFrame(
            component_rows=packed.component_rows,
            numeric_value_rows=packed.numeric_value_rows,
            numeric_mask_rows=packed.numeric_mask_rows,
            action_plan=compile_legal_action_frame(legal_actions),
            decision_key=decision_key,
        )
    )


def policy_request_batch_to_device(
    *,
    batch: PolicyRequestBatchFrame,
    max_observation_tokens: int,
    device: torch.device,
) -> DevicePolicyRequestBatch:
    """Tensorize policy request frames on a model-rank device."""
    frames = batch.frames
    return DevicePolicyRequestBatch(
        observation_batch=tensorize_packed_observations(
            observations=tuple(
                frame.packed_observation() for frame in frames
            ),
            max_observation_tokens=max_observation_tokens,
            device=device,
        ),
        action_plan_batch=plan_batch_to_device(
            tuple(frame.action_plan for frame in frames),
            device=device,
        ),
        sampling_thresholds=torch.tensor(
            tuple(
                tuple(
                    policy_choice_threshold(
                        key=frame.decision_key,
                        argument_index=argument_index,
                    )
                    for argument_index in range(
                        SEMANTIC_CODEC.max_argument_tokens
                    )
                )
                for frame in frames
            ),
            dtype=torch.float64,
            device=device,
        ),
    )


def decode_policy_response(
    *,
    legal_actions: LegalActionIndex,
    response: PolicyResponseFrame,
) -> Ok[PolicyDecision] | Rejected:
    """Decode a model-rank response through the rule-layer index."""
    if isinstance(response, RejectedPolicyResponseFrame):
        return Rejected(reason=response.reason)
    trace_result = semantic_trace_from_token_ids(
        response.trace_token_ids
    )
    if isinstance(trace_result, Rejected):
        return trace_result
    decoded = legal_actions.decode(trace_result.value)
    if isinstance(decoded, Rejected):
        return decoded
    return Ok(
        value=PolicyDecision(
            action=decoded.value,
            decision_handle=response.decision_handle,
            choice_count=response.choice_count,
        )
    )
