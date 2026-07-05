"""Device-local policy sampling records."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from server.training.tensorize import ObservationTensorBatch


@dataclass(frozen=True, slots=True)
class DecisionHandle:
    """Stable reference to one model-rank-owned replay record."""

    model_rank_index: int
    policy_version: int
    slot_index: int
    slot_generation: int

    def __post_init__(self) -> None:
        assert self.model_rank_index >= 0
        assert self.policy_version >= 0
        assert self.slot_index >= 0
        assert self.slot_generation >= 0


@dataclass(frozen=True, slots=True)
class DeviceDecisionReplayRecord:
    """One sampled decision retained on the model-rank device."""

    policy_version: int
    observation_batch: ObservationTensorBatch
    selected_token_ids: Tensor
    legal_token_masks: Tensor
    old_log_probability: Tensor
    old_value: Tensor

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert int(self.observation_batch.component_ids.shape[0]) == 1
        assert int(self.observation_batch.numeric_values.shape[0]) == 1
        assert int(self.observation_batch.numeric_masks.shape[0]) == 1
        assert self.selected_token_ids.ndim == 1
        assert self.legal_token_masks.ndim == 2
        assert int(self.legal_token_masks.shape[0]) == int(
            self.selected_token_ids.shape[0]
        )
        assert self.old_log_probability.ndim == 0
        assert self.old_value.ndim == 0
        device = self.observation_batch.component_ids.device
        assert self.old_log_probability.device == device
        assert self.old_value.device == device
        assert self.selected_token_ids.device == device
        assert self.legal_token_masks.device == device


@dataclass(frozen=True, slots=True)
class SampledPolicyDecision:
    """Torch sampler output before a model rank assigns a handle."""

    trace_token_ids: tuple[int, ...]
    replay_record: DeviceDecisionReplayRecord
    choice_count: int

    def __post_init__(self) -> None:
        assert self.trace_token_ids
        assert self.choice_count > 0


@dataclass(frozen=True, slots=True)
class ModelRankPolicyDecision:
    """Model-rank response before worker-side action decoding."""

    trace_token_ids: tuple[int, ...]
    decision_handle: DecisionHandle
    choice_count: int

    def __post_init__(self) -> None:
        assert self.trace_token_ids
        assert self.choice_count > 0
