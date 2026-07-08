"""Device-local policy sampling records."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.training.semantic_actions.codec import SEMANTIC_CODEC
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
class SampledPolicyBatch:
    """Device policy samples before model-rank slot assignment."""

    policy_versions: tuple[int, ...]
    status_codes: Tensor
    observation_batch: ObservationTensorBatch
    selected_token_ids_padded: Tensor
    legal_choice_ids_padded: Tensor
    legal_choice_masks_padded: Tensor
    selected_choice_offsets_padded: Tensor
    step_counts: Tensor
    choice_counts: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor

    def __post_init__(self) -> None:
        batch_size = len(self.policy_versions)
        assert batch_size > 0
        assert all(version >= 0 for version in self.policy_versions)
        assert self.status_codes.shape == (batch_size,)
        assert self.status_codes.dtype == torch.long
        assert self.selected_token_ids_padded.shape == (
            batch_size,
            SEMANTIC_CODEC.max_argument_tokens,
        )
        assert self.selected_token_ids_padded.dtype == torch.long
        assert self.legal_choice_ids_padded.ndim == 3
        assert self.legal_choice_ids_padded.shape[:2] == (
            batch_size,
            SEMANTIC_CODEC.max_argument_tokens,
        )
        assert self.legal_choice_ids_padded.dtype == torch.int16
        assert self.legal_choice_masks_padded.shape == (
            self.legal_choice_ids_padded.shape
        )
        assert self.legal_choice_masks_padded.dtype == torch.bool
        assert self.selected_choice_offsets_padded.shape == (
            batch_size,
            SEMANTIC_CODEC.max_argument_tokens,
        )
        assert self.selected_choice_offsets_padded.dtype == torch.long
        assert self.step_counts.shape == (batch_size,)
        assert self.step_counts.dtype == torch.long
        assert self.choice_counts.shape == (batch_size,)
        assert self.choice_counts.dtype == torch.long
        assert self.old_log_probabilities.shape == (batch_size,)
        assert self.old_values.shape == (batch_size,)
        assert int(self.observation_batch.component_ids.shape[0]) == (
            batch_size
        )
        device = self.observation_batch.component_ids.device
        assert self.status_codes.device == device
        assert self.selected_token_ids_padded.device == device
        assert self.legal_choice_ids_padded.device == device
        assert self.legal_choice_masks_padded.device == device
        assert self.selected_choice_offsets_padded.device == device
        assert self.step_counts.device == device
        assert self.choice_counts.device == device
        assert self.old_log_probabilities.device == device
        assert self.old_values.device == device


@dataclass(frozen=True, slots=True)
class ModelRankPolicyDecision:
    """Model-rank response before worker-side action decoding."""

    trace_token_ids: tuple[int, ...]
    decision_handle: DecisionHandle
    choice_count: int

    def __post_init__(self) -> None:
        assert self.trace_token_ids
        assert self.choice_count > 0


@dataclass(frozen=True, slots=True)
class RankReturnBatch:
    """Rank-local tensor-ready return targets for stored decisions."""

    policy_version: int
    model_rank_index: int
    slot_indices: Tensor
    slot_generations: Tensor
    return_values: Tensor
    round_count: int

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert self.model_rank_index >= 0
        assert self.slot_indices.ndim == 1
        assert self.slot_generations.shape == self.slot_indices.shape
        assert self.return_values.shape == self.slot_indices.shape
        assert self.slot_indices.dtype == torch.long
        assert self.slot_generations.dtype == torch.long
        assert self.return_values.dtype == torch.float32
        assert self.round_count >= 0

    def is_empty(self) -> bool:
        """Return whether this rank has no committed samples."""
        return int(self.slot_indices.shape[0]) == 0
