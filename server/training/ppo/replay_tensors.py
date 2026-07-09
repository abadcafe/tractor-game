"""Device-resident PPO replay tensors for semantic token traces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class PPOReplayTensorBatch:
    """Recorded semantic-token replay tensors for one rollout."""

    sample_count: int
    max_step_count: int
    selected_token_ids_padded: Tensor
    choice_token_ids: Tensor
    choice_masks: Tensor
    selected_choice_offsets: Tensor
    step_mask: Tensor
    step_counts: Tensor

    def __post_init__(self) -> None:
        assert self.sample_count > 0
        assert self.max_step_count > 0
        assert self.selected_token_ids_padded.shape == (
            self.sample_count,
            self.max_step_count,
        )
        assert self.choice_token_ids.ndim == 3
        assert self.choice_token_ids.shape[:2] == (
            self.sample_count,
            self.max_step_count,
        )
        assert self.choice_masks.shape == self.choice_token_ids.shape
        assert self.selected_choice_offsets.shape == (
            self.sample_count,
            self.max_step_count,
        )
        assert self.step_mask.shape == (
            self.sample_count,
            self.max_step_count,
        )
        assert self.step_counts.shape == (self.sample_count,)
        assert self.selected_token_ids_padded.dtype == torch.long
        assert self.choice_token_ids.dtype == torch.int16
        assert self.choice_masks.dtype == torch.bool
        assert self.selected_choice_offsets.dtype == torch.long
        assert self.step_mask.dtype == torch.bool
        assert self.step_counts.dtype == torch.long
