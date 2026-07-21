"""Device-resident PPO replay for fixed-vocabulary action traces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.training.semantic_actions.choices import ACTION_CHOICE_COUNT


@dataclass(frozen=True, slots=True)
class PPOReplayTensorBatch:
    """Recorded choices and their exact legal masks."""

    sample_count: int
    max_step_count: int
    active_step_count: int
    choice_ids_padded: Tensor
    active_sample_indices: Tensor
    active_step_indices: Tensor
    legal_choice_masks: Tensor
    step_counts: Tensor

    def __post_init__(self) -> None:
        assert self.sample_count > 0
        assert self.max_step_count > 0
        assert self.active_step_count >= 0
        assert self.choice_ids_padded.shape == (
            self.sample_count,
            self.max_step_count,
        )
        assert self.active_sample_indices.shape == (
            self.active_step_count,
        )
        assert self.active_step_indices.shape == (
            self.active_step_count,
        )
        assert self.legal_choice_masks.shape == (
            self.active_step_count,
            ACTION_CHOICE_COUNT,
        )
        assert self.step_counts.shape == (self.sample_count,)
        assert self.choice_ids_padded.dtype == torch.long
        assert self.active_sample_indices.dtype == torch.long
        assert self.active_step_indices.dtype == torch.long
        assert self.legal_choice_masks.dtype == torch.bool
        assert self.step_counts.dtype == torch.long


__all__ = ("PPOReplayTensorBatch",)
