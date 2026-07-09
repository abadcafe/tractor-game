"""Fixed-width device legal candidates for semantic action plans."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class DeviceLegalCandidateBatch:
    """Fixed-width legal token candidates on one torch device."""

    token_ids: Tensor
    masks: Tensor
    choice_counts: Tensor

    def __post_init__(self) -> None:
        assert self.token_ids.ndim == 2
        assert self.masks.shape == self.token_ids.shape
        assert self.choice_counts.ndim == 1
        assert int(self.choice_counts.shape[0]) == int(
            self.token_ids.shape[0]
        )
        assert self.token_ids.dtype == torch.long
        assert self.masks.dtype == torch.bool
        assert self.choice_counts.dtype == torch.long
        device = self.choice_counts.device
        assert self.token_ids.device == device
        assert self.masks.device == device

    def batch_size(self) -> int:
        """Return the number of rows represented by this batch."""
        return int(self.choice_counts.shape[0])


def legal_candidate_batch(
    *, candidate_token_ids: Tensor, candidate_mask: Tensor
) -> DeviceLegalCandidateBatch:
    """Return fixed-width legal candidate tensors."""
    assert candidate_token_ids.ndim == 2
    assert candidate_mask.shape == candidate_token_ids.shape
    assert candidate_token_ids.dtype == torch.long
    assert candidate_mask.dtype == torch.bool
    return DeviceLegalCandidateBatch(
        token_ids=torch.where(
            candidate_mask,
            candidate_token_ids,
            torch.zeros_like(candidate_token_ids),
        ),
        masks=candidate_mask,
        choice_counts=candidate_mask.sum(dim=1).to(dtype=torch.long),
    )
