"""Compact device legal-choice batches for semantic action plans."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class DeviceLegalChoices:
    """Row-grouped legal semantic token ids on one torch device."""

    token_ids: Tensor
    row_indices: Tensor
    choice_offsets: Tensor
    choice_counts: Tensor

    def __post_init__(self) -> None:
        assert self.token_ids.ndim == 1
        assert self.row_indices.shape == self.token_ids.shape
        assert self.choice_offsets.ndim == 1
        assert self.choice_counts.ndim == 1
        assert int(self.choice_offsets.shape[0]) == (
            int(self.choice_counts.shape[0]) + 1
        )
        assert self.token_ids.dtype == torch.long
        assert self.row_indices.dtype == torch.long
        assert self.choice_offsets.dtype == torch.long
        assert self.choice_counts.dtype == torch.long
        device = self.choice_counts.device
        assert self.token_ids.device == device
        assert self.row_indices.device == device
        assert self.choice_offsets.device == device

    def batch_size(self) -> int:
        """Return the number of rows represented by this batch."""
        return int(self.choice_counts.shape[0])


def compact_legal_choices(
    *, candidate_token_ids: Tensor, candidate_mask: Tensor
) -> DeviceLegalChoices:
    """Pack a padded candidate matrix into row-grouped legal ids."""
    assert candidate_token_ids.ndim == 2
    assert candidate_mask.shape == candidate_token_ids.shape
    assert candidate_token_ids.dtype == torch.long
    assert candidate_mask.dtype == torch.bool
    row_indices, column_indices = torch.nonzero(
        candidate_mask, as_tuple=True
    )
    token_ids = candidate_token_ids[row_indices, column_indices]
    batch_size = int(candidate_token_ids.shape[0])
    choice_counts = torch.bincount(
        row_indices, minlength=batch_size
    ).to(dtype=torch.long, device=candidate_token_ids.device)
    offsets = torch.cat(
        (
            torch.zeros(
                (1,),
                dtype=torch.long,
                device=candidate_token_ids.device,
            ),
            choice_counts.cumsum(dim=0),
        ),
        dim=0,
    )
    return DeviceLegalChoices(
        token_ids=token_ids,
        row_indices=row_indices,
        choice_offsets=offsets,
        choice_counts=choice_counts,
    )
