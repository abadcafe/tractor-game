"""Fixed-vocabulary legal choice masks for action plans."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.training.semantic_actions.choices import ACTION_CHOICE_COUNT


@dataclass(frozen=True, slots=True)
class DeviceLegalChoiceBatch:
    """Legal mask over the complete 110-choice vocabulary."""

    masks: Tensor
    choice_counts: Tensor

    def __post_init__(self) -> None:
        assert self.masks.ndim == 2
        assert int(self.masks.shape[1]) == ACTION_CHOICE_COUNT
        assert self.masks.dtype == torch.bool
        assert self.choice_counts.shape == (int(self.masks.shape[0]),)
        assert self.choice_counts.dtype == torch.long
        assert self.choice_counts.device == self.masks.device

    def batch_size(self) -> int:
        """Return represented plan count."""
        return int(self.masks.shape[0])


def legal_choice_batch(*, masks: Tensor) -> DeviceLegalChoiceBatch:
    """Build a legal choice batch from a full-vocabulary mask."""
    assert masks.ndim == 2
    assert int(masks.shape[1]) == ACTION_CHOICE_COUNT
    assert masks.dtype == torch.bool
    return DeviceLegalChoiceBatch(
        masks=masks,
        choice_counts=masks.sum(dim=1).to(dtype=torch.long),
    )


__all__ = ("DeviceLegalChoiceBatch", "legal_choice_batch")
