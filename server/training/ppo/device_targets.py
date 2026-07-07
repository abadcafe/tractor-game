"""Device-side PPO update helpers."""

from __future__ import annotations

import torch
from torch import Tensor


def shuffled_index_tensor(
    *, indices: tuple[int, ...], device: torch.device
) -> Tensor:
    """Create a learner-device tensor for deterministic sample order."""
    assert indices
    return torch.tensor(indices, dtype=torch.long, device=device)
