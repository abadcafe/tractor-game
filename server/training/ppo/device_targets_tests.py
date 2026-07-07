"""Tests for device-side PPO helper tensors."""

from __future__ import annotations

import torch

from server.training.ppo.device_targets import shuffled_index_tensor


def test_shuffled_index_tensor_creates_long_tensor_on_device() -> None:
    device = torch.device("cpu")

    result = shuffled_index_tensor(indices=(2, 0, 1), device=device)

    assert result.dtype == torch.long
    assert result.device == device
    assert bool(
        torch.equal(
            result,
            torch.tensor((2, 0, 1), dtype=torch.long, device=device),
        )
    )
