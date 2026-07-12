"""Distributed scalar synchronization for PPO update control."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor

from server.foundation import result as _result
from server.training.ppo.collectives import all_reduce_sum
from server.training.ppo.distributed import PPOUpdatePartition


def synchronized_count_sum(
    *,
    value: int,
    partition: PPOUpdatePartition,
    device: torch.device,
) -> _result.Ok[Tensor] | _result.Rejected:
    """Return a device scalar containing the global count sum."""
    assert value >= 0
    tensor = torch.tensor(value, dtype=torch.long, device=device)
    if partition.world_size == 1:
        return _result.Ok(value=tensor)
    if not dist.is_initialized():
        return _result.Rejected(
            reason="distributed PPO count sync requires process group"
        )
    return _result.Ok(value=all_reduce_sum(tensor))


def synchronized_count_vector_sum(
    *,
    values: Tensor,
    partition: PPOUpdatePartition,
    device: torch.device,
) -> _result.Ok[Tensor] | _result.Rejected:
    """Return a device vector containing global count sums."""
    assert values.ndim == 1
    assert values.dtype == torch.long
    values = values.to(device=device)
    if partition.world_size == 1:
        return _result.Ok(value=values)
    if not dist.is_initialized():
        return _result.Rejected(
            reason="distributed PPO count sync requires process group"
        )
    return _result.Ok(value=all_reduce_sum(values))


def positive_count_value(
    count: Tensor,
) -> _result.Ok[int] | _result.Rejected:
    """Return a positive CPU count at coarse update boundaries."""
    assert count.shape == ()
    value = int(count.detach().cpu().item())
    if value <= 0:
        return _result.Rejected(
            reason="synchronized PPO update requires rollout decisions"
        )
    return _result.Ok(value=value)
