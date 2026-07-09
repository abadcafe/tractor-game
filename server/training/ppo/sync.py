"""Distributed scalar synchronization for PPO update control."""

from __future__ import annotations

from typing import Protocol, cast

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_functional
from torch import Tensor

from server import result as _result
from server.training.ppo.distributed import PPOUpdatePartition


class _AllReduceTensor(Protocol):
    def __call__(self, tensor: Tensor, op: object) -> Tensor: ...


_all_reduce_object: object = getattr(dist_functional, "all_reduce")
_all_reduce_tensor = cast(_AllReduceTensor, _all_reduce_object)


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
    return _result.Ok(
        value=_all_reduce_tensor(tensor, dist.ReduceOp.SUM)
    )


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
    return _result.Ok(
        value=_all_reduce_tensor(values, dist.ReduceOp.SUM)
    )


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
