"""Functional tensor reductions for distributed PPO control data."""

from __future__ import annotations

from typing import Protocol, cast

import torch.distributed as dist
from torch import Tensor


class _AllReduceInPlace(Protocol):
    def __call__(self, tensor: Tensor, op: object) -> None: ...


_all_reduce_object: object = getattr(dist, "all_reduce")
_all_reduce_in_place = cast(_AllReduceInPlace, _all_reduce_object)


def all_reduce_sum(tensor: Tensor) -> Tensor:
    """Return the global sum without modifying the input tensor."""
    result = tensor.clone()
    _all_reduce_in_place(result, dist.ReduceOp.SUM)
    return result


def all_reduce_max(tensor: Tensor) -> Tensor:
    """Return the global maximum without modifying the input tensor."""
    result = tensor.clone()
    _all_reduce_in_place(result, dist.ReduceOp.MAX)
    return result
