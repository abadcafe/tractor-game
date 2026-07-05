"""PPO distributed update contracts and DDP loss wrapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from server import result as _result
from server.training.ppo.loss_module import (
    PPOLossForwardOutput,
    PPOLossModule,
)
from server.training.ppo.minibatch import TensorizedPPOMinibatch
from server.training.ppo.profile import PPOProfileAccumulator


@dataclass(frozen=True, slots=True)
class PPOUpdatePartition:
    """Rank-local view of one synchronized PPO update."""

    rank: int
    world_size: int

    def __post_init__(self) -> None:
        assert self.rank >= 0
        assert self.world_size > 0
        assert self.rank < self.world_size


class PPOLossForwarder(Protocol):
    """Callable train-time loss module boundary."""

    def __call__(
        self,
        minibatch: TensorizedPPOMinibatch,
        profile: PPOProfileAccumulator,
    ) -> PPOLossForwardOutput: ...

    def train(self, mode: bool = True) -> nn.Module: ...

    def zero_grad(self, set_to_none: bool = True) -> None: ...


def single_update_partition() -> PPOUpdatePartition:
    """Return the single-rank update partition."""
    return PPOUpdatePartition(rank=0, world_size=1)


def build_ppo_loss_forwarder(
    *,
    module: PPOLossModule,
    partition: PPOUpdatePartition,
    device: torch.device,
) -> _result.Ok[PPOLossForwarder] | _result.Rejected:
    """Wrap a PPO loss module in DDP for multi-rank updates."""
    if partition.world_size == 1:
        return _result.Ok(value=cast(PPOLossForwarder, module))
    if not dist.is_initialized():
        return _result.Rejected(
            reason="DDP PPO update requires initialized process group"
        )
    if device.type == "cuda":
        device_index = _cuda_device_index(device)
        torch.cuda.set_device(device_index)
        try:
            ddp = DistributedDataParallel(
                module,
                device_ids=[device_index],
                output_device=device_index,
            )
        except RuntimeError as exc:
            return _result.Rejected(
                reason=f"DDP PPO loss wrapper failed: {exc}"
            )
        return _result.Ok(value=cast(PPOLossForwarder, ddp))
    try:
        ddp = DistributedDataParallel(module)
    except RuntimeError as exc:
        return _result.Rejected(
            reason=f"DDP PPO loss wrapper failed: {exc}"
        )
    return _result.Ok(value=cast(PPOLossForwarder, ddp))


def _cuda_device_index(device: torch.device) -> int:
    assert device.type == "cuda"
    return device.index
