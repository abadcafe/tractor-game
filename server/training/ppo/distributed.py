"""PPO distributed update contracts and DDP loss wrapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from server import result as _result
from server.training.ppo.loss_module import (
    PPOLossForwardOutput,
    PPOLossForwardTensors,
    PPOLossModule,
    loss_forward_output_from_tensors,
)
from server.training.ppo.minibatch import TensorizedPPOMinibatch
from server.training.ppo.profile import PPOProfileAccumulator


class _AllReduceInPlace(Protocol):
    def __call__(self, tensor: Tensor, op: object) -> object: ...


_all_reduce_object: object = getattr(dist, "all_reduce")
_all_reduce_in_place = cast(_AllReduceInPlace, _all_reduce_object)


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


class _TensorLossForwarder(Protocol):
    def __call__(
        self,
        minibatch: TensorizedPPOMinibatch,
        profile: PPOProfileAccumulator,
    ) -> PPOLossForwardTensors: ...

    def train(self, mode: bool = True) -> nn.Module: ...

    def zero_grad(self, set_to_none: bool = True) -> None: ...


@dataclass(slots=True)
class _PPOLossForwarderAdapter:
    module: _TensorLossForwarder
    partition: PPOUpdatePartition
    device: torch.device

    def __call__(
        self,
        minibatch: TensorizedPPOMinibatch,
        profile: PPOProfileAccumulator,
    ) -> PPOLossForwardOutput:
        tensors = self.module(minibatch, profile)
        rejection_flag = tensors[6]
        if self.partition.world_size > 1:
            if not dist.is_initialized():
                return PPOLossForwardOutput(
                    loss=None,
                    rejection_reason=(
                        "distributed PPO loss sync requires "
                        "process group"
                    ),
                )
            rejection_flag = rejection_flag.detach().clone()
            _all_reduce_in_place(rejection_flag, dist.ReduceOp.MAX)
        rejected = bool(rejection_flag.detach().cpu().item() > 0.5)
        return loss_forward_output_from_tensors(
            tensors,
            rejected=rejected,
        )

    def train(self, mode: bool = True) -> nn.Module:
        return self.module.train(mode)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.module.zero_grad(set_to_none=set_to_none)


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
        return _result.Ok(
            value=_PPOLossForwarderAdapter(
                module=cast(_TensorLossForwarder, module),
                partition=partition,
                device=device,
            )
        )
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
        return _result.Ok(
            value=_PPOLossForwarderAdapter(
                module=cast(_TensorLossForwarder, ddp),
                partition=partition,
                device=device,
            )
        )
    try:
        ddp = DistributedDataParallel(module)
    except RuntimeError as exc:
        return _result.Rejected(
            reason=f"DDP PPO loss wrapper failed: {exc}"
        )
    return _result.Ok(
        value=_PPOLossForwarderAdapter(
            module=cast(_TensorLossForwarder, ddp),
            partition=partition,
            device=device,
        )
    )


def _cuda_device_index(device: torch.device) -> int:
    assert device.type == "cuda"
    return device.index
