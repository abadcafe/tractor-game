"""Torch distributed process group runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

import torch.distributed as dist

from server import result as _result

type DistributedBackend = Literal["gloo", "nccl"]


@dataclass(frozen=True, slots=True)
class DistributedRankConfig:
    """One rank in a synchronized update process group."""

    backend: DistributedBackend
    init_method: str
    rank: int
    world_size: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        assert self.backend in ("gloo", "nccl")
        assert self.init_method
        assert self.rank >= 0
        assert self.world_size > 1
        assert self.rank < self.world_size
        assert self.timeout_seconds > 0.0


def initialize_distributed_rank(
    config: DistributedRankConfig | None,
) -> _result.Ok[None] | _result.Rejected:
    """Initialize a torch distributed rank for multi-rank updates."""
    if config is None:
        return _result.Ok(value=None)
    if not dist.is_available():
        return _result.Rejected(
            reason="torch distributed is unavailable in this runtime"
        )
    if config.backend == "nccl" and not dist.is_nccl_available():
        return _result.Rejected(
            reason="torch distributed NCCL backend is unavailable"
        )
    if config.backend == "gloo" and not dist.is_gloo_available():
        return _result.Rejected(
            reason="torch distributed Gloo backend is unavailable"
        )
    if dist.is_initialized():
        return _result.Rejected(
            reason=(
                "torch distributed process group is already initialized"
            )
        )
    try:
        dist.init_process_group(
            backend=config.backend,
            init_method=config.init_method,
            rank=config.rank,
            world_size=config.world_size,
            timeout=timedelta(seconds=config.timeout_seconds),
        )
    except RuntimeError as exc:
        return _result.Rejected(
            reason=f"torch distributed init failed: {exc}"
        )
    return _result.Ok(value=None)


def destroy_distributed_rank() -> None:
    """Destroy the current process group if this process owns one."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
