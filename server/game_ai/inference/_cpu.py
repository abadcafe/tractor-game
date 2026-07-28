"""Inference-thread placement for heterogeneous CPU hosts."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True, slots=True)
class CPUExecutionPlan:
    """Highest-capacity CPUs available to the current process."""

    thread_count: int
    cpu_ids: tuple[int, ...] | None

    def __post_init__(self) -> None:
        assert self.thread_count > 0
        if self.cpu_ids is not None:
            assert self.cpu_ids
            assert self.thread_count == len(self.cpu_ids)
            assert len(set(self.cpu_ids)) == len(self.cpu_ids)
            assert all(cpu_id >= 0 for cpu_id in self.cpu_ids)


def cpu_execution_plan() -> CPUExecutionPlan:
    """Select one homogeneous performance cluster when discoverable."""
    if sys.platform != "linux":
        return CPUExecutionPlan(
            thread_count=os.cpu_count() or 1,
            cpu_ids=None,
        )
    allowed = tuple(sorted(os.sched_getaffinity(0)))
    assert allowed
    capacities = _cpu_capacities(allowed)
    if capacities is None:
        return CPUExecutionPlan(
            thread_count=len(allowed),
            cpu_ids=allowed,
        )
    maximum = max(capacity for _cpu_id, capacity in capacities)
    cpu_ids = tuple(
        cpu_id for cpu_id, capacity in capacities if capacity == maximum
    )
    return CPUExecutionPlan(
        thread_count=len(cpu_ids),
        cpu_ids=cpu_ids,
    )


def initialize_cpu_worker(plan: CPUExecutionPlan) -> None:
    """Pin the model owner before PyTorch creates intra-op workers."""
    if plan.cpu_ids is not None:
        os.sched_setaffinity(0, set(plan.cpu_ids))
    torch.set_num_threads(plan.thread_count)


def _cpu_capacities(
    cpu_ids: tuple[int, ...],
) -> tuple[tuple[int, int], ...] | None:
    paths = tuple(
        Path(f"/sys/devices/system/cpu/cpu{cpu_id}/cpu_capacity")
        for cpu_id in cpu_ids
    )
    if not all(path.is_file() for path in paths):
        return None
    return tuple(
        (cpu_id, int(path.read_text().strip()))
        for cpu_id, path in zip(cpu_ids, paths, strict=True)
    )


__all__ = (
    "CPUExecutionPlan",
    "cpu_execution_plan",
    "initialize_cpu_worker",
)
