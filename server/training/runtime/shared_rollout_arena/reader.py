"""Model-rank reads of atomic rollout trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import Literal, assert_never

import torch
from torch import Tensor

from server.foundation import result as _result
from server.foundation.result import Ok, Rejected
from server.training.rollout_inference.samples import (
    RankTrajectoryBatch,
)
from server.training.runtime.shared_rollout_arena.schema import (
    rollout_column_views,
    unpack_header,
)
from server.training.runtime.shared_rollout_arena.types import (
    RolloutArenaHandle,
)


@dataclass(slots=True)
class SharedRolloutArenaReader:
    """Materialize rank-local completed trajectories on demand."""

    handles: tuple[RolloutArenaHandle, ...]
    _segments: tuple[shared_memory.SharedMemory, ...]
    _workspace: _RankTrajectoryTransferWorkspace = field(
        default_factory=lambda: _RankTrajectoryTransferWorkspace()
    )

    def read_rank_batch(
        self,
        *,
        policy_version: int,
        model_rank_index: int,
        device: torch.device,
    ) -> _result.Ok[RankTrajectoryBatch] | _result.Rejected:
        """Return every committed trajectory assigned to this rank."""
        row_indices: list[_ColumnBlock] = []
        step_counts: list[_ColumnBlock] = []
        trajectory_offsets: list[_ColumnBlock] = []
        trajectory_lengths: list[_ColumnBlock] = []
        terminal_rewards: list[_ColumnBlock] = []
        round_count = 0
        total_step_count = 0
        max_step_count = 0
        decision_base = 0
        for handle, segment in zip(
            self.handles, self._segments, strict=True
        ):
            _ = handle.lock.acquire()
            try:
                buffer = _segment_buffer(segment)
                header = unpack_header(buffer)
                if header.policy_version != policy_version:
                    return Rejected(
                        reason="rollout arena policy version mismatch"
                    )
                columns = rollout_column_views(
                    buffer=buffer,
                    header=header,
                )
                row_indices.append(
                    _ColumnBlock(
                        columns.row_indices, header.decision_count
                    )
                )
                step_counts.append(
                    _ColumnBlock(
                        columns.step_counts, header.decision_count
                    )
                )
                trajectory_offsets.append(
                    _ColumnBlock(
                        columns.trajectory_offsets,
                        header.trajectory_count,
                        addend=decision_base,
                    )
                )
                trajectory_lengths.append(
                    _ColumnBlock(
                        columns.trajectory_lengths,
                        header.trajectory_count,
                    )
                )
                terminal_rewards.append(
                    _ColumnBlock(
                        columns.terminal_rewards,
                        header.trajectory_count,
                    )
                )
                decision_base += header.decision_count
                round_count += header.round_count
                total_step_count += header.total_trace_step_count
                max_step_count = max(
                    max_step_count,
                    header.max_trace_step_count,
                )
            finally:
                handle.lock.release()
        return Ok(
            RankTrajectoryBatch(
                policy_version=policy_version,
                model_rank_index=model_rank_index,
                row_indices=self._workspace.materialize(
                    name="row_indices",
                    blocks=tuple(row_indices),
                    dtype=torch.long,
                    device=device,
                ),
                step_counts=self._workspace.materialize(
                    name="step_counts",
                    blocks=tuple(step_counts),
                    dtype=torch.long,
                    device=device,
                ),
                trajectory_offsets=self._workspace.materialize(
                    name="trajectory_offsets",
                    blocks=tuple(trajectory_offsets),
                    dtype=torch.long,
                    device=device,
                ),
                trajectory_lengths=self._workspace.materialize(
                    name="trajectory_lengths",
                    blocks=tuple(trajectory_lengths),
                    dtype=torch.long,
                    device=device,
                ),
                terminal_rewards=self._workspace.materialize(
                    name="terminal_rewards",
                    blocks=tuple(terminal_rewards),
                    dtype=torch.float32,
                    device=device,
                ),
                round_count=round_count,
                total_step_count=total_step_count,
                max_step_count=max_step_count,
            )
        )

    def close(self) -> None:
        for segment in self._segments:
            segment.close()


def attach_rollout_arena_reader(
    handles: tuple[RolloutArenaHandle, ...],
) -> SharedRolloutArenaReader:
    return SharedRolloutArenaReader(
        handles=handles,
        _segments=tuple(
            shared_memory.SharedMemory(name=handle.shared_memory_name)
            for handle in handles
        ),
    )


@dataclass(frozen=True, slots=True)
class _ColumnBlock:
    values: memoryview[int]
    count: int
    addend: int = 0


@dataclass(slots=True)
class _TransferWorkspace:
    host: Tensor | None = None
    device_tensor: Tensor | None = None

    def materialize(
        self,
        *,
        blocks: tuple[_ColumnBlock, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        count = sum(block.count for block in blocks)
        if count == 0:
            return torch.empty((0,), dtype=dtype, device=device)
        host = self._host_buffer(
            count=count,
            dtype=dtype,
            pin_memory=device.type == "cuda",
        )
        _copy_blocks(blocks=blocks, dtype=dtype, target=host)
        if device.type == "cpu":
            return host[:count]
        target = self._device_buffer(
            count=count,
            dtype=dtype,
            device=device,
        )
        _ = target[:count].copy_(
            host[:count],
            non_blocking=device.type == "cuda",
        )
        return target[:count]

    def _host_buffer(
        self, *, count: int, dtype: torch.dtype, pin_memory: bool
    ) -> Tensor:
        current = self.host
        if (
            current is None
            or int(current.shape[0]) < count
            or current.dtype != dtype
            or current.is_pinned() != pin_memory
        ):
            current = torch.empty(
                (count,), dtype=dtype, pin_memory=pin_memory
            )
            self.host = current
        return current

    def _device_buffer(
        self, *, count: int, dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        current = self.device_tensor
        if (
            current is None
            or int(current.shape[0]) < count
            or current.dtype != dtype
            or current.device != device
        ):
            current = torch.empty((count,), dtype=dtype, device=device)
            self.device_tensor = current
        return current


type _ColumnName = Literal[
    "row_indices",
    "step_counts",
    "trajectory_offsets",
    "trajectory_lengths",
    "terminal_rewards",
]


@dataclass(slots=True)
class _RankTrajectoryTransferWorkspace:
    row_indices: _TransferWorkspace = field(
        default_factory=_TransferWorkspace
    )
    step_counts: _TransferWorkspace = field(
        default_factory=_TransferWorkspace
    )
    trajectory_offsets: _TransferWorkspace = field(
        default_factory=_TransferWorkspace
    )
    trajectory_lengths: _TransferWorkspace = field(
        default_factory=_TransferWorkspace
    )
    terminal_rewards: _TransferWorkspace = field(
        default_factory=_TransferWorkspace
    )

    def materialize(
        self,
        *,
        name: _ColumnName,
        blocks: tuple[_ColumnBlock, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        return self._column(name).materialize(
            blocks=blocks,
            dtype=dtype,
            device=device,
        )

    def _column(self, name: _ColumnName) -> _TransferWorkspace:
        match name:
            case "row_indices":
                return self.row_indices
            case "step_counts":
                return self.step_counts
            case "trajectory_offsets":
                return self.trajectory_offsets
            case "trajectory_lengths":
                return self.trajectory_lengths
            case "terminal_rewards":
                return self.terminal_rewards
        assert_never(name)


def _copy_blocks(
    *,
    blocks: tuple[_ColumnBlock, ...],
    dtype: torch.dtype,
    target: Tensor,
) -> None:
    offset = 0
    for block in blocks:
        if block.count == 0:
            continue
        source = torch.frombuffer(
            block.values,
            dtype=dtype,
            count=block.count,
        )
        selected = target[offset : offset + block.count]
        _ = selected.copy_(source)
        if block.addend != 0:
            _ = selected.add_(block.addend)
        offset += block.count


def _segment_buffer(
    segment: shared_memory.SharedMemory,
) -> memoryview[int]:
    buffer = segment.buf
    assert buffer is not None
    return buffer


__all__ = ("SharedRolloutArenaReader", "attach_rollout_arena_reader")
