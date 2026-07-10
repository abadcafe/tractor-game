"""Model-rank read operations for shared rollout arenas."""

from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import Literal

import torch
from torch import Tensor

from server import result as _result
from server.result import Ok, Rejected
from server.training.policy_sampling import RankReturnTargets
from server.training.runtime.shared_rollout_arena.schema import (
    sample_reference_column_views,
    unpack_header,
)
from server.training.runtime.shared_rollout_arena.types import (
    RolloutArenaHandle,
)


@dataclass(slots=True)
class SharedRolloutArenaReader:
    """Read committed rollout handles for one model rank."""

    handles: tuple[RolloutArenaHandle, ...]
    _segments: tuple[shared_memory.SharedMemory, ...]
    _workspace: "_RankReturnTransferWorkspace" = field(
        default_factory=lambda: _RankReturnTransferWorkspace()
    )

    def read_rank_batch(
        self,
        *,
        policy_version: int,
        model_rank_index: int,
        device: torch.device,
    ) -> _result.Ok[RankReturnTargets] | _result.Rejected:
        """Return every row from this rank's assigned worker arenas."""
        assert policy_version >= 0
        assert model_rank_index >= 0
        row_indices: list[_ColumnBlock] = []
        step_counts: list[_ColumnBlock] = []
        return_values: list[_ColumnBlock] = []
        round_count = 0
        total_step_count = 0
        max_step_count = 0
        for handle, segment in zip(
            self.handles, self._segments, strict=True
        ):
            handle.lock.acquire()
            try:
                buffer = _segment_buffer(segment)
                header = unpack_header(buffer)
                if header.policy_version != policy_version:
                    return Rejected(
                        reason="rollout arena policy version mismatch"
                    )
                round_count += header.round_count
                total_step_count += header.total_step_count
                max_step_count = max(
                    max_step_count, header.max_step_count
                )
                columns = sample_reference_column_views(
                    buffer=buffer,
                    capacity=header.capacity,
                    count=header.sample_count,
                )
                row_indices.append(
                    _ColumnBlock(
                        values=columns.row_indices,
                        count=header.sample_count,
                    )
                )
                step_counts.append(
                    _ColumnBlock(
                        values=columns.step_counts,
                        count=header.sample_count,
                    )
                )
                return_values.append(
                    _ColumnBlock(
                        values=columns.return_values,
                        count=header.sample_count,
                    )
                )
            finally:
                handle.lock.release()
        return Ok(
            value=RankReturnTargets(
                policy_version=policy_version,
                model_rank_index=model_rank_index,
                row_indices=self._workspace.materialize_column(
                    name="row_indices",
                    blocks=tuple(row_indices),
                    dtype=torch.long,
                    device=device,
                ),
                step_counts=self._workspace.materialize_column(
                    name="step_counts",
                    blocks=tuple(step_counts),
                    dtype=torch.long,
                    device=device,
                ),
                return_values=self._workspace.materialize_column(
                    name="return_values",
                    blocks=tuple(return_values),
                    dtype=torch.float32,
                    device=device,
                ),
                round_count=round_count,
                total_step_count=total_step_count,
                max_step_count=max_step_count,
            )
        )

    def close(self) -> None:
        """Detach this process from all shared memory segments."""
        for segment in self._segments:
            segment.close()


def attach_rollout_arena_reader(
    handles: tuple[RolloutArenaHandle, ...],
) -> SharedRolloutArenaReader:
    """Attach a compute rank to its assigned worker arenas."""
    return SharedRolloutArenaReader(
        handles=handles,
        _segments=tuple(
            shared_memory.SharedMemory(name=handle.shared_memory_name)
            for handle in handles
        ),
    )


def _segment_buffer(
    segment: shared_memory.SharedMemory,
) -> memoryview[int]:
    buffer = segment.buf
    assert buffer is not None
    return buffer


@dataclass(frozen=True, slots=True)
class _ColumnBlock:
    values: memoryview[int]
    count: int

    def __post_init__(self) -> None:
        assert self.count >= 0


type _ReturnColumnName = Literal[
    "row_indices",
    "step_counts",
    "return_values",
]


@dataclass(slots=True)
class _TransferColumnWorkspace:
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
        if device.type == "cuda":
            host = self._host_buffer(
                count=count, dtype=dtype, pin_memory=True
            )
            _copy_blocks_to_host(
                blocks=blocks, dtype=dtype, target=host
            )
            device_tensor = self._device_buffer(
                count=count, dtype=dtype, device=device
            )
            device_tensor[:count].copy_(host[:count], non_blocking=True)
            return device_tensor[:count]
        if device.type == "cpu":
            host = self._host_buffer(
                count=count, dtype=dtype, pin_memory=False
            )
            _copy_blocks_to_host(
                blocks=blocks, dtype=dtype, target=host
            )
            return host[:count]
        host = self._host_buffer(
            count=count, dtype=dtype, pin_memory=False
        )
        _copy_blocks_to_host(blocks=blocks, dtype=dtype, target=host)
        return host[:count].to(device=device)

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


@dataclass(slots=True)
class _RankReturnTransferWorkspace:
    row_indices: _TransferColumnWorkspace = field(
        default_factory=_TransferColumnWorkspace
    )
    step_counts: _TransferColumnWorkspace = field(
        default_factory=_TransferColumnWorkspace
    )
    return_values: _TransferColumnWorkspace = field(
        default_factory=_TransferColumnWorkspace
    )

    def materialize_column(
        self,
        *,
        name: _ReturnColumnName,
        blocks: tuple[_ColumnBlock, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        workspace = self._column(name)
        return workspace.materialize(
            blocks=blocks,
            dtype=dtype,
            device=device,
        )

    def _column(
        self, name: _ReturnColumnName
    ) -> _TransferColumnWorkspace:
        if name == "row_indices":
            return self.row_indices
        if name == "step_counts":
            return self.step_counts
        if name == "return_values":
            return self.return_values
        raise AssertionError(name)


def _copy_blocks_to_host(
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
            block.values, dtype=dtype, count=block.count
        )
        target[offset : offset + block.count].copy_(source)
        offset += block.count
