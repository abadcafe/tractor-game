"""Model-rank read operations for shared rollout arenas."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory

import torch

from server import result as _result
from server.result import Ok, Rejected
from server.training.policy_sampling import RankReturnBatch
from server.training.runtime.shared_rollout_arena.schema import (
    unpack_header,
    unpack_model_rank_index,
    unpack_return_value,
    unpack_slot_generation,
    unpack_slot_index,
)
from server.training.runtime.shared_rollout_arena.types import (
    RolloutArenaHandle,
)


@dataclass(slots=True)
class SharedRolloutArenaReader:
    """Read committed rollout handles for one model rank."""

    handles: tuple[RolloutArenaHandle, ...]
    _segments: tuple[shared_memory.SharedMemory, ...]

    def read_rank_batch(
        self,
        *,
        policy_version: int,
        model_rank_index: int,
    ) -> _result.Ok[RankReturnBatch] | _result.Rejected:
        """Return rows owned by one model rank across all arenas."""
        assert policy_version >= 0
        assert model_rank_index >= 0
        slot_indices: list[int] = []
        slot_generations: list[int] = []
        return_values: list[float] = []
        round_count = 0
        for handle, segment in zip(
            self.handles, self._segments, strict=True
        ):
            handle.condition.acquire()
            try:
                buffer = _segment_buffer(segment)
                header = unpack_header(buffer)
                if header.policy_version != policy_version:
                    return Rejected(
                        reason="rollout arena policy version mismatch"
                    )
                if not header.full:
                    return Rejected(
                        reason="rollout arena is not full for update"
                    )
                round_count += header.round_count
                for row_index in range(header.sample_count):
                    row_model_rank = unpack_model_rank_index(
                        buffer=buffer, index=row_index
                    )
                    if row_model_rank != model_rank_index:
                        continue
                    slot_indices.append(
                        unpack_slot_index(
                            buffer=buffer,
                            capacity=header.capacity,
                            index=row_index,
                        )
                    )
                    slot_generations.append(
                        unpack_slot_generation(
                            buffer=buffer,
                            capacity=header.capacity,
                            index=row_index,
                        )
                    )
                    return_values.append(
                        unpack_return_value(
                            buffer=buffer,
                            capacity=header.capacity,
                            index=row_index,
                        )
                    )
            finally:
                handle.condition.release()
        return Ok(
            value=RankReturnBatch(
                policy_version=policy_version,
                model_rank_index=model_rank_index,
                slot_indices=torch.tensor(
                    slot_indices, dtype=torch.long
                ),
                slot_generations=torch.tensor(
                    slot_generations, dtype=torch.long
                ),
                return_values=torch.tensor(
                    return_values, dtype=torch.float32
                ),
                round_count=round_count,
            )
        )

    def close(self) -> None:
        """Detach this process from all shared memory segments."""
        for segment in self._segments:
            segment.close()


def attach_rollout_arena_reader(
    handles: tuple[RolloutArenaHandle, ...],
) -> SharedRolloutArenaReader:
    """Attach a compute rank to every worker arena."""
    assert handles
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
