"""Model-rank read operations for shared rollout arenas."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory

from server import result as _result
from server.result import Ok, Rejected
from server.training.policy_sampling import DecisionHandle
from server.training.returns import ReturnCommit
from server.training.runtime.shared_rollout_arena.schema import (
    unpack_header,
    unpack_row,
)
from server.training.runtime.shared_rollout_arena.types import (
    RolloutArenaHandle,
)


@dataclass(slots=True)
class SharedRolloutArenaReader:
    """Read committed rollout handles for one model rank."""

    handles: tuple[RolloutArenaHandle, ...]
    _segments: tuple[shared_memory.SharedMemory, ...]

    def read_commit_for_rank(
        self,
        *,
        policy_version: int,
        model_rank_index: int,
    ) -> _result.Ok[ReturnCommit] | _result.Rejected:
        """Return rows owned by one model rank across all arenas."""
        assert policy_version >= 0
        assert model_rank_index >= 0
        decision_handles: list[DecisionHandle] = []
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
                    row = unpack_row(buffer, index=row_index)
                    if row.model_rank_index != model_rank_index:
                        continue
                    decision_handles.append(
                        DecisionHandle(
                            model_rank_index=row.model_rank_index,
                            policy_version=policy_version,
                            slot_index=row.slot_index,
                            slot_generation=row.slot_generation,
                        )
                    )
                    return_values.append(row.return_value)
            finally:
                handle.condition.release()
        return Ok(
            value=ReturnCommit(
                policy_version=policy_version,
                first_episode_id=0,
                episode_count=round_count,
                decision_handles=tuple(decision_handles),
                return_values=tuple(return_values),
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
