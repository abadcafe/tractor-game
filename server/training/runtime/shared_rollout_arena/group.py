"""Create, reset, wait for, and close shared rollout arenas."""

from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.context import SpawnContext

from server import result as _result
from server.result import Ok, Rejected
from server.training.runtime.shared_rollout_arena.schema import (
    RolloutArenaHeader,
    arena_byte_size,
    empty_header,
    pack_header,
    unpack_header,
)
from server.training.runtime.shared_rollout_arena.types import (
    RolloutArenaHandle,
    RolloutArenaSnapshot,
)


@dataclass(frozen=True, slots=True)
class SharedRolloutArenaGroup:
    """Coordinator-owned lifetime handle for rollout arenas."""

    handles: tuple[RolloutArenaHandle, ...]
    segments: tuple[shared_memory.SharedMemory, ...]

    def __post_init__(self) -> None:
        assert self.handles
        assert len(self.handles) == len(self.segments)


def create_shared_rollout_arena_group(
    *,
    context: SpawnContext,
    worker_count: int,
    samples_per_worker_update: int,
    slack_sample_count: int,
    policy_version: int = 0,
) -> _result.Ok[SharedRolloutArenaGroup] | _result.Rejected:
    """Create one fixed-capacity rollout arena per worker."""
    assert worker_count > 0
    assert samples_per_worker_update > 0
    assert slack_sample_count >= 0
    assert policy_version >= 0
    capacity = samples_per_worker_update + slack_sample_count
    handles: list[RolloutArenaHandle] = []
    segments: list[shared_memory.SharedMemory] = []
    try:
        for worker_index in range(worker_count):
            segment = shared_memory.SharedMemory(
                create=True,
                size=arena_byte_size(capacity=capacity),
            )
            _write_empty_header(
                segment=segment,
                policy_version=policy_version,
                capacity=capacity,
                target_sample_count=samples_per_worker_update,
            )
            condition = context.Condition()
            handles.append(
                RolloutArenaHandle(
                    worker_index=worker_index,
                    shared_memory_name=segment.name,
                    capacity=capacity,
                    target_sample_count=samples_per_worker_update,
                    condition=condition,
                )
            )
            segments.append(segment)
    except OSError as exc:
        _close_created_segments(tuple(segments))
        return Rejected(
            reason=f"shared rollout arena creation failed: {exc}"
        )
    return Ok(
        value=SharedRolloutArenaGroup(
            handles=tuple(handles),
            segments=tuple(segments),
        )
    )


def wait_all_rollout_arenas_full(
    *,
    group: SharedRolloutArenaGroup,
    policy_version: int,
    timeout_seconds: float,
) -> _result.Ok[RolloutArenaSnapshot] | _result.Rejected:
    """Block until every arena is full for one policy version."""
    assert policy_version >= 0
    assert timeout_seconds > 0.0
    deadline = time.monotonic() + timeout_seconds
    for handle in group.handles:
        wait_result = _wait_arena_full(
            handle=handle,
            policy_version=policy_version,
            deadline=deadline,
        )
        if isinstance(wait_result, Rejected):
            return wait_result
    return snapshot_rollout_arenas(
        group=group, policy_version=policy_version
    )


def snapshot_rollout_arenas(
    *,
    group: SharedRolloutArenaGroup,
    policy_version: int,
) -> _result.Ok[RolloutArenaSnapshot] | _result.Rejected:
    """Read aggregate counters from all arenas."""
    assert policy_version >= 0
    snapshot = RolloutArenaSnapshot(
        policy_version=policy_version,
        target_sample_count=0,
        round_count=0,
        sample_count=0,
        generated_action_count=0,
        accepted_action_count=0,
        action_choice_count=0,
        game_over_count=0,
        dropped_sample_count=0,
        cancelled_env_count=0,
        total_step_count=0,
        max_step_count=0,
        team0_reward_sum=0.0,
        team1_reward_sum=0.0,
        elapsed_seconds_max=0.0,
    )
    for handle in group.handles:
        segment = _attach_segment(handle)
        try:
            handle.condition.acquire()
            try:
                header = unpack_header(_segment_buffer(segment))
            finally:
                handle.condition.release()
        finally:
            segment.close()
        if header.policy_version != policy_version:
            return Rejected(
                reason="rollout arena policy version mismatch"
            )
        snapshot = _merge_snapshot(snapshot, header_snapshot(header))
    return Ok(value=snapshot)


def reset_rollout_arenas(
    *,
    group: SharedRolloutArenaGroup,
    policy_version: int,
) -> _result.Ok[None] | _result.Rejected:
    """Reset every arena to accept a new policy version."""
    assert policy_version >= 0
    for handle in group.handles:
        segment = _attach_segment(handle)
        try:
            handle.condition.acquire()
            try:
                _write_empty_header(
                    segment=segment,
                    policy_version=policy_version,
                    capacity=handle.capacity,
                    target_sample_count=handle.target_sample_count,
                )
                handle.condition.notify_all()
            finally:
                handle.condition.release()
        finally:
            segment.close()
    return Ok(value=None)


def close_shared_rollout_arenas(
    group: SharedRolloutArenaGroup,
) -> None:
    """Close and unlink coordinator-owned shared memory segments."""
    for segment in group.segments:
        try:
            segment.close()
        finally:
            segment.unlink()


def header_snapshot(header: RolloutArenaHeader) -> RolloutArenaSnapshot:
    """Convert a header-like object into an aggregate snapshot."""
    return RolloutArenaSnapshot(
        policy_version=header.policy_version,
        target_sample_count=header.target_sample_count,
        round_count=header.round_count,
        sample_count=header.sample_count,
        generated_action_count=header.generated_action_count,
        accepted_action_count=header.accepted_action_count,
        action_choice_count=header.action_choice_count,
        game_over_count=header.game_over_count,
        dropped_sample_count=header.dropped_sample_count,
        cancelled_env_count=header.cancelled_env_count,
        total_step_count=header.total_step_count,
        max_step_count=header.max_step_count,
        team0_reward_sum=header.team0_reward_sum,
        team1_reward_sum=header.team1_reward_sum,
        elapsed_seconds_max=header.elapsed_seconds_max,
    )


def _wait_arena_full(
    *,
    handle: RolloutArenaHandle,
    policy_version: int,
    deadline: float,
) -> _result.Ok[None] | _result.Rejected:
    segment = _attach_segment(handle)
    try:
        handle.condition.acquire()
        try:
            while True:
                header = unpack_header(_segment_buffer(segment))
                if (
                    header.policy_version == policy_version
                    and header.full
                ):
                    return Ok(value=None)
                if header.policy_version != policy_version:
                    return Rejected(
                        reason="rollout arena policy version mismatch"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return Rejected(
                        reason="rollout arena fill timed out"
                    )
                handle.condition.wait(remaining)
        finally:
            handle.condition.release()
    finally:
        segment.close()


def _write_empty_header(
    *,
    segment: shared_memory.SharedMemory,
    policy_version: int,
    capacity: int,
    target_sample_count: int,
) -> None:
    pack_header(
        _segment_buffer(segment),
        header=empty_header(
            policy_version=policy_version,
            capacity=capacity,
            target_sample_count=target_sample_count,
        ),
    )


def _attach_segment(
    handle: RolloutArenaHandle,
) -> shared_memory.SharedMemory:
    return shared_memory.SharedMemory(name=handle.shared_memory_name)


def _segment_buffer(
    segment: shared_memory.SharedMemory,
) -> memoryview[int]:
    buffer = segment.buf
    assert buffer is not None
    return buffer


def _merge_snapshot(
    first: RolloutArenaSnapshot,
    second: RolloutArenaSnapshot,
) -> RolloutArenaSnapshot:
    assert first.policy_version == second.policy_version
    return RolloutArenaSnapshot(
        policy_version=first.policy_version,
        target_sample_count=(
            first.target_sample_count + second.target_sample_count
        ),
        round_count=first.round_count + second.round_count,
        sample_count=first.sample_count + second.sample_count,
        generated_action_count=(
            first.generated_action_count + second.generated_action_count
        ),
        accepted_action_count=(
            first.accepted_action_count + second.accepted_action_count
        ),
        action_choice_count=(
            first.action_choice_count + second.action_choice_count
        ),
        game_over_count=first.game_over_count + second.game_over_count,
        dropped_sample_count=(
            first.dropped_sample_count + second.dropped_sample_count
        ),
        cancelled_env_count=(
            first.cancelled_env_count + second.cancelled_env_count
        ),
        total_step_count=(
            first.total_step_count + second.total_step_count
        ),
        max_step_count=max(first.max_step_count, second.max_step_count),
        team0_reward_sum=(
            first.team0_reward_sum + second.team0_reward_sum
        ),
        team1_reward_sum=(
            first.team1_reward_sum + second.team1_reward_sum
        ),
        elapsed_seconds_max=max(
            first.elapsed_seconds_max, second.elapsed_seconds_max
        ),
    )


def _close_created_segments(
    segments: tuple[shared_memory.SharedMemory, ...],
) -> None:
    for segment in segments:
        try:
            segment.close()
        finally:
            segment.unlink()
