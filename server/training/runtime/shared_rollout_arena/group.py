"""Create, reset, wait for, and close shared rollout arenas."""

from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.context import SpawnContext

from server.foundation import result as _result
from server.foundation.result import Ok, Rejected
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
from server.training.stop import TrainingStopRequest


@dataclass(frozen=True, slots=True)
class SharedRolloutArenaGroup:
    """Coordinator-owned lifetime handle for rollout arenas."""

    handles: tuple[RolloutArenaHandle, ...]
    segments: tuple[shared_memory.SharedMemory, ...]

    def __post_init__(self) -> None:
        assert self.handles
        assert len(self.handles) == len(self.segments)


@dataclass(frozen=True, slots=True)
class RolloutSampleTargetReached:
    """Sampling reached its configured aggregate sample target."""

    snapshot: RolloutArenaSnapshot


@dataclass(frozen=True, slots=True)
class RolloutStopRequested:
    """Sampling was interrupted by a cooperative stop request."""


type RolloutWaitOutcome = (
    RolloutSampleTargetReached | RolloutStopRequested
)

_STOP_CHECK_INTERVAL_SECONDS = 0.05


def create_shared_rollout_arena_group(
    *,
    context: SpawnContext,
    worker_count: int,
    arena_capacity_per_worker: int,
    policy_version: int = 0,
) -> _result.Ok[SharedRolloutArenaGroup] | _result.Rejected:
    """Create one fixed-capacity rollout arena per worker."""
    assert worker_count > 0
    assert arena_capacity_per_worker > 0
    assert policy_version >= 0
    handles: list[RolloutArenaHandle] = []
    segments: list[shared_memory.SharedMemory] = []
    progress_condition = context.Condition()
    try:
        for worker_index in range(worker_count):
            segment = shared_memory.SharedMemory(
                create=True,
                size=arena_byte_size(
                    capacity=arena_capacity_per_worker
                ),
            )
            _write_empty_header(
                segment=segment,
                policy_version=policy_version,
                capacity=arena_capacity_per_worker,
            )
            lock = context.Condition()
            handles.append(
                RolloutArenaHandle(
                    worker_index=worker_index,
                    shared_memory_name=segment.name,
                    capacity=arena_capacity_per_worker,
                    lock=lock,
                    progress_condition=progress_condition,
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


def wait_rollout_sample_target_or_stop(
    *,
    group: SharedRolloutArenaGroup,
    policy_version: int,
    target_sample_count: int,
    timeout_seconds: float,
    stop_request: TrainingStopRequest,
) -> _result.Ok[RolloutWaitOutcome] | _result.Rejected:
    """Block until aggregate samples reach a target or a stop."""
    assert policy_version >= 0
    assert target_sample_count > 0
    assert timeout_seconds > 0.0
    deadline = time.monotonic() + timeout_seconds
    progress_condition = group.handles[0].progress_condition
    progress_condition.acquire()
    try:
        while True:
            snapshot_result = snapshot_rollout_arenas(
                group=group, policy_version=policy_version
            )
            if isinstance(snapshot_result, Rejected):
                return snapshot_result
            if (
                snapshot_result.value.sample_count
                >= target_sample_count
            ):
                return Ok(
                    value=RolloutSampleTargetReached(
                        snapshot=snapshot_result.value
                    )
                )
            if stop_request.is_requested():
                return Ok(value=RolloutStopRequested())
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return Rejected(
                    reason="rollout sample target timed out"
                )
            progress_condition.wait(
                min(remaining, _STOP_CHECK_INTERVAL_SECONDS)
            )
    finally:
        progress_condition.release()


def snapshot_rollout_arenas(
    *,
    group: SharedRolloutArenaGroup,
    policy_version: int,
) -> _result.Ok[RolloutArenaSnapshot] | _result.Rejected:
    """Read aggregate counters from all arenas."""
    assert policy_version >= 0
    snapshot = RolloutArenaSnapshot(
        policy_version=policy_version,
        capacity=0,
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
    for handle, segment in zip(
        group.handles, group.segments, strict=True
    ):
        handle.lock.acquire()
        try:
            header = unpack_header(_segment_buffer(segment))
        finally:
            handle.lock.release()
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
    progress_condition = group.handles[0].progress_condition
    progress_condition.acquire()
    try:
        for handle, segment in zip(
            group.handles, group.segments, strict=True
        ):
            handle.lock.acquire()
            try:
                _write_empty_header(
                    segment=segment,
                    policy_version=policy_version,
                    capacity=handle.capacity,
                )
            finally:
                handle.lock.release()
        progress_condition.notify_all()
    finally:
        progress_condition.release()
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
        capacity=header.capacity,
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


def _write_empty_header(
    *,
    segment: shared_memory.SharedMemory,
    policy_version: int,
    capacity: int,
) -> None:
    pack_header(
        _segment_buffer(segment),
        header=empty_header(
            policy_version=policy_version,
            capacity=capacity,
        ),
    )


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
        capacity=first.capacity + second.capacity,
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
