"""Coordinator ownership and progress waits for rollout arenas."""

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

_STOP_CHECK_INTERVAL_SECONDS = 0.05


@dataclass(frozen=True, slots=True)
class SharedRolloutArenaGroup:
    handles: tuple[RolloutArenaHandle, ...]
    segments: tuple[shared_memory.SharedMemory, ...]

    def __post_init__(self) -> None:
        assert self.handles
        assert len(self.handles) == len(self.segments)


@dataclass(frozen=True, slots=True)
class RolloutRoundTargetReached:
    """Collection reached its complete-round target."""

    snapshot: RolloutArenaSnapshot


@dataclass(frozen=True, slots=True)
class RolloutStopRequested:
    """Collection was interrupted by a cooperative stop."""


type RolloutWaitOutcome = (
    RolloutRoundTargetReached | RolloutStopRequested
)


def create_shared_rollout_arena_group(
    *,
    context: SpawnContext,
    worker_count: int,
    round_capacity_per_worker: int,
    policy_version: int = 0,
) -> _result.Ok[SharedRolloutArenaGroup] | _result.Rejected:
    """Create one fixed complete-round arena per worker."""
    assert worker_count > 0
    assert round_capacity_per_worker > 0
    handles: list[RolloutArenaHandle] = []
    segments: list[shared_memory.SharedMemory] = []
    progress_condition = context.Condition()
    header = empty_header(
        policy_version=policy_version,
        round_capacity=round_capacity_per_worker,
    )
    try:
        for worker_index in range(worker_count):
            segment = shared_memory.SharedMemory(
                create=True,
                size=arena_byte_size(
                    decision_capacity=header.decision_capacity,
                    trajectory_capacity=header.trajectory_capacity,
                ),
            )
            pack_header(_segment_buffer(segment), header=header)
            handles.append(
                RolloutArenaHandle(
                    worker_index=worker_index,
                    shared_memory_name=segment.name,
                    round_capacity=header.round_capacity,
                    decision_capacity=header.decision_capacity,
                    trajectory_capacity=header.trajectory_capacity,
                    lock=context.Condition(),
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
        SharedRolloutArenaGroup(
            handles=tuple(handles),
            segments=tuple(segments),
        )
    )


def wait_rollout_round_target_or_stop(
    *,
    group: SharedRolloutArenaGroup,
    policy_version: int,
    target_round_count: int,
    timeout_seconds: float,
    stop_request: TrainingStopRequest,
    abort_request: TrainingStopRequest,
) -> _result.Ok[RolloutWaitOutcome] | _result.Rejected:
    """Block until complete rounds reach the target or stop is set."""
    assert target_round_count > 0
    deadline = time.monotonic() + timeout_seconds
    condition = group.handles[0].progress_condition
    _ = condition.acquire()
    try:
        while True:
            snapshot = snapshot_rollout_arenas(
                group=group,
                policy_version=policy_version,
            )
            if isinstance(snapshot, Rejected):
                return snapshot
            if snapshot.value.round_count >= target_round_count:
                return Ok(RolloutRoundTargetReached(snapshot.value))
            if stop_request.is_requested():
                return Ok(RolloutStopRequested())
            if abort_request.is_requested():
                return Rejected(reason="rollout wait aborted")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return Rejected(reason="rollout round target timed out")
            _ = condition.wait(
                min(remaining, _STOP_CHECK_INTERVAL_SECONDS)
            )
    finally:
        condition.release()


def snapshot_rollout_arenas(
    *, group: SharedRolloutArenaGroup, policy_version: int
) -> _result.Ok[RolloutArenaSnapshot] | _result.Rejected:
    """Aggregate all accepted complete-round counters."""
    snapshot = _empty_snapshot(policy_version)
    for handle, segment in zip(
        group.handles, group.segments, strict=True
    ):
        _ = handle.lock.acquire()
        try:
            header = unpack_header(_segment_buffer(segment))
        finally:
            handle.lock.release()
        if header.policy_version != policy_version:
            return Rejected(
                reason="rollout arena policy version mismatch"
            )
        snapshot = _merge_snapshot(snapshot, header_snapshot(header))
    return Ok(snapshot)


def reset_rollout_arenas(
    *, group: SharedRolloutArenaGroup, policy_version: int
) -> _result.Ok[None] | _result.Rejected:
    """Reset every arena for a new policy version."""
    condition = group.handles[0].progress_condition
    _ = condition.acquire()
    try:
        for handle, segment in zip(
            group.handles, group.segments, strict=True
        ):
            _ = handle.lock.acquire()
            try:
                pack_header(
                    _segment_buffer(segment),
                    header=empty_header(
                        policy_version=policy_version,
                        round_capacity=handle.round_capacity,
                    ),
                )
            finally:
                handle.lock.release()
        condition.notify_all()
    finally:
        condition.release()
    return Ok(None)


def close_shared_rollout_arenas(
    group: SharedRolloutArenaGroup,
) -> None:
    for segment in group.segments:
        try:
            segment.close()
        finally:
            segment.unlink()


def header_snapshot(header: RolloutArenaHeader) -> RolloutArenaSnapshot:
    return RolloutArenaSnapshot(
        policy_version=header.policy_version,
        round_capacity=header.round_capacity,
        round_count=header.round_count,
        decision_count=header.decision_count,
        trajectory_count=header.trajectory_count,
        model_action_count=header.model_action_count,
        auto_action_count=header.auto_action_count,
        forced_action_count=header.forced_action_count,
        trainable_decision_count=header.trainable_decision_count,
        scored_choice_step_count=header.scored_choice_step_count,
        game_over_count=header.game_over_count,
        rejected_round_count=header.rejected_round_count,
        cancelled_env_count=header.cancelled_env_count,
        total_trace_step_count=header.total_trace_step_count,
        max_trace_step_count=header.max_trace_step_count,
        model_win_count=header.model_win_count,
        auto_win_count=header.auto_win_count,
        model_declarer_round_count=(header.model_declarer_round_count),
        model_reward_sum=header.model_reward_sum,
        auto_reward_sum=header.auto_reward_sum,
        elapsed_seconds_max=header.elapsed_seconds_max,
    )


def _empty_snapshot(policy_version: int) -> RolloutArenaSnapshot:
    return RolloutArenaSnapshot(
        policy_version=policy_version,
        round_capacity=0,
        round_count=0,
        decision_count=0,
        trajectory_count=0,
        model_action_count=0,
        auto_action_count=0,
        forced_action_count=0,
        trainable_decision_count=0,
        scored_choice_step_count=0,
        game_over_count=0,
        rejected_round_count=0,
        cancelled_env_count=0,
        total_trace_step_count=0,
        max_trace_step_count=0,
        model_win_count=0,
        auto_win_count=0,
        model_declarer_round_count=0,
        model_reward_sum=0.0,
        auto_reward_sum=0.0,
        elapsed_seconds_max=0.0,
    )


def _merge_snapshot(
    first: RolloutArenaSnapshot,
    second: RolloutArenaSnapshot,
) -> RolloutArenaSnapshot:
    assert first.policy_version == second.policy_version
    return RolloutArenaSnapshot(
        policy_version=first.policy_version,
        round_capacity=first.round_capacity + second.round_capacity,
        round_count=first.round_count + second.round_count,
        decision_count=first.decision_count + second.decision_count,
        trajectory_count=(
            first.trajectory_count + second.trajectory_count
        ),
        model_action_count=(
            first.model_action_count + second.model_action_count
        ),
        auto_action_count=(
            first.auto_action_count + second.auto_action_count
        ),
        forced_action_count=(
            first.forced_action_count + second.forced_action_count
        ),
        trainable_decision_count=(
            first.trainable_decision_count
            + second.trainable_decision_count
        ),
        scored_choice_step_count=(
            first.scored_choice_step_count
            + second.scored_choice_step_count
        ),
        game_over_count=first.game_over_count + second.game_over_count,
        rejected_round_count=(
            first.rejected_round_count + second.rejected_round_count
        ),
        cancelled_env_count=(
            first.cancelled_env_count + second.cancelled_env_count
        ),
        total_trace_step_count=(
            first.total_trace_step_count + second.total_trace_step_count
        ),
        max_trace_step_count=max(
            first.max_trace_step_count,
            second.max_trace_step_count,
        ),
        model_win_count=first.model_win_count + second.model_win_count,
        auto_win_count=first.auto_win_count + second.auto_win_count,
        model_declarer_round_count=(
            first.model_declarer_round_count
            + second.model_declarer_round_count
        ),
        model_reward_sum=first.model_reward_sum
        + second.model_reward_sum,
        auto_reward_sum=first.auto_reward_sum + second.auto_reward_sum,
        elapsed_seconds_max=max(
            first.elapsed_seconds_max,
            second.elapsed_seconds_max,
        ),
    )


def _segment_buffer(
    segment: shared_memory.SharedMemory,
) -> memoryview[int]:
    buffer = segment.buf
    assert buffer is not None
    return buffer


def _close_created_segments(
    segments: tuple[shared_memory.SharedMemory, ...],
) -> None:
    for segment in segments:
        try:
            segment.close()
        finally:
            segment.unlink()


__all__ = (
    "RolloutRoundTargetReached",
    "RolloutStopRequested",
    "RolloutWaitOutcome",
    "SharedRolloutArenaGroup",
    "close_shared_rollout_arenas",
    "create_shared_rollout_arena_group",
    "header_snapshot",
    "reset_rollout_arenas",
    "snapshot_rollout_arenas",
    "wait_rollout_round_target_or_stop",
)
