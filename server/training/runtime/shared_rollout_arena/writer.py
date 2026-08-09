"""Worker-side atomic completed-round arena writes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from multiprocessing import shared_memory

from server.foundation import result as _result
from server.foundation.result import Ok, Rejected
from server.training.rollout import CompletedRoundTrajectory
from server.training.runtime.shared_rollout_arena.schema import (
    MAX_TRAINABLE_DECISIONS_PER_ROUND,
    RolloutArenaHeader,
    pack_decisions,
    pack_header,
    pack_trajectories,
    unpack_header,
)
from server.training.runtime.shared_rollout_arena.types import (
    RolloutArenaAppendResult,
    RolloutArenaHandle,
    RolloutRoundMetrics,
)


@dataclass(slots=True)
class SharedRolloutArenaWriter:
    """Append whole rounds to one worker-owned shared arena."""

    handle: RolloutArenaHandle
    _segment: shared_memory.SharedMemory

    def append_round(
        self,
        *,
        policy_version: int,
        metrics: RolloutRoundMetrics,
        trajectory: CompletedRoundTrajectory,
    ) -> _result.Ok[RolloutArenaAppendResult] | _result.Rejected:
        """Accept all trajectory rows or write none of them."""
        assert policy_version >= 0
        if trajectory.policy_version != policy_version:
            return Rejected(
                reason="round trajectory policy version mismatch"
            )
        decisions = trajectory.decisions()
        decision_count = len(decisions)
        if decision_count == 0:
            return Rejected(
                reason="completed round has no trainable decisions"
            )
        if decision_count > MAX_TRAINABLE_DECISIONS_PER_ROUND:
            return Rejected(
                reason="completed round exceeds rollout decision bound"
            )
        seat_trajectories = tuple(
            item for item in trajectory.seats if item.decisions
        )
        assert seat_trajectories
        assert len(seat_trajectories) <= 2
        if metrics.trainable_decision_count != decision_count:
            return Rejected(reason="round trajectory metric mismatch")
        _ = self.handle.lock.acquire()
        try:
            buffer = _segment_buffer(self._segment)
            header = unpack_header(buffer)
            if header.policy_version != policy_version:
                return Rejected(
                    reason="rollout arena policy version mismatch"
                )
            if not _round_fits(
                header=header,
                decision_count=decision_count,
                trajectory_count=len(seat_trajectories),
            ):
                pack_header(
                    buffer,
                    header=_reject_round(header),
                )
                result = Ok(
                    RolloutArenaAppendResult(
                        accepted=False,
                        capacity_reached=True,
                    )
                )
            else:
                row_indices = tuple(
                    item.replay.row_index for item in decisions
                )
                step_counts = tuple(
                    item.trace_step_count for item in decisions
                )
                offsets, lengths = _trajectory_spans(
                    decision_start=header.decision_count,
                    trajectory=trajectory,
                )
                pack_decisions(
                    buffer=buffer,
                    decision_capacity=header.decision_capacity,
                    start=header.decision_count,
                    row_indices=row_indices,
                    step_counts=step_counts,
                )
                pack_trajectories(
                    buffer=buffer,
                    decision_capacity=header.decision_capacity,
                    trajectory_capacity=header.trajectory_capacity,
                    start=header.trajectory_count,
                    offsets=offsets,
                    lengths=lengths,
                    terminal_rewards=tuple(
                        trajectory.terminal_reward for _ in offsets
                    ),
                )
                updated = _accept_round(
                    header=header,
                    metrics=metrics,
                    step_counts=step_counts,
                    trajectory_count=len(offsets),
                )
                pack_header(buffer, header=updated)
                result = Ok(
                    RolloutArenaAppendResult(
                        accepted=True,
                        capacity_reached=(
                            updated.round_count
                            >= updated.round_capacity
                        ),
                    )
                )
        finally:
            self.handle.lock.release()
        _notify_progress(self.handle)
        return result

    def record_cancelled_envs(self, count: int) -> None:
        """Record game environments cancelled at update boundary."""
        assert count >= 0
        if count == 0:
            return
        _ = self.handle.lock.acquire()
        try:
            buffer = _segment_buffer(self._segment)
            header = unpack_header(buffer)
            pack_header(
                buffer,
                header=replace(
                    header,
                    cancelled_env_count=(
                        header.cancelled_env_count + count
                    ),
                ),
            )
        finally:
            self.handle.lock.release()
        _notify_progress(self.handle)

    def close(self) -> None:
        self._segment.close()


def attach_rollout_arena_writer(
    handle: RolloutArenaHandle,
) -> SharedRolloutArenaWriter:
    return SharedRolloutArenaWriter(
        handle=handle,
        _segment=shared_memory.SharedMemory(
            name=handle.shared_memory_name
        ),
    )


def _round_fits(
    *,
    header: RolloutArenaHeader,
    decision_count: int,
    trajectory_count: int,
) -> bool:
    return (
        header.round_count < header.round_capacity
        and header.decision_count + decision_count
        <= header.decision_capacity
        and header.trajectory_count + trajectory_count
        <= header.trajectory_capacity
    )


def _trajectory_spans(
    *, decision_start: int, trajectory: CompletedRoundTrajectory
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    offsets: list[int] = []
    lengths: list[int] = []
    next_offset = decision_start
    for seat_trajectory in trajectory.seats:
        length = len(seat_trajectory.decisions)
        if length == 0:
            continue
        offsets.append(next_offset)
        lengths.append(length)
        next_offset += length
    assert next_offset == decision_start + trajectory.decision_count()
    return tuple(offsets), tuple(lengths)


def _accept_round(
    *,
    header: RolloutArenaHeader,
    metrics: RolloutRoundMetrics,
    step_counts: tuple[int, ...],
    trajectory_count: int,
) -> RolloutArenaHeader:
    decision_count = len(step_counts)
    return RolloutArenaHeader(
        policy_version=header.policy_version,
        decision_count=header.decision_count + decision_count,
        decision_capacity=header.decision_capacity,
        trajectory_count=header.trajectory_count + trajectory_count,
        trajectory_capacity=header.trajectory_capacity,
        round_count=header.round_count + 1,
        round_capacity=header.round_capacity,
        model_action_count=(
            header.model_action_count + metrics.model_action_count
        ),
        auto_action_count=(
            header.auto_action_count + metrics.auto_action_count
        ),
        forced_action_count=(
            header.forced_action_count + metrics.forced_action_count
        ),
        trainable_decision_count=(
            header.trainable_decision_count
            + metrics.trainable_decision_count
        ),
        scored_choice_step_count=(
            header.scored_choice_step_count
            + metrics.scored_choice_step_count
        ),
        game_over_count=header.game_over_count + int(metrics.game_over),
        rejected_round_count=header.rejected_round_count,
        cancelled_env_count=header.cancelled_env_count,
        total_trace_step_count=(
            header.total_trace_step_count + sum(step_counts)
        ),
        max_trace_step_count=max(
            header.max_trace_step_count,
            max(step_counts),
        ),
        model_win_count=(
            header.model_win_count + int(metrics.model_reward > 0.0)
        ),
        auto_win_count=(
            header.auto_win_count + int(metrics.auto_reward > 0.0)
        ),
        model_declarer_round_count=(
            header.model_declarer_round_count
            + int(metrics.model_declarer)
        ),
        model_reward_sum=header.model_reward_sum + metrics.model_reward,
        auto_reward_sum=header.auto_reward_sum + metrics.auto_reward,
        elapsed_seconds_max=max(
            header.elapsed_seconds_max,
            metrics.elapsed_seconds,
        ),
    )


def _reject_round(header: RolloutArenaHeader) -> RolloutArenaHeader:
    return RolloutArenaHeader(
        policy_version=header.policy_version,
        decision_count=header.decision_count,
        decision_capacity=header.decision_capacity,
        trajectory_count=header.trajectory_count,
        trajectory_capacity=header.trajectory_capacity,
        round_count=header.round_count,
        round_capacity=header.round_capacity,
        model_action_count=header.model_action_count,
        auto_action_count=header.auto_action_count,
        forced_action_count=header.forced_action_count,
        trainable_decision_count=header.trainable_decision_count,
        scored_choice_step_count=header.scored_choice_step_count,
        game_over_count=header.game_over_count,
        rejected_round_count=header.rejected_round_count + 1,
        cancelled_env_count=header.cancelled_env_count,
        total_trace_step_count=header.total_trace_step_count,
        max_trace_step_count=header.max_trace_step_count,
        model_win_count=header.model_win_count,
        auto_win_count=header.auto_win_count,
        model_declarer_round_count=header.model_declarer_round_count,
        model_reward_sum=header.model_reward_sum,
        auto_reward_sum=header.auto_reward_sum,
        elapsed_seconds_max=header.elapsed_seconds_max,
    )


def _segment_buffer(
    segment: shared_memory.SharedMemory,
) -> memoryview[int]:
    buffer = segment.buf
    assert buffer is not None
    return buffer


def _notify_progress(handle: RolloutArenaHandle) -> None:
    _ = handle.progress_condition.acquire()
    try:
        handle.progress_condition.notify_all()
    finally:
        handle.progress_condition.release()


__all__ = ("SharedRolloutArenaWriter", "attach_rollout_arena_writer")
