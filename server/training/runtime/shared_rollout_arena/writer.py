"""Worker-side append operations for shared rollout arenas."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory

from server import result as _result
from server.result import Ok, Rejected
from server.training.returns import ReturnCommit
from server.training.runtime.shared_rollout_arena.schema import (
    RolloutArenaHeader,
    pack_header,
    pack_sample_references,
    unpack_header,
)
from server.training.runtime.shared_rollout_arena.types import (
    RolloutArenaAppendResult,
    RolloutArenaHandle,
    RolloutRoundMetrics,
)


@dataclass(slots=True)
class SharedRolloutArenaWriter:
    """Append completed worker samples to one shared arena."""

    handle: RolloutArenaHandle
    _segment: shared_memory.SharedMemory

    def append_round(
        self,
        *,
        policy_version: int,
        metrics: RolloutRoundMetrics,
        commit: ReturnCommit,
    ) -> _result.Ok[RolloutArenaAppendResult] | _result.Rejected:
        """Append as many rows from a completed round as fit."""
        assert policy_version >= 0
        if commit.policy_version != policy_version:
            return Rejected(
                reason="return commit policy version mismatch"
            )
        self.handle.condition.acquire()
        try:
            buffer = _segment_buffer(self._segment)
            header = unpack_header(buffer)
            if header.policy_version != policy_version:
                return Rejected(
                    reason="rollout arena policy version mismatch"
                )
            if header.sample_count >= header.capacity:
                dropped = commit.sample_count()
                updated = _add_dropped_samples(header, dropped)
                pack_header(buffer, header=updated)
                return Ok(
                    value=RolloutArenaAppendResult(
                        accepted_sample_count=0,
                        dropped_sample_count=dropped,
                        arena_full=True,
                    )
                )
            remaining = header.capacity - header.sample_count
            accepted = min(remaining, commit.sample_count())
            dropped = commit.sample_count() - accepted
            _write_commit_rows(
                buffer=buffer,
                capacity=header.capacity,
                start_index=header.sample_count,
                commit=commit,
                accepted_count=accepted,
            )
            updated = _advance_header(
                header=header,
                metrics=metrics,
                accepted_step_counts=commit.step_counts[:accepted],
                accepted_sample_count=accepted,
                dropped_sample_count=dropped,
            )
            pack_header(buffer, header=updated)
            if updated.full:
                self.handle.condition.notify_all()
            return Ok(
                value=RolloutArenaAppendResult(
                    accepted_sample_count=accepted,
                    dropped_sample_count=dropped,
                    arena_full=updated.full,
                )
            )
        finally:
            self.handle.condition.release()

    def record_cancelled_envs(self, count: int) -> None:
        """Add cancelled game-env count after an arena reaches full."""
        assert count >= 0
        if count == 0:
            return
        self.handle.condition.acquire()
        try:
            buffer = _segment_buffer(self._segment)
            header = unpack_header(buffer)
            updated = RolloutArenaHeader(
                policy_version=header.policy_version,
                sample_count=header.sample_count,
                capacity=header.capacity,
                target_sample_count=header.target_sample_count,
                full=header.full,
                round_count=header.round_count,
                generated_action_count=header.generated_action_count,
                accepted_action_count=header.accepted_action_count,
                action_choice_count=header.action_choice_count,
                game_over_count=header.game_over_count,
                dropped_sample_count=header.dropped_sample_count,
                cancelled_env_count=header.cancelled_env_count + count,
                total_step_count=header.total_step_count,
                max_step_count=header.max_step_count,
                team0_reward_sum=header.team0_reward_sum,
                team1_reward_sum=header.team1_reward_sum,
                elapsed_seconds_max=header.elapsed_seconds_max,
            )
            pack_header(buffer, header=updated)
        finally:
            self.handle.condition.release()

    def close(self) -> None:
        """Detach this process from the shared memory segment."""
        self._segment.close()


def attach_rollout_arena_writer(
    handle: RolloutArenaHandle,
) -> SharedRolloutArenaWriter:
    """Attach a worker process to its rollout arena."""
    return SharedRolloutArenaWriter(
        handle=handle,
        _segment=shared_memory.SharedMemory(
            name=handle.shared_memory_name
        ),
    )


def _segment_buffer(
    segment: shared_memory.SharedMemory,
) -> memoryview[int]:
    buffer = segment.buf
    assert buffer is not None
    return buffer


def _write_commit_rows(
    *,
    buffer: memoryview,
    capacity: int,
    start_index: int,
    commit: ReturnCommit,
    accepted_count: int,
) -> None:
    assert capacity > 0
    assert start_index >= 0
    assert accepted_count >= 0
    pack_sample_references(
        buffer=buffer,
        capacity=capacity,
        start_index=start_index,
        row_indices=commit.row_indices[:accepted_count],
        step_counts=commit.step_counts[:accepted_count],
        return_values=commit.return_values[:accepted_count],
    )


def _advance_header(
    *,
    header: RolloutArenaHeader,
    metrics: RolloutRoundMetrics,
    accepted_step_counts: tuple[int, ...],
    accepted_sample_count: int,
    dropped_sample_count: int,
) -> RolloutArenaHeader:
    assert accepted_sample_count >= 0
    assert dropped_sample_count >= 0
    assert len(accepted_step_counts) == accepted_sample_count
    assert all(count > 0 for count in accepted_step_counts)
    new_count = header.sample_count + accepted_sample_count
    accepted_round = accepted_sample_count > 0
    accepted_total_steps = sum(accepted_step_counts)
    accepted_max_steps = max(accepted_step_counts, default=0)
    full = header.full or new_count >= header.target_sample_count
    return RolloutArenaHeader(
        policy_version=header.policy_version,
        sample_count=new_count,
        capacity=header.capacity,
        target_sample_count=header.target_sample_count,
        full=full,
        round_count=header.round_count + (1 if accepted_round else 0),
        generated_action_count=(
            header.generated_action_count
            + (metrics.generated_action_count if accepted_round else 0)
        ),
        accepted_action_count=(
            header.accepted_action_count
            + (metrics.accepted_action_count if accepted_round else 0)
        ),
        action_choice_count=(
            header.action_choice_count
            + (metrics.action_choice_count if accepted_round else 0)
        ),
        game_over_count=(
            header.game_over_count
            + (1 if accepted_round and metrics.game_over else 0)
        ),
        dropped_sample_count=(
            header.dropped_sample_count + dropped_sample_count
        ),
        cancelled_env_count=header.cancelled_env_count,
        total_step_count=(
            header.total_step_count + accepted_total_steps
        ),
        max_step_count=max(header.max_step_count, accepted_max_steps),
        team0_reward_sum=(
            header.team0_reward_sum
            + (metrics.team0_reward if accepted_round else 0.0)
        ),
        team1_reward_sum=(
            header.team1_reward_sum
            + (metrics.team1_reward if accepted_round else 0.0)
        ),
        elapsed_seconds_max=max(
            header.elapsed_seconds_max,
            metrics.elapsed_seconds if accepted_round else 0.0,
        ),
    )


def _add_dropped_samples(
    header: RolloutArenaHeader, count: int
) -> RolloutArenaHeader:
    assert count >= 0
    return RolloutArenaHeader(
        policy_version=header.policy_version,
        sample_count=header.sample_count,
        capacity=header.capacity,
        target_sample_count=header.target_sample_count,
        full=header.full,
        round_count=header.round_count,
        generated_action_count=header.generated_action_count,
        accepted_action_count=header.accepted_action_count,
        action_choice_count=header.action_choice_count,
        game_over_count=header.game_over_count,
        dropped_sample_count=header.dropped_sample_count + count,
        cancelled_env_count=header.cancelled_env_count,
        total_step_count=header.total_step_count,
        max_step_count=header.max_step_count,
        team0_reward_sum=header.team0_reward_sum,
        team1_reward_sum=header.team1_reward_sum,
        elapsed_seconds_max=header.elapsed_seconds_max,
    )
