"""Black-box tests for shared rollout arenas."""

from __future__ import annotations

import multiprocessing as mp
from typing import NoReturn

import pytest
import torch

from server.foundation.result import Ok, Rejected
from server.training.returns import ReturnCommit
from server.training.runtime.shared_rollout_arena import (
    RolloutArenaHandle,
    SharedRolloutArenaGroup,
    attach_rollout_arena_reader,
    attach_rollout_arena_writer,
    close_shared_rollout_arenas,
    create_shared_rollout_arena_group,
    reset_rollout_arenas,
    snapshot_rollout_arenas,
    wait_rollout_sample_target,
)
from server.training.runtime.shared_rollout_arena import (
    group as arena_group_module,
)
from server.training.runtime.shared_rollout_arena.types import (
    RolloutRoundMetrics,
)


def test_read_rank_batch_reads_assigned_worker_arenas() -> None:
    group_result = _arena_group(worker_count=2, capacity=1)
    assert isinstance(group_result, Ok)
    group = group_result.value
    try:
        first_writer = attach_rollout_arena_writer(group.handles[0])
        second_writer = attach_rollout_arena_writer(group.handles[1])
        first_reader = attach_rollout_arena_reader((group.handles[0],))
        second_reader = attach_rollout_arena_reader((group.handles[1],))
        try:
            first_append = first_writer.append_round(
                policy_version=3,
                metrics=_metrics(decision_count=1),
                commit=_commit(
                    policy_version=3,
                    model_ranks=(1,),
                    step_counts=(2,),
                    return_values=(10.0,),
                ),
            )
            second_append = second_writer.append_round(
                policy_version=3,
                metrics=_metrics(decision_count=1),
                commit=_commit(
                    policy_version=3,
                    model_ranks=(0,),
                    step_counts=(3,),
                    return_values=(20.0,),
                ),
            )
            assert isinstance(first_append, Ok)
            assert first_append.value.accepted_sample_count == 1
            assert first_append.value.dropped_sample_count == 0
            assert isinstance(second_append, Ok)
            assert second_append.value.accepted_sample_count == 1
            assert second_append.value.dropped_sample_count == 0

            device = torch.device("cpu")
            rank0 = first_reader.read_rank_batch(
                policy_version=3, model_rank_index=0, device=device
            )
            assert isinstance(rank0, Ok)
            assert int(rank0.value.row_indices.shape[0]) == 1
            assert torch.equal(
                rank0.value.step_counts, torch.tensor((2,))
            )
            assert torch.equal(
                rank0.value.return_values, torch.tensor((10.0,))
            )
            assert rank0.value.total_step_count == 2
            assert rank0.value.max_step_count == 2

            rank1 = second_reader.read_rank_batch(
                policy_version=3, model_rank_index=1, device=device
            )
            assert isinstance(rank1, Ok)
            assert int(rank1.value.row_indices.shape[0]) == 1
            assert torch.equal(
                rank1.value.step_counts, torch.tensor((3,))
            )
            assert torch.equal(
                rank1.value.return_values, torch.tensor((20.0,))
            )
            assert rank1.value.total_step_count == 3
            assert rank1.value.max_step_count == 3
        finally:
            first_writer.close()
            second_writer.close()
            first_reader.close()
            second_reader.close()
    finally:
        close_shared_rollout_arenas(group)


def test_wait_rollout_sample_target_uses_aggregate_samples() -> None:
    group_result = _arena_group(worker_count=2, capacity=2)
    assert isinstance(group_result, Ok)
    group = group_result.value
    try:
        first_writer = attach_rollout_arena_writer(group.handles[0])
        second_writer = attach_rollout_arena_writer(group.handles[1])
        try:
            first_append = first_writer.append_round(
                policy_version=3,
                metrics=_metrics(decision_count=1),
                commit=_commit(policy_version=3, model_ranks=(0,)),
            )
            assert isinstance(first_append, Ok)
            early_wait = wait_rollout_sample_target(
                group=group,
                policy_version=3,
                target_sample_count=2,
                timeout_seconds=0.001,
            )
            assert isinstance(early_wait, Rejected)

            second_append = second_writer.append_round(
                policy_version=3,
                metrics=_metrics(decision_count=1),
                commit=_commit(policy_version=3, model_ranks=(0,)),
            )
            assert isinstance(second_append, Ok)
            full_wait = wait_rollout_sample_target(
                group=group,
                policy_version=3,
                target_sample_count=2,
                timeout_seconds=1.0,
            )
            assert isinstance(full_wait, Ok)
            assert full_wait.value.sample_count == 2
            assert full_wait.value.round_count == 2
            assert full_wait.value.total_step_count == 2
            assert full_wait.value.max_step_count == 1
        finally:
            first_writer.close()
            second_writer.close()
    finally:
        close_shared_rollout_arenas(group)


def test_snapshot_and_reset_reuse_group_owned_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group_result = _arena_group(worker_count=2, capacity=1)
    assert isinstance(group_result, Ok)
    group = group_result.value
    try:
        first_writer = attach_rollout_arena_writer(group.handles[0])
        second_writer = attach_rollout_arena_writer(group.handles[1])
        try:
            first_append = first_writer.append_round(
                policy_version=3,
                metrics=_metrics(decision_count=1),
                commit=_commit(policy_version=3, model_ranks=(0,)),
            )
            second_append = second_writer.append_round(
                policy_version=3,
                metrics=_metrics(decision_count=1),
                commit=_commit(policy_version=3, model_ranks=(0,)),
            )
        finally:
            first_writer.close()
            second_writer.close()

        assert isinstance(first_append, Ok)
        assert isinstance(second_append, Ok)
        monkeypatch.setattr(
            arena_group_module.shared_memory,
            "SharedMemory",
            _reject_group_shared_memory_attach,
        )

        snapshot = snapshot_rollout_arenas(
            group=group, policy_version=3
        )
        reset = reset_rollout_arenas(group=group, policy_version=4)
        reset_snapshot = snapshot_rollout_arenas(
            group=group, policy_version=4
        )
    finally:
        close_shared_rollout_arenas(group)

    assert isinstance(snapshot, Ok)
    assert snapshot.value.sample_count == 2
    assert isinstance(reset, Ok)
    assert isinstance(reset_snapshot, Ok)
    assert reset_snapshot.value.sample_count == 0


def test_arena_group_uses_explicit_per_worker_capacity() -> None:
    group_result = _arena_group(worker_count=3, capacity=5)
    assert isinstance(group_result, Ok)
    group = group_result.value
    try:
        snapshot = snapshot_rollout_arenas(
            group=group, policy_version=3
        )
    finally:
        close_shared_rollout_arenas(group)

    assert isinstance(snapshot, Ok)
    assert tuple(handle.capacity for handle in group.handles) == (
        5,
        5,
        5,
    )
    assert snapshot.value.capacity == 15


def test_append_round_notifies_progress_after_arena_write() -> None:
    group_result = _arena_group(worker_count=1, capacity=1)
    assert isinstance(group_result, Ok)
    group = group_result.value
    events: list[str] = []
    arena_lock = _RecordingCondition(name="arena", events=events)
    progress_condition = _RecordingCondition(
        name="progress",
        events=events,
    )
    real_handle = group.handles[0]
    handle = RolloutArenaHandle(
        worker_index=real_handle.worker_index,
        shared_memory_name=real_handle.shared_memory_name,
        capacity=real_handle.capacity,
        lock=arena_lock,
        progress_condition=progress_condition,
    )
    try:
        writer = attach_rollout_arena_writer(handle)
        try:
            append = writer.append_round(
                policy_version=3,
                metrics=_metrics(decision_count=1),
                commit=_commit(policy_version=3, model_ranks=(0,)),
            )
        finally:
            writer.close()
    finally:
        close_shared_rollout_arenas(group)

    assert isinstance(append, Ok)
    assert events == [
        "arena.acquire",
        "arena.release",
        "progress.acquire",
        "progress.notify_all",
        "progress.release",
    ]


def test_reset_rollout_arenas_notifies_only_progress() -> None:
    group_result = _arena_group(worker_count=1, capacity=1)
    assert isinstance(group_result, Ok)
    group = group_result.value
    events: list[str] = []
    arena_lock = _RecordingCondition(name="arena", events=events)
    progress_condition = _RecordingCondition(
        name="progress",
        events=events,
    )
    real_handle = group.handles[0]
    handle = RolloutArenaHandle(
        worker_index=real_handle.worker_index,
        shared_memory_name=real_handle.shared_memory_name,
        capacity=real_handle.capacity,
        lock=arena_lock,
        progress_condition=progress_condition,
    )
    recording_group = SharedRolloutArenaGroup(
        handles=(handle,), segments=group.segments
    )
    try:
        reset = reset_rollout_arenas(
            group=recording_group,
            policy_version=4,
        )
    finally:
        close_shared_rollout_arenas(group)

    assert isinstance(reset, Ok)
    assert events == [
        "progress.acquire",
        "arena.acquire",
        "arena.release",
        "progress.notify_all",
        "progress.release",
    ]


def test_append_round_drops_only_after_capacity() -> None:
    group_result = _arena_group(worker_count=1, capacity=2)
    assert isinstance(group_result, Ok)
    group = group_result.value
    try:
        writer = attach_rollout_arena_writer(group.handles[0])
        try:
            first_append = writer.append_round(
                policy_version=3,
                metrics=_metrics(decision_count=1),
                commit=_commit(policy_version=3, model_ranks=(0,)),
            )
            second_append = writer.append_round(
                policy_version=3,
                metrics=_metrics(decision_count=1),
                commit=_commit(policy_version=3, model_ranks=(0,)),
            )
            third_append = writer.append_round(
                policy_version=3,
                metrics=_metrics(decision_count=1),
                commit=_commit(policy_version=3, model_ranks=(0,)),
            )
            snapshot = snapshot_rollout_arenas(
                group=group,
                policy_version=3,
            )
        finally:
            writer.close()
    finally:
        close_shared_rollout_arenas(group)

    assert isinstance(first_append, Ok)
    assert not first_append.value.capacity_reached
    assert isinstance(second_append, Ok)
    assert second_append.value.accepted_sample_count == 1
    assert second_append.value.dropped_sample_count == 0
    assert second_append.value.capacity_reached
    assert isinstance(third_append, Ok)
    assert third_append.value.accepted_sample_count == 0
    assert third_append.value.dropped_sample_count == 1
    assert third_append.value.capacity_reached
    assert isinstance(snapshot, Ok)
    assert snapshot.value.capacity == 2
    assert snapshot.value.sample_count == 2
    assert snapshot.value.dropped_sample_count == 1


def test_read_rank_batch_allows_partial_assigned_arena() -> None:
    group_result = _arena_group(worker_count=1, capacity=4)
    assert isinstance(group_result, Ok)
    group = group_result.value
    try:
        writer = attach_rollout_arena_writer(group.handles[0])
        reader = attach_rollout_arena_reader((group.handles[0],))
        try:
            append = writer.append_round(
                policy_version=3,
                metrics=_metrics(decision_count=1),
                commit=_commit(policy_version=3, model_ranks=(0,)),
            )
            assert isinstance(append, Ok)
            assert not append.value.capacity_reached
            result = reader.read_rank_batch(
                policy_version=3,
                model_rank_index=0,
                device=torch.device("cpu"),
            )
        finally:
            writer.close()
            reader.close()
    finally:
        close_shared_rollout_arenas(group)

    assert isinstance(result, Ok)
    assert int(result.value.row_indices.shape[0]) == 1
    assert result.value.round_count == 1
    assert result.value.total_step_count == 1


def test_read_rank_batch_allows_empty_assigned_shard() -> None:
    reader = attach_rollout_arena_reader(())
    try:
        result = reader.read_rank_batch(
            policy_version=3,
            model_rank_index=1,
            device=torch.device("cpu"),
        )
    finally:
        reader.close()

    assert isinstance(result, Ok)
    assert result.value.policy_version == 3
    assert result.value.model_rank_index == 1
    assert int(result.value.row_indices.shape[0]) == 0
    assert int(result.value.step_counts.shape[0]) == 0
    assert int(result.value.return_values.shape[0]) == 0
    assert result.value.round_count == 0
    assert result.value.total_step_count == 0
    assert result.value.max_step_count == 0


def test_reset_rollout_arenas_allows_new_policy_version() -> None:
    group_result = _arena_group(worker_count=1, capacity=1)
    assert isinstance(group_result, Ok)
    group = group_result.value
    try:
        writer = attach_rollout_arena_writer(group.handles[0])
        try:
            first_append = writer.append_round(
                policy_version=3,
                metrics=_metrics(decision_count=1),
                commit=_commit(policy_version=3, model_ranks=(0,)),
            )
            assert isinstance(first_append, Ok)
            reset_result = reset_rollout_arenas(
                group=group,
                policy_version=4,
            )
            assert isinstance(reset_result, Ok)
            second_append = writer.append_round(
                policy_version=4,
                metrics=_metrics(decision_count=1),
                commit=_commit(policy_version=4, model_ranks=(0,)),
            )
            assert isinstance(second_append, Ok)
            snapshot = snapshot_rollout_arenas(
                group=group,
                policy_version=4,
            )
            assert isinstance(snapshot, Ok)
            assert snapshot.value.sample_count == 1
            assert snapshot.value.total_step_count == 1
            assert snapshot.value.max_step_count == 1
        finally:
            writer.close()
    finally:
        close_shared_rollout_arenas(group)


def _arena_group(
    *, worker_count: int, capacity: int
) -> Ok[SharedRolloutArenaGroup] | Rejected:
    context = mp.get_context("spawn")
    return create_shared_rollout_arena_group(
        context=context,
        worker_count=worker_count,
        arena_capacity_per_worker=capacity,
        policy_version=3,
    )


def _reject_group_shared_memory_attach(*, name: str) -> NoReturn:
    raise AssertionError(
        f"group snapshot/reset must reuse owned segment: {name}"
    )


def _metrics(*, decision_count: int) -> RolloutRoundMetrics:
    return RolloutRoundMetrics(
        team0_reward=1.0,
        team1_reward=-1.0,
        generated_action_count=decision_count,
        accepted_action_count=decision_count,
        action_choice_count=decision_count * 2,
        decision_count=decision_count,
        elapsed_seconds=0.25,
        game_over=False,
    )


def _commit(
    *,
    policy_version: int,
    model_ranks: tuple[int, ...],
    step_counts: tuple[int, ...] | None = None,
    return_values: tuple[float, ...] | None = None,
) -> ReturnCommit:
    values = (
        tuple(float(index + 1) for index in range(len(model_ranks)))
        if return_values is None
        else return_values
    )
    assert len(values) == len(model_ranks)
    counts = (
        tuple(1 for _ in model_ranks)
        if step_counts is None
        else step_counts
    )
    assert len(counts) == len(model_ranks)
    return ReturnCommit(
        policy_version=policy_version,
        first_episode_id=0,
        episode_count=1,
        row_indices=tuple(index for index in range(len(model_ranks))),
        step_counts=counts,
        return_values=values,
    )


class _RecordingCondition:
    def __init__(self, *, name: str, events: list[str]) -> None:
        self._name = name
        self._events = events

    def acquire(self) -> bool:
        self._events.append(f"{self._name}.acquire")
        return True

    def release(self) -> None:
        self._events.append(f"{self._name}.release")

    def wait(self, timeout: float | None = None) -> bool:
        self._events.append(f"{self._name}.wait")
        return True

    def notify_all(self) -> None:
        self._events.append(f"{self._name}.notify_all")
