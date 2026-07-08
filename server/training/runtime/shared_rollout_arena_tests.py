"""Black-box tests for shared rollout arenas."""

from __future__ import annotations

import multiprocessing as mp

import torch

from server.result import Ok, Rejected
from server.training.policy_sampling import DecisionHandle
from server.training.returns import ReturnCommit
from server.training.runtime.shared_rollout_arena import (
    SharedRolloutArenaGroup,
    attach_rollout_arena_reader,
    attach_rollout_arena_writer,
    close_shared_rollout_arenas,
    create_shared_rollout_arena_group,
    reset_rollout_arenas,
    snapshot_rollout_arenas,
    wait_all_rollout_arenas_full,
)
from server.training.runtime.shared_rollout_arena.types import (
    RolloutRoundMetrics,
)


def test_append_round_partially_fills_and_filters_rank() -> None:
    group_result = _arena_group(worker_count=1, capacity=2)
    assert isinstance(group_result, Ok)
    group = group_result.value
    try:
        writer = attach_rollout_arena_writer(group.handles[0])
        reader = attach_rollout_arena_reader(group.handles)
        try:
            append_result = writer.append_round(
                policy_version=3,
                metrics=_metrics(decision_count=3),
                commit=_commit(
                    policy_version=3,
                    model_ranks=(0, 1, 0),
                ),
            )
            assert isinstance(append_result, Ok)
            assert append_result.value.accepted_sample_count == 2
            assert append_result.value.dropped_sample_count == 1
            assert append_result.value.arena_full

            rank0 = reader.read_rank_batch(
                policy_version=3, model_rank_index=0
            )
            assert isinstance(rank0, Ok)
            assert int(rank0.value.slot_indices.shape[0]) == 1
            assert torch.equal(
                rank0.value.return_values, torch.tensor((1.0,))
            )

            rank1 = reader.read_rank_batch(
                policy_version=3, model_rank_index=1
            )
            assert isinstance(rank1, Ok)
            assert int(rank1.value.slot_indices.shape[0]) == 1
            assert torch.equal(
                rank1.value.return_values, torch.tensor((2.0,))
            )
        finally:
            writer.close()
            reader.close()
    finally:
        close_shared_rollout_arenas(group)


def test_wait_all_rollout_arenas_full_uses_predicate_state() -> None:
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
            assert isinstance(first_append, Ok)
            early_wait = wait_all_rollout_arenas_full(
                group=group,
                policy_version=3,
                timeout_seconds=0.001,
            )
            assert isinstance(early_wait, Rejected)

            second_append = second_writer.append_round(
                policy_version=3,
                metrics=_metrics(decision_count=1),
                commit=_commit(policy_version=3, model_ranks=(0,)),
            )
            assert isinstance(second_append, Ok)
            full_wait = wait_all_rollout_arenas_full(
                group=group,
                policy_version=3,
                timeout_seconds=1.0,
            )
            assert isinstance(full_wait, Ok)
            assert full_wait.value.sample_count == 2
            assert full_wait.value.round_count == 2
        finally:
            first_writer.close()
            second_writer.close()
    finally:
        close_shared_rollout_arenas(group)


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
        samples_per_worker_update=capacity,
        policy_version=3,
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
    *, policy_version: int, model_ranks: tuple[int, ...]
) -> ReturnCommit:
    return ReturnCommit(
        policy_version=policy_version,
        first_episode_id=0,
        episode_count=1,
        decision_handles=tuple(
            DecisionHandle(
                model_rank_index=model_rank_index,
                policy_version=policy_version,
                slot_index=index,
                slot_generation=1,
            )
            for index, model_rank_index in enumerate(model_ranks)
        ),
        return_values=tuple(
            float(index + 1) for index in range(len(model_ranks))
        ),
    )
