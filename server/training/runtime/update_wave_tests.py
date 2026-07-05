"""Black-box tests for synchronized update wave partitioning."""

from __future__ import annotations

from server.result import Ok, Rejected
from server.training.policy_sampling import DecisionHandle
from server.training.rollout_commit import RolloutCommit
from server.training.runtime.update_wave import (
    build_synchronized_update_wave,
)


def test_build_synchronized_update_wave_keeps_idle_rank() -> None:
    result = build_synchronized_update_wave(
        rollout_commit=_single_rank_commit(rank_index=0),
        rank_count=2,
    )

    assert isinstance(result, Ok)
    wave = result.value
    assert wave.policy_version == 7
    assert wave.rank_count == 2
    assert tuple(shard.rank_index for shard in wave.shards) == (0, 1)
    assert not wave.shards[0].is_empty()
    assert wave.shards[1].is_empty()
    assert wave.shards[0].rollout_commit.decision_handles == (
        DecisionHandle(
            model_rank_index=0,
            policy_version=7,
            slot_index=3,
            slot_generation=1,
        ),
    )
    assert wave.shards[1].rollout_commit.decision_handles == ()


def test_build_synchronized_update_wave_rejects_empty_commit() -> None:
    result = build_synchronized_update_wave(
        rollout_commit=RolloutCommit(
            policy_version=7,
            first_episode_id=11,
            episode_count=1,
            decision_handles=(),
            reward_after_step=(),
            terminal_rewards=(),
            trajectory_team_indices=(),
            trajectory_offsets=(0,),
        ),
        rank_count=2,
    )

    assert isinstance(result, Rejected)
    assert result.reason == "update wave requires rollout decisions"


def test_build_synchronized_update_wave_rejects_empty_rank_count() -> (
    None
):
    result = build_synchronized_update_wave(
        rollout_commit=_single_rank_commit(rank_index=0),
        rank_count=0,
    )

    assert isinstance(result, Rejected)
    assert result.reason == "update wave rank_count must be > 0"


def _single_rank_commit(*, rank_index: int) -> RolloutCommit:
    handle = DecisionHandle(
        model_rank_index=rank_index,
        policy_version=7,
        slot_index=3,
        slot_generation=1,
    )
    return RolloutCommit(
        policy_version=7,
        first_episode_id=11,
        episode_count=1,
        decision_handles=(handle,),
        reward_after_step=(0.0,),
        terminal_rewards=(1.0,),
        trajectory_team_indices=(0,),
        trajectory_offsets=(0, 1),
    )
