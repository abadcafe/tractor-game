"""Synchronized PPO update wave partitioning."""

from __future__ import annotations

from dataclasses import dataclass

from server import result as _result
from server.result import Ok, Rejected
from server.training.rollout_commit import (
    RolloutCommit,
    rollout_commit_for_model_rank,
)


@dataclass(frozen=True, slots=True)
class SynchronizedUpdateShard:
    """One fixed-rank shard in a synchronized PPO update wave."""

    rank_index: int
    policy_version: int
    rollout_commit: RolloutCommit

    def __post_init__(self) -> None:
        assert self.rank_index >= 0
        assert self.policy_version >= 0
        assert self.rollout_commit.policy_version == self.policy_version
        for handle in self.rollout_commit.decision_handles:
            assert handle.model_rank_index == self.rank_index

    def is_empty(self) -> bool:
        """Return whether this rank owns no rollout decisions."""
        return self.rollout_commit.is_empty()


@dataclass(frozen=True, slots=True)
class SynchronizedUpdateWave:
    """A complete fixed-width PPO update wave."""

    policy_version: int
    rank_count: int
    shards: tuple[SynchronizedUpdateShard, ...]

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert self.rank_count > 0
        assert len(self.shards) == self.rank_count
        rank_indices = tuple(shard.rank_index for shard in self.shards)
        assert rank_indices == tuple(range(self.rank_count))
        assert all(
            shard.policy_version == self.policy_version
            for shard in self.shards
        )
        assert any(not shard.is_empty() for shard in self.shards)


def build_synchronized_update_wave(
    *,
    rollout_commit: RolloutCommit,
    rank_count: int,
) -> _result.Ok[SynchronizedUpdateWave] | _result.Rejected:
    """Partition a global commit into one shard per update rank."""
    if rank_count <= 0:
        return Rejected(reason="update wave rank_count must be > 0")
    if rollout_commit.is_empty():
        return Rejected(reason="update wave requires rollout decisions")
    shards: list[SynchronizedUpdateShard] = []
    for rank_index in range(rank_count):
        rank_commit = rollout_commit_for_model_rank(
            commit=rollout_commit,
            model_rank_index=rank_index,
        )
        shards.append(
            SynchronizedUpdateShard(
                rank_index=rank_index,
                policy_version=rollout_commit.policy_version,
                rollout_commit=rank_commit,
            )
        )
    return Ok(
        value=SynchronizedUpdateWave(
            policy_version=rollout_commit.policy_version,
            rank_count=rank_count,
            shards=tuple(shards),
        )
    )
