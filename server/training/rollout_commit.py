"""Committed rollout metadata referencing model-rank replay records."""

from __future__ import annotations

import math
from dataclasses import dataclass

from server.sm.constants import get_team_index
from server.training.policy_sampling.records import DecisionHandle
from server.training.trajectory import DecisionStep


@dataclass(frozen=True, slots=True)
class RolloutCommit:
    """Rollout reward metadata keyed by replay handles."""

    policy_version: int
    first_episode_id: int
    episode_count: int
    decision_handles: tuple[DecisionHandle, ...]
    reward_after_step: tuple[float, ...]
    terminal_rewards: tuple[float, ...]
    trajectory_team_indices: tuple[int, ...]
    trajectory_offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert self.first_episode_id >= 0
        assert self.episode_count >= 0
        transition_count = len(self.decision_handles)
        assert len(self.reward_after_step) == transition_count
        assert self.trajectory_offsets
        assert self.trajectory_offsets[0] == 0
        assert self.trajectory_offsets[-1] == transition_count
        trajectory_count = len(self.trajectory_offsets) - 1
        assert len(self.terminal_rewards) == trajectory_count
        assert len(self.trajectory_team_indices) == trajectory_count
        assert all(
            handle.policy_version == self.policy_version
            for handle in self.decision_handles
        )
        assert all(
            math.isfinite(value)
            for value in (
                *self.reward_after_step,
                *self.terminal_rewards,
            )
        )
        for team_index in self.trajectory_team_indices:
            assert team_index in (0, 1)
        for index in range(trajectory_count):
            start = self.trajectory_offsets[index]
            end = self.trajectory_offsets[index + 1]
            assert 0 <= start < end <= transition_count

    def transition_count(self) -> int:
        """Return the number of committed decision transitions."""
        return len(self.decision_handles)

    def trajectory_count(self) -> int:
        """Return the number of committed team trajectories."""
        return len(self.terminal_rewards)

    def max_trajectory_length(self) -> int:
        """Return the widest committed team trajectory."""
        widest = 0
        for index in range(self.trajectory_count()):
            widest = max(
                widest,
                self.trajectory_offsets[index + 1]
                - self.trajectory_offsets[index],
            )
        return widest

    def is_empty(self) -> bool:
        """Return whether this commit has no decisions."""
        return self.transition_count() == 0


def terminal_reward_rollout_commit(
    *,
    policy_version: int,
    episode_id: int,
    steps: tuple[DecisionStep, ...],
    team0_reward: float,
    team1_reward: float,
) -> RolloutCommit:
    """Build a terminal-reward rollout commit for one round."""
    assert policy_version >= 0
    assert episode_id >= 0
    assert math.isfinite(team0_reward)
    assert math.isfinite(team1_reward)
    assert team0_reward + team1_reward == 0.0
    builder = _RolloutCommitBuilder(
        policy_version=policy_version,
        first_episode_id=episode_id,
        episode_count=1,
    )
    for team_index, reward in ((0, team0_reward), (1, team1_reward)):
        team_steps = tuple(
            step
            for step in steps
            if get_team_index(step.player_index) == team_index
        )
        if team_steps:
            builder.append_terminal_trajectory(
                team_index=team_index,
                steps=team_steps,
                terminal_reward=reward,
            )
    return builder.build()


def merge_rollout_commits(
    commits: tuple[RolloutCommit, ...],
) -> RolloutCommit:
    """Merge rollout commits collected under one policy version."""
    assert commits
    policy_version = commits[0].policy_version
    assert all(
        commit.policy_version == policy_version for commit in commits
    )
    builder = _RolloutCommitBuilder(
        policy_version=policy_version,
        first_episode_id=min(
            commit.first_episode_id for commit in commits
        ),
        episode_count=sum(commit.episode_count for commit in commits),
    )
    for commit in commits:
        builder.extend(commit)
    return builder.build()


def rollout_commit_for_model_rank(
    *,
    commit: RolloutCommit,
    model_rank_index: int,
) -> RolloutCommit:
    """Return the subset of a commit owned by one model rank."""
    assert model_rank_index >= 0
    builder = _RolloutCommitBuilder(
        policy_version=commit.policy_version,
        first_episode_id=commit.first_episode_id,
        episode_count=commit.episode_count,
    )
    for trajectory_index in range(commit.trajectory_count()):
        start = commit.trajectory_offsets[trajectory_index]
        end = commit.trajectory_offsets[trajectory_index + 1]
        trajectory_handles = commit.decision_handles[start:end]
        owner = trajectory_handles[0].model_rank_index
        assert all(
            handle.model_rank_index == owner
            for handle in trajectory_handles
        )
        handles = (
            trajectory_handles if owner == model_rank_index else ()
        )
        if not handles:
            continue
        builder.append_handles_trajectory(
            team_index=commit.trajectory_team_indices[trajectory_index],
            handles=handles,
            terminal_reward=commit.terminal_rewards[trajectory_index],
        )
    return builder.build()


@dataclass(slots=True)
class _RolloutCommitBuilder:
    policy_version: int
    first_episode_id: int
    episode_count: int
    decision_handles: list[DecisionHandle]
    reward_after_step: list[float]
    terminal_rewards: list[float]
    trajectory_team_indices: list[int]
    trajectory_offsets: list[int]

    def __init__(
        self,
        *,
        policy_version: int,
        first_episode_id: int,
        episode_count: int,
    ) -> None:
        self.policy_version = policy_version
        self.first_episode_id = first_episode_id
        self.episode_count = episode_count
        self.decision_handles = []
        self.reward_after_step = []
        self.terminal_rewards = []
        self.trajectory_team_indices = []
        self.trajectory_offsets = [0]

    def append_terminal_trajectory(
        self,
        *,
        team_index: int,
        steps: tuple[DecisionStep, ...],
        terminal_reward: float,
    ) -> None:
        assert steps
        assert all(
            get_team_index(step.player_index) == team_index
            for step in steps
        )
        self.append_handles_trajectory(
            team_index=team_index,
            handles=tuple(step.decision_handle for step in steps),
            terminal_reward=terminal_reward,
        )

    def append_handles_trajectory(
        self,
        *,
        team_index: int,
        handles: tuple[DecisionHandle, ...],
        terminal_reward: float,
    ) -> None:
        assert handles
        self.trajectory_team_indices.append(team_index)
        self.terminal_rewards.append(terminal_reward)
        self.decision_handles.extend(handles)
        self.reward_after_step.extend(0.0 for _ in handles)
        self.trajectory_offsets.append(len(self.decision_handles))

    def extend(self, commit: RolloutCommit) -> None:
        assert commit.policy_version == self.policy_version
        offset = len(self.decision_handles)
        self.decision_handles.extend(commit.decision_handles)
        self.reward_after_step.extend(commit.reward_after_step)
        self.terminal_rewards.extend(commit.terminal_rewards)
        self.trajectory_team_indices.extend(
            commit.trajectory_team_indices
        )
        self.trajectory_offsets.extend(
            offset + trajectory_offset
            for trajectory_offset in commit.trajectory_offsets[1:]
        )

    def build(self) -> RolloutCommit:
        return RolloutCommit(
            policy_version=self.policy_version,
            first_episode_id=self.first_episode_id,
            episode_count=self.episode_count,
            decision_handles=tuple(self.decision_handles),
            reward_after_step=tuple(self.reward_after_step),
            terminal_rewards=tuple(self.terminal_rewards),
            trajectory_team_indices=tuple(self.trajectory_team_indices),
            trajectory_offsets=tuple(self.trajectory_offsets),
        )
