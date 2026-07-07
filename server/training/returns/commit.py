"""Per-decision return targets produced by self-play workers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from server.sm.constants import get_team_index
from server.training.policy_sampling.records import DecisionHandle
from server.training.trajectory import DecisionStep


@dataclass(frozen=True, slots=True)
class ReturnCommit:
    """Return targets keyed by model-rank replay handles."""

    policy_version: int
    first_episode_id: int
    episode_count: int
    decision_handles: tuple[DecisionHandle, ...]
    return_values: tuple[float, ...]

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert self.first_episode_id >= 0
        assert self.episode_count >= 0
        assert len(self.decision_handles) == len(self.return_values)
        assert all(
            handle.policy_version == self.policy_version
            for handle in self.decision_handles
        )
        assert all(math.isfinite(value) for value in self.return_values)

    def sample_count(self) -> int:
        """Return committed trainable decision count."""
        return len(self.decision_handles)

    def is_empty(self) -> bool:
        """Return whether this commit has no trainable decisions."""
        return self.sample_count() == 0


def terminal_return_commit(
    *,
    policy_version: int,
    episode_id: int,
    steps: tuple[DecisionStep, ...],
    team0_reward: float,
    team1_reward: float,
) -> ReturnCommit:
    """Build per-decision returns for one completed round."""
    assert policy_version >= 0
    assert episode_id >= 0
    assert math.isfinite(team0_reward)
    assert math.isfinite(team1_reward)
    assert team0_reward + team1_reward == 0.0
    builder = _ReturnCommitBuilder(
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
                steps=team_steps,
                terminal_reward=reward,
            )
    return builder.build()


@dataclass(slots=True)
class _ReturnCommitBuilder:
    policy_version: int
    first_episode_id: int
    episode_count: int
    decision_handles: list[DecisionHandle]
    return_values: list[float]

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
        self.return_values = []

    def append_terminal_trajectory(
        self,
        *,
        steps: tuple[DecisionStep, ...],
        terminal_reward: float,
    ) -> None:
        assert steps
        suffix_return = terminal_reward
        returns_reversed: list[float] = []
        for _step in reversed(steps):
            returns_reversed.append(suffix_return)
        for step, return_value in zip(
            steps, reversed(returns_reversed), strict=True
        ):
            self.append_decision(
                handle=step.decision_handle,
                return_value=return_value,
            )

    def append_decision(
        self,
        *,
        handle: DecisionHandle,
        return_value: float,
    ) -> None:
        assert handle.policy_version == self.policy_version
        assert math.isfinite(return_value)
        self.decision_handles.append(handle)
        self.return_values.append(return_value)

    def build(self) -> ReturnCommit:
        return ReturnCommit(
            policy_version=self.policy_version,
            first_episode_id=self.first_episode_id,
            episode_count=self.episode_count,
            decision_handles=tuple(self.decision_handles),
            return_values=tuple(self.return_values),
        )
