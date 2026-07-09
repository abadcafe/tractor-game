"""Tests for worker-side return commit construction."""

from __future__ import annotations

from server.training.policy_sampling import DecisionHandle
from server.training.returns import (
    terminal_return_commit,
)
from server.training.semantic_actions import (
    GeneratedAction,
    SemanticArgument,
    SemanticArgumentTrace,
)
from server.training.trajectory import DecisionStep


def test_terminal_return_commit_assigns_team_returns() -> None:
    commit = terminal_return_commit(
        policy_version=4,
        episode_id=9,
        steps=(
            _step(player_index=0, row_index=0),
            _step(player_index=1, row_index=1),
            _step(player_index=2, row_index=2),
        ),
        team0_reward=1.5,
        team1_reward=-1.5,
    )

    assert commit.policy_version == 4
    assert commit.first_episode_id == 9
    assert commit.episode_count == 1
    assert commit.row_indices == (0, 2, 1)
    assert commit.step_counts == (1, 1, 1)
    assert commit.return_values == (1.5, 1.5, -1.5)


def _step(*, player_index: int, row_index: int) -> DecisionStep:
    return DecisionStep(
        player_index=player_index,
        seq=row_index,
        action=GeneratedAction(
            action_kind="pass",
            message_type="play",
            face_counts=(),
            semantic_trace=SemanticArgumentTrace(
                arguments=(SemanticArgument("pass"),)
            ),
            is_pass=True,
        ),
        decision_handle=_handle(
            model_rank_index=0,
            policy_version=4,
            row_index=row_index,
        ),
        choice_count=1,
    )


def _handle(
    *,
    model_rank_index: int,
    policy_version: int,
    row_index: int,
) -> DecisionHandle:
    return DecisionHandle(
        model_rank_index=model_rank_index,
        policy_version=policy_version,
        row_index=row_index,
    )
