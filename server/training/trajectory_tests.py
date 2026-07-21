"""Tests for training trajectory records."""

from __future__ import annotations

from server.training.policy_sampling import DecisionHandle
from server.training.semantic_actions import (
    ActionChoice,
    ActionTrace,
)
from server.training.semantic_actions.values import GeneratedAction
from server.training.trajectory import DecisionStep, TrajectoryRecorder


def test_decision_step_records_replay_handle() -> None:
    decision = _decision_step(choice=ActionChoice("pass"))

    assert decision.decision_handle == DecisionHandle(
        model_rank_index=0,
        policy_version=3,
        row_index=7,
    )


def test_trajectory_recorder_appends_and_clears_steps() -> None:
    recorder = TrajectoryRecorder()
    first = _decision_step(choice=ActionChoice("pass"))
    second = _decision_step(choice=ActionChoice("finish"))

    recorder.append(first)
    recorder.append(second)
    assert recorder.steps() == (first, second)
    recorder.clear()
    assert recorder.steps() == ()


def _decision_step(*, choice: ActionChoice) -> DecisionStep:
    trace = ActionTrace(choices=(choice,))
    return DecisionStep(
        player_index=0,
        seq=1,
        action=GeneratedAction(
            action_kind="pass" if choice.kind == "pass" else "bid",
            message_type="bid",
            face_counts=(),
            trace=trace,
            is_pass=choice.kind == "pass",
        ),
        decision_handle=DecisionHandle(
            model_rank_index=0,
            policy_version=3,
            row_index=7,
        ),
        choice_count=1,
    )
