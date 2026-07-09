"""Tests for training trajectory records."""

from __future__ import annotations

from server.training.policy_sampling import DecisionHandle
from server.training.semantic_actions import (
    SemanticArgument,
    SemanticArgumentTrace,
)
from server.training.semantic_actions.values import GeneratedAction
from server.training.trajectory import DecisionStep, TrajectoryRecorder


def test_decision_step_records_replay_handle() -> None:
    decision = _decision_step(argument=SemanticArgument("pass"))

    assert decision.decision_handle == DecisionHandle(
        model_rank_index=0,
        policy_version=3,
        row_index=7,
    )


def test_trajectory_recorder_appends_and_clears_steps() -> None:
    recorder = TrajectoryRecorder()
    first = _decision_step(argument=SemanticArgument("pass"))
    second = _decision_step(argument=SemanticArgument("stop"))

    recorder.append(first)
    recorder.append(second)
    assert recorder.steps() == (first, second)
    recorder.clear()
    assert recorder.steps() == ()


def _decision_step(*, argument: SemanticArgument) -> DecisionStep:
    trace = SemanticArgumentTrace(arguments=(argument,))
    return DecisionStep(
        player_index=0,
        seq=1,
        action=GeneratedAction(
            action_kind="pass" if argument.kind == "pass" else "bid",
            message_type="bid",
            face_counts=(),
            semantic_trace=trace,
            is_pass=argument.kind == "pass",
        ),
        decision_handle=DecisionHandle(
            model_rank_index=0,
            policy_version=3,
            row_index=7,
        ),
        choice_count=1,
    )
