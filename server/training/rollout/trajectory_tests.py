"""Black-box tests for full-round rollout trajectories."""

from __future__ import annotations

from server.game import Partnership, Seat
from server.policy_model.actions import (
    ActionChoice,
    ActionTrace,
    GeneratedAction,
)
from server.training.rollout import (
    CompletedRoundTrajectory,
    SeatTrajectory,
    TrainableDecision,
)
from server.training.rollout_inference.samples import DecisionHandle


def test_round_groups_chronological_decisions_by_model_seat() -> None:
    first = _decision(seat=Seat.A, seq=3, row=7)
    second = _decision(seat=Seat.A, seq=9, row=8)
    partner = _decision(seat=Seat.C, seq=6, row=11)
    trajectory = CompletedRoundTrajectory(
        policy_version=4,
        round_id=13,
        model_partnership=Partnership.FIRST,
        seats=(
            SeatTrajectory(seat=Seat.A, decisions=(first, second)),
            SeatTrajectory(seat=Seat.C, decisions=(partner,)),
        ),
        terminal_reward=2.0,
    )

    assert trajectory.decisions() == (first, second, partner)
    assert trajectory.decision_count() == 3
    assert trajectory.trajectory_count() == 2


def test_completed_round_counts_only_nonempty_seat_trajectories() -> (
    None
):
    decision = _decision(seat=Seat.B, seq=5, row=2)
    trajectory = CompletedRoundTrajectory(
        policy_version=4,
        round_id=1,
        model_partnership=Partnership.SECOND,
        seats=(
            SeatTrajectory(seat=Seat.B, decisions=(decision,)),
            SeatTrajectory(seat=Seat.D, decisions=()),
        ),
        terminal_reward=-1.0,
    )

    assert trajectory.decision_count() == 1
    assert trajectory.trajectory_count() == 1


def _decision(*, seat: Seat, seq: int, row: int) -> TrainableDecision:
    return TrainableDecision(
        seat=seat,
        seq=seq,
        action=GeneratedAction(
            action_kind="pass",
            message_type="bid",
            face_counts=(),
            trace=ActionTrace(choices=(ActionChoice("pass"),)),
            is_pass=True,
        ),
        replay=DecisionHandle(
            model_rank_index=0,
            policy_version=4,
            row_index=row,
        ),
        trace_step_count=1,
        legal_choice_count=2,
        scored_choice_step_count=1,
    )
