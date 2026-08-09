"""Complete, per-seat rollout trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from server.game import Partnership, Seat, partnership_of
from server.policy_model.actions import GeneratedAction
from server.training.rollout_inference.samples.records import (
    DecisionHandle,
)


@dataclass(frozen=True, slots=True)
class TrainableDecision:
    """One accepted non-forced model decision."""

    seat: Seat
    seq: int
    action: GeneratedAction
    replay: DecisionHandle
    trace_step_count: int
    legal_choice_count: int
    scored_choice_step_count: int

    def __post_init__(self) -> None:
        assert self.seq >= 0
        assert self.trace_step_count > 0
        assert self.legal_choice_count > 1
        assert self.scored_choice_step_count > 0


def _decision_list() -> list[TrainableDecision]:
    return []


@dataclass(slots=True)
class SeatTrajectoryRecorder:
    """Append-only round trajectory for exactly one model seat."""

    seat: Seat
    _decisions: list[TrainableDecision] = field(
        default_factory=_decision_list
    )

    def append(self, decision: TrainableDecision) -> None:
        assert decision.seat == self.seat
        if self._decisions:
            assert decision.seq > self._decisions[-1].seq
        self._decisions.append(decision)

    def finish(self) -> SeatTrajectory:
        """Freeze the current round without reordering decisions."""
        return SeatTrajectory(
            seat=self.seat,
            decisions=tuple(self._decisions),
        )

    def clear(self) -> None:
        self._decisions.clear()


@dataclass(frozen=True, slots=True)
class SeatTrajectory:
    """Chronological decisions observed from one seat only."""

    seat: Seat
    decisions: tuple[TrainableDecision, ...]

    def __post_init__(self) -> None:
        assert all(item.seat == self.seat for item in self.decisions)
        assert all(
            left.seq < right.seq
            for left, right in zip(
                self.decisions,
                self.decisions[1:],
                strict=False,
            )
        )


@dataclass(frozen=True, slots=True)
class CompletedRoundTrajectory:
    """Atomic terminal rollout unit for one model partnership."""

    policy_version: int
    round_id: int
    model_partnership: Partnership
    seats: tuple[SeatTrajectory, SeatTrajectory]
    terminal_reward: float

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert self.round_id >= 0
        assert math.isfinite(self.terminal_reward)
        assert self.seats[0].seat != self.seats[1].seat
        assert all(
            partnership_of(item.seat) == self.model_partnership
            for item in self.seats
        )
        assert self.seats[0].seat.value < self.seats[1].seat.value
        for decision in self.decisions():
            assert decision.replay.policy_version == self.policy_version

    def decisions(self) -> tuple[TrainableDecision, ...]:
        """Return rows grouped by actor, never interleaved."""
        return self.seats[0].decisions + self.seats[1].decisions

    def decision_count(self) -> int:
        """Return the number of trainable policy decisions."""
        return sum(len(item.decisions) for item in self.seats)

    def trajectory_count(self) -> int:
        """Return non-empty actor trajectories in this round."""
        return sum(bool(item.decisions) for item in self.seats)


__all__ = (
    "CompletedRoundTrajectory",
    "SeatTrajectory",
    "SeatTrajectoryRecorder",
    "TrainableDecision",
)
