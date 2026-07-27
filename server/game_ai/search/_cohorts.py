"""Independent rollout lanes over shared branches."""

from __future__ import annotations

from dataclasses import dataclass

from server.training.semantic_actions import GeneratedAction
from server.training.semantic_actions.choices import (
    action_trace_choice_ids,
)

from ..branch import EngineBranch
from ..model import SampledAction


@dataclass(frozen=True, slots=True)
class RolloutLane:
    """One statistically independent rollout carried by a cohort."""

    candidate_index: int
    particle_weight: float
    draw_ordinal: int

    def __post_init__(self) -> None:
        assert self.candidate_index >= 0
        assert self.particle_weight > 0.0
        assert self.draw_ordinal >= 0


@dataclass(frozen=True, slots=True)
class RolloutCohort:
    """An engine branch shared by equivalent rollout lanes."""

    branch: EngineBranch
    lanes: tuple[RolloutLane, ...]

    def __post_init__(self) -> None:
        assert self.lanes


@dataclass(frozen=True, slots=True)
class ActionLaneGroup:
    """Lanes from one cohort that sampled the same semantic action."""

    action: GeneratedAction
    lanes: tuple[RolloutLane, ...]

    def __post_init__(self) -> None:
        assert self.lanes


def group_sampled_lanes(
    *,
    cohort: RolloutCohort,
    sampled: tuple[SampledAction, ...],
) -> tuple[ActionLaneGroup, ...]:
    """Group independent lanes by their sampled semantic action."""
    assert len(sampled) == len(cohort.lanes)
    actions: dict[tuple[int, ...], GeneratedAction] = {}
    grouped: dict[tuple[int, ...], list[RolloutLane]] = {}
    for lane, evaluation in zip(cohort.lanes, sampled, strict=True):
        key = action_trace_choice_ids(evaluation.action.trace)
        existing = actions.get(key)
        if existing is None:
            actions[key] = evaluation.action
            grouped[key] = [lane]
        else:
            assert existing == evaluation.action
            grouped[key].append(lane)
    return tuple(
        ActionLaneGroup(
            action=actions[key],
            lanes=tuple(grouped[key]),
        )
        for key in grouped
    )


__all__ = (
    "ActionLaneGroup",
    "RolloutCohort",
    "RolloutLane",
    "group_sampled_lanes",
)
