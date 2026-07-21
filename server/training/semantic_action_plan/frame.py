"""Flat semantic action plan frames for runtime sampling."""

from __future__ import annotations

from dataclasses import dataclass

from server.training.legal_actions import LegalActionIndex
from server.training.semantic_action_plan.compile import (
    compile_legal_action_spec,
)
from server.training.semantic_action_plan.spec import (
    ACTION_FACE_COUNT,
    CompiledActionSpec,
)
from server.training.semantic_actions.choices import MAX_ACTION_STEPS

ACTION_KIND_EMPTY = 0
ACTION_KIND_TRACE_SET = 1
ACTION_KIND_DISCARD = 2
ACTION_KIND_LEAD = 3
ACTION_KIND_FOLLOW = 4


@dataclass(frozen=True, slots=True)
class ActionPlanFrame:
    """Flat queue/device-friendly legal action constraints."""

    kind_code: int
    available_counts: tuple[int, ...]
    effective_suits: tuple[int, ...]
    same_suit_mask: tuple[bool, ...]
    off_suit_mask: tuple[bool, ...]
    pair_face_mask: tuple[bool, ...]
    min_select: int
    max_select: int
    exact_select: int
    required_same_suit_count: int
    pair_floor: int
    has_tractor: bool
    trace_choice_ids: tuple[tuple[int, ...], ...]
    pair_plan_masks: tuple[tuple[bool, ...], ...]

    def __post_init__(self) -> None:
        assert self.kind_code in (
            ACTION_KIND_EMPTY,
            ACTION_KIND_TRACE_SET,
            ACTION_KIND_DISCARD,
            ACTION_KIND_LEAD,
            ACTION_KIND_FOLLOW,
        )
        assert len(self.available_counts) == ACTION_FACE_COUNT
        assert len(self.effective_suits) == ACTION_FACE_COUNT
        assert len(self.same_suit_mask) == ACTION_FACE_COUNT
        assert len(self.off_suit_mask) == ACTION_FACE_COUNT
        assert len(self.pair_face_mask) == ACTION_FACE_COUNT
        assert self.min_select >= 0
        assert self.max_select >= self.min_select
        assert self.exact_select >= -1
        assert self.required_same_suit_count >= 0
        assert self.pair_floor >= 0
        for row in self.pair_plan_masks:
            assert len(row) == ACTION_FACE_COUNT


def compile_legal_action_frame(
    legal_action: LegalActionIndex,
) -> ActionPlanFrame:
    """Compile one legal action index into a flat runtime frame."""
    return action_plan_frame_from_spec(
        compile_legal_action_spec(legal_action)
    )


def action_plan_generation_step_count(
    action_plan: ActionPlanFrame,
) -> int:
    """Return the maximum choice count for one action plan."""
    if action_plan.kind_code == ACTION_KIND_EMPTY:
        return 1
    if action_plan.kind_code == ACTION_KIND_TRACE_SET:
        return max(len(trace) for trace in action_plan.trace_choice_ids)
    if action_plan.kind_code == ACTION_KIND_LEAD:
        assert action_plan.max_select + 1 <= MAX_ACTION_STEPS
        return max(action_plan.max_select + 1, 1)
    if action_plan.exact_select >= 0:
        assert action_plan.exact_select <= MAX_ACTION_STEPS
        return max(action_plan.exact_select, 1)
    assert action_plan.max_select <= MAX_ACTION_STEPS
    return max(action_plan.max_select, 1)


def action_plan_frame_from_spec(
    spec: CompiledActionSpec,
) -> ActionPlanFrame:
    """Flatten an internal compiled spec into a runtime frame."""
    selection = spec.selection
    trace_set = spec.trace_set
    return ActionPlanFrame(
        kind_code=_kind_code(spec),
        available_counts=tuple(0 for _ in range(ACTION_FACE_COUNT))
        if selection is None
        else selection.face_plan.available_counts,
        effective_suits=tuple(-1 for _ in range(ACTION_FACE_COUNT))
        if selection is None
        else selection.face_plan.effective_suits,
        same_suit_mask=tuple(False for _ in range(ACTION_FACE_COUNT))
        if selection is None
        else selection.face_plan.same_suit_mask,
        off_suit_mask=tuple(False for _ in range(ACTION_FACE_COUNT))
        if selection is None
        else selection.face_plan.off_suit_mask,
        pair_face_mask=tuple(False for _ in range(ACTION_FACE_COUNT))
        if selection is None
        else selection.face_plan.pair_face_mask,
        min_select=0 if selection is None else selection.min_select,
        max_select=0 if selection is None else selection.max_select,
        exact_select=-1
        if selection is None or selection.exact_select is None
        else selection.exact_select,
        required_same_suit_count=0
        if selection is None
        else selection.required_same_suit_count,
        pair_floor=0
        if selection is None
        else selection.pair_plan.pair_floor,
        has_tractor=False
        if selection is None
        else selection.pair_plan.has_tractor,
        trace_choice_ids=() if trace_set is None else trace_set.traces,
        pair_plan_masks=()
        if selection is None
        else selection.pair_plan.pair_plan_masks,
    )


def _kind_code(spec: CompiledActionSpec) -> int:
    if spec.kind == "empty":
        return ACTION_KIND_EMPTY
    if spec.kind == "trace_set":
        return ACTION_KIND_TRACE_SET
    if spec.kind == "discard":
        return ACTION_KIND_DISCARD
    if spec.kind == "lead_play":
        return ACTION_KIND_LEAD
    if spec.kind == "follow_play":
        return ACTION_KIND_FOLLOW
    raise AssertionError(spec.kind)
