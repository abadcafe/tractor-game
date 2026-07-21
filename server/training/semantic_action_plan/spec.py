"""Queue-safe compiled semantic action plan data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from server.game.rules.card_faces import MAX_FACE_COUNT
from server.training.semantic_actions.choices import (
    ACTION_CHOICE_COUNT,
    CARD_FACE_COUNT,
    MAX_ACTION_STEPS,
)

ACTION_FACE_COUNT: int = CARD_FACE_COUNT
MAX_TRACE_COUNT: int = 128
MAX_PAIR_PLAN_COUNT: int = 64

type CompiledActionKind = Literal[
    "empty",
    "trace_set",
    "discard",
    "lead_play",
    "follow_play",
]


@dataclass(frozen=True, slots=True)
class FacePlan:
    """Per-face static constraints for one decision."""

    available_counts: tuple[int, ...]
    effective_suits: tuple[int, ...]
    same_suit_mask: tuple[bool, ...]
    off_suit_mask: tuple[bool, ...]
    pair_face_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        assert len(self.available_counts) == ACTION_FACE_COUNT
        assert len(self.effective_suits) == ACTION_FACE_COUNT
        assert len(self.same_suit_mask) == ACTION_FACE_COUNT
        assert len(self.off_suit_mask) == ACTION_FACE_COUNT
        assert len(self.pair_face_mask) == ACTION_FACE_COUNT
        assert all(
            0 <= value <= MAX_FACE_COUNT
            for value in self.available_counts
        )
        assert all(value >= -1 for value in self.effective_suits)


@dataclass(frozen=True, slots=True)
class PairPlanConstraints:
    """Follow-play pair/tractor completion constraints."""

    pair_plan_masks: tuple[tuple[bool, ...], ...]
    has_tractor: bool
    pair_floor: int

    def __post_init__(self) -> None:
        assert self.pair_floor >= 0
        for row in self.pair_plan_masks:
            assert len(row) == ACTION_FACE_COUNT


@dataclass(frozen=True, slots=True)
class CompiledSelectionConstraints:
    """Dynamic selection constraints shared by discard/lead/follow."""

    min_select: int
    max_select: int
    exact_select: int | None
    required_same_suit_count: int
    lead_effective_suit: int
    face_plan: FacePlan
    pair_plan: PairPlanConstraints

    def __post_init__(self) -> None:
        assert self.min_select >= 0
        assert self.max_select >= self.min_select
        assert self.exact_select is None or self.exact_select >= 0
        assert self.required_same_suit_count >= 0
        assert self.lead_effective_suit >= -1


@dataclass(frozen=True, slots=True)
class CompiledActionTraceSet:
    """Closed set of complete traces for bid/stir style decisions."""

    traces: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        assert self.traces
        for trace in self.traces:
            assert trace
            assert len(trace) <= MAX_ACTION_STEPS
            for choice_id in trace:
                assert 0 <= choice_id < ACTION_CHOICE_COUNT


@dataclass(frozen=True, slots=True)
class CompiledActionSpec:
    """Compiled legal action constraints for one policy decision."""

    kind: CompiledActionKind
    trace_set: CompiledActionTraceSet | None
    selection: CompiledSelectionConstraints | None

    def __post_init__(self) -> None:
        if self.kind == "trace_set":
            assert self.trace_set is not None
            assert self.selection is None
            return
        if self.kind in ("discard", "lead_play", "follow_play"):
            assert self.trace_set is None
            assert self.selection is not None
            return
        assert self.kind == "empty"
        assert self.trace_set is None
        assert self.selection is None
