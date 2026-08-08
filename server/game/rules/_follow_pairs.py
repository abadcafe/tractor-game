"""Private pair-run planning for follow-play masks."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product as iterproduct

from server.game.rules._patterns import PatternAnalysis
from server.game.rules.cards.faces import (
    CardFace,
    FaceCount,
    face_count_width,
    face_sort_key,
)

type FaceCountMap = dict[CardFace, int]
type PairFaceSet = frozenset[CardFace]


@dataclass(frozen=True, slots=True)
class PairSegment:
    """One contiguous pair run from a hand pattern analysis."""

    level: int
    faces: tuple[CardFace, ...]


@dataclass(frozen=True, slots=True)
class FollowPairPlanner:
    """Decide whether same-suit follow prefixes can be completed."""

    same_suit_faces: tuple[FaceCount, ...]
    lead_pair_count: int
    pair_floor: int
    target_width: int
    decompositions: tuple[tuple[PairSegment, ...], ...]
    pair_plans: tuple[PairFaceSet, ...]
    _pair_faces: PairFaceSet = field(repr=False)

    def can_complete(self, prefix: tuple[FaceCount, ...]) -> bool:
        """Return whether a same-suit prefix has a legal completion."""
        target_width = self.target_width
        prefix_width = face_count_width(prefix)
        if prefix_width > target_width:
            return False
        if prefix_width == target_width:
            return self.pair_selection_is_valid(
                _pair_faces_from_counts(prefix)
            )
        prefix_counts = _face_count_map(prefix)
        last_face = _last_face(prefix)
        if not self._has_tractor_segment():
            return self._can_complete_without_tractor(
                prefix_counts=prefix_counts,
                prefix_width=prefix_width,
                last_face=last_face,
            )
        return self._can_complete_with_pair_plans(
            prefix_counts=prefix_counts,
            prefix_width=prefix_width,
            last_face=last_face,
        )

    @property
    def pair_faces(self) -> PairFaceSet:
        """Return all hand faces that can participate as pairs."""
        return self._pair_faces

    def has_tractor_segment(self) -> bool:
        """Return whether the hand contains a multi-pair segment."""
        return self._has_tractor_segment()

    def pair_selection_is_valid(
        self, selected_pair_faces: PairFaceSet
    ) -> bool:
        """Return whether selected pair faces satisfy priority."""
        if not selected_pair_faces <= self._pair_faces:
            return False
        if len(selected_pair_faces) < self.pair_floor:
            return False
        return any(
            _pair_selection_is_valid_for_decomposition(
                selected_pair_faces=selected_pair_faces,
                segments=segments,
                lead_pair_count=self.lead_pair_count,
                pair_floor=self.pair_floor,
            )
            for segments in self.decompositions
        )

    def _can_complete_without_tractor(
        self,
        *,
        prefix_counts: FaceCountMap,
        prefix_width: int,
        last_face: CardFace | None,
    ) -> bool:
        selected_pair_count = sum(
            1 for count in prefix_counts.values() if count == 2
        )
        future_pair_capacity = self._future_pair_capacity(
            selected=prefix_counts,
            last_face=last_face,
        )
        single_capacity = _single_capacity_after_last(
            available_faces=self.same_suit_faces,
            selected_faces=frozenset(prefix_counts),
            last_face=last_face,
        )
        for future_pair_count in range(future_pair_capacity + 1):
            final_pair_count = selected_pair_count + future_pair_count
            if final_pair_count < self.pair_floor:
                continue
            fixed_width = prefix_width + future_pair_count * 2
            if fixed_width > self.target_width:
                continue
            remaining = self.target_width - fixed_width
            if remaining <= single_capacity - future_pair_count:
                return True
        return False

    def _can_complete_with_pair_plans(
        self,
        *,
        prefix_counts: FaceCountMap,
        prefix_width: int,
        last_face: CardFace | None,
    ) -> bool:
        required_pairs = frozenset(
            face for face, count in prefix_counts.items() if count == 2
        )
        forbidden_pairs = frozenset(
            face for face, count in prefix_counts.items() if count == 1
        )
        if not required_pairs <= self._pair_faces:
            return False
        for plan in self.pair_plans:
            if not required_pairs <= plan:
                continue
            if plan & forbidden_pairs:
                continue
            if not _plan_respects_last_face(
                plan=plan,
                required_pairs=required_pairs,
                last_face=last_face,
            ):
                continue
            future_pairs = plan - required_pairs
            fixed_width = prefix_width + len(future_pairs) * 2
            if fixed_width > self.target_width:
                continue
            selected_faces = frozenset(prefix_counts) | plan
            single_capacity = _single_capacity_after_last(
                available_faces=self.same_suit_faces,
                selected_faces=selected_faces,
                last_face=last_face,
            )
            if self.target_width - fixed_width <= single_capacity:
                return True
        return False

    def _has_tractor_segment(self) -> bool:
        return any(
            len(segment.faces) > 1
            for segments in self.decompositions
            for segment in segments
        )

    def _future_pair_capacity(
        self,
        *,
        selected: FaceCountMap,
        last_face: CardFace | None,
    ) -> int:
        capacity = 0
        for face in self._pair_faces:
            if face in selected:
                continue
            if last_face is not None and face_sort_key(face) <= (
                face_sort_key(last_face)
            ):
                continue
            capacity += 1
        return capacity


def build_follow_pair_planner(
    *,
    hand_analysis: PatternAnalysis | None,
    same_suit_faces: tuple[FaceCount, ...],
    lead_pair_count: int,
    pair_floor: int,
    target_width: int,
) -> FollowPairPlanner:
    """Build a pair-run planner from every valid hand decomposition."""
    decompositions = (
        ((),)
        if hand_analysis is None
        else _pair_segment_decompositions(hand_analysis)
    )
    pair_plans = {
        plan
        for segments in decompositions
        for plan in _valid_pair_plans(
            segments=segments,
            hand_pairs_by_level=_hand_pairs_by_level(segments),
            lead_pair_count=lead_pair_count,
            pair_floor=pair_floor,
            target_width=target_width,
        )
    }
    return FollowPairPlanner(
        same_suit_faces=same_suit_faces,
        lead_pair_count=lead_pair_count,
        pair_floor=pair_floor,
        target_width=target_width,
        decompositions=decompositions,
        pair_plans=tuple(sorted(pair_plans, key=_pair_plan_sort_key)),
        _pair_faces=(
            frozenset()
            if hand_analysis is None
            else frozenset(hand_analysis.pair_faces)
        ),
    )


def _pair_segment_decompositions(
    analysis: PatternAnalysis,
) -> tuple[tuple[PairSegment, ...], ...]:
    run_options = tuple(
        tuple(iterproduct(*run.tiers)) for run in analysis.tractor_runs
    )
    selected_runs = iterproduct(*run_options) if run_options else ((),)
    result: list[tuple[PairSegment, ...]] = []
    for selected in selected_runs:
        used = frozenset(face for run in selected for face in run)
        tractor_segments = tuple(
            PairSegment(level=len(run), faces=tuple(run))
            for run in selected
        )
        pair_segments = tuple(
            PairSegment(level=1, faces=(face,))
            for face in analysis.pair_faces
            if face not in used
        )
        decomposition = (*tractor_segments, *pair_segments)
        if decomposition not in result:
            result.append(decomposition)
    return tuple(result)


def _hand_pairs_by_level(
    segments: tuple[PairSegment, ...],
) -> tuple[tuple[int, int], ...]:
    counts: dict[int, int] = {}
    for segment in segments:
        counts[segment.level] = counts.get(segment.level, 0) + len(
            segment.faces
        )
    return tuple(sorted(counts.items(), reverse=True))


def _pair_selection_is_valid_for_decomposition(
    *,
    selected_pair_faces: PairFaceSet,
    segments: tuple[PairSegment, ...],
    lead_pair_count: int,
    pair_floor: int,
) -> bool:
    if len(selected_pair_faces) < pair_floor:
        return False
    hand_pairs_by_level = _hand_pairs_by_level(segments)
    if not _pair_priority_is_satisfied(
        selected_pair_faces=selected_pair_faces,
        segments=segments,
        hand_pairs_by_level=hand_pairs_by_level,
        lead_pair_count=lead_pair_count,
    ):
        return False
    for segment in segments:
        if len(segment.faces) < 2:
            continue
        positions = tuple(
            index
            for index, face in enumerate(segment.faces)
            if face in selected_pair_faces
        )
        if not positions or len(positions) == len(segment.faces):
            continue
        if positions[-1] - positions[0] != len(positions) - 1:
            return False
    return True


def _valid_pair_plans(
    *,
    segments: tuple[PairSegment, ...],
    hand_pairs_by_level: tuple[tuple[int, int], ...],
    lead_pair_count: int,
    pair_floor: int,
    target_width: int,
) -> tuple[PairFaceSet, ...]:
    if not any(len(segment.faces) > 1 for segment in segments):
        return ()
    max_pair_count = target_width // 2
    result: list[PairFaceSet] = []
    _collect_pair_plans(
        segments=segments,
        hand_pairs_by_level=hand_pairs_by_level,
        lead_pair_count=lead_pair_count,
        pair_floor=pair_floor,
        max_pair_count=max_pair_count,
        index=0,
        selected=frozenset(),
        result=result,
    )
    return tuple(sorted(result, key=_pair_plan_sort_key))


def _collect_pair_plans(
    *,
    segments: tuple[PairSegment, ...],
    hand_pairs_by_level: tuple[tuple[int, int], ...],
    lead_pair_count: int,
    pair_floor: int,
    max_pair_count: int,
    index: int,
    selected: PairFaceSet,
    result: list[PairFaceSet],
) -> None:
    if len(selected) > max_pair_count:
        return
    if index == len(segments):
        if len(selected) < pair_floor:
            return
        if not _pair_priority_is_satisfied(
            selected_pair_faces=selected,
            segments=segments,
            hand_pairs_by_level=hand_pairs_by_level,
            lead_pair_count=lead_pair_count,
        ):
            return
        result.append(selected)
        return
    segment = segments[index]
    for option in _segment_options(segment):
        _collect_pair_plans(
            segments=segments,
            hand_pairs_by_level=hand_pairs_by_level,
            lead_pair_count=lead_pair_count,
            pair_floor=pair_floor,
            max_pair_count=max_pair_count,
            index=index + 1,
            selected=selected | option,
            result=result,
        )


def _segment_options(segment: PairSegment) -> tuple[PairFaceSet, ...]:
    options: list[PairFaceSet] = [frozenset()]
    face_count = len(segment.faces)
    for width in range(1, face_count + 1):
        for start in range(face_count - width + 1):
            options.append(
                frozenset(segment.faces[start : start + width])
            )
    return tuple(options)


def _pair_priority_is_satisfied(
    *,
    selected_pair_faces: PairFaceSet,
    segments: tuple[PairSegment, ...],
    hand_pairs_by_level: tuple[tuple[int, int], ...],
    lead_pair_count: int,
) -> bool:
    face_level: dict[CardFace, int] = {}
    for segment in segments:
        for face in segment.faces:
            face_level[face] = segment.level
    played_by_level: dict[int, int] = {}
    for face in selected_pair_faces:
        level = face_level[face]
        played_by_level[level] = played_by_level.get(level, 0) + 1

    remaining_needed = lead_pair_count
    levels = sorted(
        {level for level, _count in hand_pairs_by_level}
        | set(played_by_level),
        reverse=True,
    )
    hand_by_level = dict(hand_pairs_by_level)
    for level in levels:
        hand_count = hand_by_level.get(level, 0)
        played_count = played_by_level.get(level, 0)
        if hand_count == 0:
            continue
        expected = min(hand_count, remaining_needed)
        if played_count < expected:
            return False
        remaining_needed -= played_count
    return True


def _pair_plan_sort_key(
    plan: PairFaceSet,
) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((face_sort_key(face) for face in plan)))


def _pair_faces_from_counts(
    face_counts: tuple[FaceCount, ...],
) -> PairFaceSet:
    return frozenset(
        item.face for item in face_counts if item.count == 2
    )


def _face_count_map(face_counts: tuple[FaceCount, ...]) -> FaceCountMap:
    return {item.face: item.count for item in face_counts}


def _last_face(prefix: tuple[FaceCount, ...]) -> CardFace | None:
    if not prefix:
        return None
    return prefix[-1].face


def _plan_respects_last_face(
    *,
    plan: PairFaceSet,
    required_pairs: PairFaceSet,
    last_face: CardFace | None,
) -> bool:
    if last_face is None:
        return True
    last_key = face_sort_key(last_face)
    for face in plan:
        if face in required_pairs:
            continue
        if face_sort_key(face) <= last_key:
            return False
    return True


def _single_capacity_after_last(
    *,
    available_faces: tuple[FaceCount, ...],
    selected_faces: PairFaceSet,
    last_face: CardFace | None,
) -> int:
    capacity = 0
    for available in available_faces:
        if available.face in selected_faces:
            continue
        if last_face is not None and face_sort_key(
            available.face
        ) <= face_sort_key(last_face):
            continue
        capacity += 1
    return capacity
