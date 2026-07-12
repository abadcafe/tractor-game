"""Pair-run planning for exact follow-play action masks."""

from __future__ import annotations

from dataclasses import dataclass, field

from server.game.rules.card_faces import (
    CardFace,
    FaceCount,
    card_face,
    face_count_width,
    face_sort_key,
)
from server.game.rules.cards import Card
from server.game.rules.types import SubPlay

type FaceCountMap = dict[CardFace, int]
type PairFaceSet = frozenset[CardFace]


@dataclass(frozen=True, slots=True)
class PairSegment:
    """One contiguous pair run from a decomposed hand sub-play."""

    level: int
    faces: tuple[CardFace, ...]


@dataclass(frozen=True, slots=True)
class FollowPairPlanner:
    """Decide whether same-suit follow prefixes can be completed."""

    same_suit_faces: tuple[FaceCount, ...]
    lead_pair_count: int
    pair_floor: int
    target_width: int
    segments: tuple[PairSegment, ...]
    hand_pairs_by_level: tuple[tuple[int, int], ...]
    pair_plans: tuple[PairFaceSet, ...]
    _pair_faces: PairFaceSet = field(init=False, repr=False)
    _face_level: dict[CardFace, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        pair_faces = frozenset(
            face for segment in self.segments for face in segment.faces
        )
        face_level: dict[CardFace, int] = {}
        for segment in self.segments:
            for face in segment.faces:
                face_level[face] = segment.level
        object.__setattr__(self, "_pair_faces", pair_faces)
        object.__setattr__(self, "_face_level", face_level)

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
        if not self._pair_priority_is_satisfied(selected_pair_faces):
            return False
        return self._tractor_continuity_is_satisfied(
            selected_pair_faces
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
        return any(len(segment.faces) > 1 for segment in self.segments)

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

    def _pair_priority_is_satisfied(
        self, selected_pair_faces: PairFaceSet
    ) -> bool:
        played_by_level: dict[int, int] = {}
        for face in selected_pair_faces:
            level = self._face_level[face]
            played_by_level[level] = played_by_level.get(level, 0) + 1

        remaining_needed = self.lead_pair_count
        levels = sorted(
            {level for level, _count in self.hand_pairs_by_level}
            | set(played_by_level),
            reverse=True,
        )
        hand_by_level = dict(self.hand_pairs_by_level)
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

    def _tractor_continuity_is_satisfied(
        self, selected_pair_faces: PairFaceSet
    ) -> bool:
        for segment in self.segments:
            if len(segment.faces) < 2:
                continue
            positions = tuple(
                index
                for index, face in enumerate(segment.faces)
                if face in selected_pair_faces
            )
            if not positions:
                continue
            if len(positions) == len(segment.faces):
                continue
            if positions[-1] - positions[0] != len(positions) - 1:
                return False
        return True


def build_follow_pair_planner(
    *,
    hand_subs: list[SubPlay],
    same_suit_faces: tuple[FaceCount, ...],
    lead_pair_count: int,
    pair_floor: int,
    target_width: int,
) -> FollowPairPlanner:
    """Build a pair-run planner from decomposed hand structure."""
    segments = _pair_segments(hand_subs)
    hand_pairs_by_level = _hand_pairs_by_level(segments)
    return FollowPairPlanner(
        same_suit_faces=same_suit_faces,
        lead_pair_count=lead_pair_count,
        pair_floor=pair_floor,
        target_width=target_width,
        segments=segments,
        hand_pairs_by_level=hand_pairs_by_level,
        pair_plans=_valid_pair_plans(
            segments=segments,
            hand_pairs_by_level=hand_pairs_by_level,
            lead_pair_count=lead_pair_count,
            pair_floor=pair_floor,
            target_width=target_width,
        ),
    )


def played_pair_faces(
    played_suit_cards: tuple[Card, ...],
) -> PairFaceSet:
    """Return semantic faces that are played as pairs."""
    counts: dict[CardFace, int] = {}
    result: set[CardFace] = set()
    for card in played_suit_cards:
        face = card_face(card)
        count = counts.get(face, 0) + 1
        counts[face] = count
        if count == 2:
            result.add(face)
    return frozenset(result)


def _pair_segments(
    hand_subs: list[SubPlay],
) -> tuple[PairSegment, ...]:
    segments: list[PairSegment] = []
    for sub in hand_subs:
        if sub.pair_count == 0:
            continue
        faces = _sub_pair_faces(sub)
        assert len(faces) == sub.pair_count
        segments.append(PairSegment(level=sub.pair_count, faces=faces))
    return tuple(segments)


def _sub_pair_faces(sub: SubPlay) -> tuple[CardFace, ...]:
    counts: dict[CardFace, int] = {}
    result: list[CardFace] = []
    for card in sub.cards:
        face = card_face(card)
        count = counts.get(face, 0) + 1
        counts[face] = count
        if count == 2:
            result.append(face)
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
