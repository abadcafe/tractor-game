"""Private exact semantic action space for follow plays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from server.foundation.result import Ok, Rejected
from server.game.rules._follow_analysis import (
    FollowAnalysis,
    analyze_follow,
)
from server.game.rules.cards import Card, Rank, Suit
from server.game.rules.cards.faces import (
    CardFace,
    FaceCount,
    bind_face_counts,
    canonical_face_counts,
    face_count_width,
    face_sort_key,
)

type FollowPrefix = tuple[FaceCount, ...]
type AllowedNextCache = dict[FollowPrefix, tuple[FaceCount, ...]]


class FollowActionSpaceRejected(Rejected):
    """Follow action space rejected a semantic selection."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)


def _allowed_next_cache() -> AllowedNextCache:
    return {}


@dataclass(frozen=True, slots=True)
class FollowSelectionPlan:
    """Complete public rule plan for vectorized follow selection."""

    hand_cards: tuple[Card, ...]
    trump_suit: Suit | None
    trump_rank: Rank
    lead_count: int
    lead_effective_suit: Suit | Literal["trump"]
    required_same_suit_count: int
    pair_faces: tuple[CardFace, ...]
    pair_plans: tuple[tuple[CardFace, ...], ...]
    has_tractor: bool
    pair_floor: int


class FollowActionSpace(Protocol):
    """Opaque exact follow-play action space for one decision."""

    def allowed_next(
        self,
        prefix: tuple[FaceCount, ...],
    ) -> tuple[FaceCount, ...]:
        """Return next face-count choices that can complete legally."""
        ...

    def decode(
        self,
        face_counts: tuple[FaceCount, ...],
    ) -> Ok[list[Card]] | Rejected:
        """Bind and validate a complete semantic follow play."""
        ...

    def selection_plan(self) -> FollowSelectionPlan:
        """Return the public rule plan needed by vectorized sampling."""
        ...


@dataclass(frozen=True, slots=True)
class _CompiledFollowActionSpace:
    """Private compiled implementation of the follow action space."""

    _analysis: FollowAnalysis
    _hand_faces: tuple[FaceCount, ...]
    _allowed_next_cache: AllowedNextCache = field(
        default_factory=_allowed_next_cache,
        init=False,
        repr=False,
    )

    def allowed_next(
        self,
        prefix: tuple[FaceCount, ...],
    ) -> tuple[FaceCount, ...]:
        """Return next face-count choices that can complete legally."""
        cached = self._allowed_next_cache.get(prefix)
        if cached is not None:
            return cached
        allowed = self._compute_allowed_next(prefix)
        self._allowed_next_cache[prefix] = allowed
        return allowed

    def _compute_allowed_next(
        self,
        prefix: tuple[FaceCount, ...],
    ) -> tuple[FaceCount, ...]:
        if face_count_width(prefix) >= self._analysis.lead_count:
            return ()
        last_face = prefix[-1].face if prefix else None
        result: list[FaceCount] = []
        selected_faces = {item.face for item in prefix}
        for available in self._hand_faces:
            if available.face in selected_faces:
                continue
            if last_face is not None and face_sort_key(
                available.face
            ) <= face_sort_key(last_face):
                continue
            for count in range(1, available.count + 1):
                candidate = FaceCount(available.face, count)
                if self._analysis.can_complete((*prefix, candidate)):
                    result.append(candidate)
        return tuple(result)

    def decode(
        self,
        face_counts: tuple[FaceCount, ...],
    ) -> Ok[list[Card]] | Rejected:
        """Bind and validate a complete semantic follow play."""
        if face_count_width(face_counts) != self._analysis.lead_count:
            return FollowActionSpaceRejected("跟牌张数不满足规则")
        cards_result = bind_face_counts(
            face_counts, self._analysis.hand_cards
        )
        if isinstance(cards_result, Rejected):
            return cards_result
        if not self._analysis.validate_cards(cards_result.value):
            return FollowActionSpaceRejected("跟牌不满足完整牌规")
        return Ok(value=cards_result.value)

    def selection_plan(self) -> FollowSelectionPlan:
        """Return the public rule plan needed by vectorized sampling."""
        planner = self._analysis.pair_planner
        pair_faces = tuple(
            sorted(planner.pair_faces, key=face_sort_key)
        )
        pair_plans = tuple(
            sorted(
                (
                    tuple(sorted(plan, key=face_sort_key))
                    for plan in planner.pair_plans
                ),
                key=lambda plan: tuple(
                    face_sort_key(face) for face in plan
                ),
            )
        )
        return FollowSelectionPlan(
            hand_cards=self._analysis.hand_cards,
            trump_suit=self._analysis.trump_suit,
            trump_rank=self._analysis.trump_rank,
            lead_count=self._analysis.lead_count,
            lead_effective_suit=self._analysis.lead_effective_suit,
            required_same_suit_count=(
                self._analysis.required_same_suit_count
            ),
            pair_faces=pair_faces,
            pair_plans=pair_plans,
            has_tractor=planner.has_tractor_segment(),
            pair_floor=planner.pair_floor,
        )


def build_follow_action_space(
    *,
    hand: list[Card],
    lead_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> Ok[FollowActionSpace] | Rejected:
    """Build an exact semantic follow action space."""
    analysis_result = analyze_follow(
        hand=hand,
        lead_cards=lead_cards,
        trump_suit=trump_suit,
        trump_rank=trump_rank,
    )
    if isinstance(analysis_result, Rejected):
        return analysis_result
    return Ok(
        value=_CompiledFollowActionSpace(
            _analysis=analysis_result.value,
            _hand_faces=canonical_face_counts(tuple(hand)),
        )
    )
