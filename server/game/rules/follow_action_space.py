"""Exact semantic action space for follow plays."""

from __future__ import annotations

from dataclasses import dataclass, field

from server.foundation.result import Ok, Rejected
from server.game.rules.card_faces import (
    FaceCount,
    bind_face_counts,
    canonical_face_counts,
    face_count_width,
    face_sort_key,
)
from server.game.rules.cards import Card, Rank, Suit
from server.game.rules.follow_analysis import (
    FollowAnalysis,
    analyze_follow,
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
class FollowActionSpace:
    """Exact follow-play action space for one player decision."""

    analysis: FollowAnalysis
    hand_faces: tuple[FaceCount, ...]
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
        if face_count_width(prefix) >= self.analysis.lead_count:
            return ()
        last_face = prefix[-1].face if prefix else None
        result: list[FaceCount] = []
        selected_faces = {item.face for item in prefix}
        for available in self.hand_faces:
            if available.face in selected_faces:
                continue
            if last_face is not None and face_sort_key(
                available.face
            ) <= face_sort_key(last_face):
                continue
            for count in range(1, available.count + 1):
                candidate = FaceCount(available.face, count)
                if self.analysis.can_complete((*prefix, candidate)):
                    result.append(candidate)
        return tuple(result)

    def decode(
        self,
        face_counts: tuple[FaceCount, ...],
    ) -> Ok[list[Card]] | Rejected:
        """Bind and validate a complete semantic follow play."""
        if face_count_width(face_counts) != self.analysis.lead_count:
            return FollowActionSpaceRejected("跟牌张数不满足规则")
        cards_result = bind_face_counts(
            face_counts, self.analysis.hand_cards
        )
        if isinstance(cards_result, Rejected):
            return cards_result
        if not self.analysis.validate_cards(cards_result.value):
            return FollowActionSpaceRejected("跟牌不满足完整牌规")
        return Ok(value=cards_result.value)


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
        value=FollowActionSpace(
            analysis=analysis_result.value,
            hand_faces=canonical_face_counts(tuple(hand)),
        )
    )
