"""Shared follow-play analysis for validation and action masking."""

from __future__ import annotations

from dataclasses import dataclass

from server.foundation.result import Ok, Rejected
from server.game.rules.card_faces import (
    CardFace,
    FaceCount,
    bind_face_counts,
    canonical_face_counts,
    face_count_width,
    face_sort_key,
)
from server.game.rules.cards import Card, Rank, Suit
from server.game.rules.decompose import decompose
from server.game.rules.follow_pair_planner import (
    FollowPairPlanner,
    build_follow_pair_planner,
    played_pair_faces,
)
from server.game.rules.ordering import effective_suit
from server.game.rules.rejections.play import EmptyLeadRejected
from server.game.rules.types import EffectiveSuit

type FaceCountMap = dict[CardFace, int]


@dataclass(frozen=True, slots=True)
class FollowAnalysis:
    """Compiled follow-play constraints for one hand and lead."""

    hand_cards: tuple[Card, ...]
    lead_cards: tuple[Card, ...]
    trump_suit: Suit | None
    trump_rank: Rank
    lead_count: int
    lead_effective_suit: EffectiveSuit
    same_suit_cards: tuple[Card, ...]
    off_suit_cards: tuple[Card, ...]
    same_suit_faces: tuple[FaceCount, ...]
    off_suit_faces: tuple[FaceCount, ...]
    required_same_suit_count: int
    lead_pair_count: int
    pair_floor: int
    pair_planner: FollowPairPlanner

    def validate_cards(self, played_cards: list[Card]) -> bool:
        """Return whether a complete physical follow play is legal."""
        if len(played_cards) != self.lead_count:
            return False
        if not played_cards:
            return False
        if not _cards_are_in_hand(played_cards, self.hand_cards):
            return False
        suit_in_played = [
            card
            for card in played_cards
            if effective_suit(card, self.trump_suit, self.trump_rank)
            == self.lead_effective_suit
        ]
        if len(suit_in_played) != self.required_same_suit_count:
            return False
        return self._verify_follow_sub_play_priority(
            tuple(suit_in_played)
        )

    def validate_face_counts(
        self, face_counts: tuple[FaceCount, ...]
    ) -> bool:
        """Return whether complete semantic face counts are legal."""
        cards_result = bind_face_counts(face_counts, self.hand_cards)
        if isinstance(cards_result, Rejected):
            return False
        return self.validate_cards(cards_result.value)

    def can_complete(
        self,
        prefix: tuple[FaceCount, ...],
    ) -> bool:
        """Return whether a prefix can still become a legal follow."""
        if not _prefix_is_ordered(prefix):
            return False
        if not _counts_available(prefix, self.hand_cards):
            return False
        selected_width = face_count_width(prefix)
        if selected_width > self.lead_count:
            return False

        same_prefix = tuple(
            item
            for item in prefix
            if _face_effective_suit(
                item.face, self.trump_suit, self.trump_rank
            )
            == self.lead_effective_suit
        )
        off_prefix = tuple(
            item for item in prefix if item not in same_prefix
        )
        same_width = face_count_width(same_prefix)
        off_width = selected_width - same_width
        required_off_count = (
            self.lead_count - self.required_same_suit_count
        )
        if same_width > self.required_same_suit_count:
            return False
        if off_width > required_off_count:
            return False
        last_face = _last_face(prefix)
        if self.required_same_suit_count == self.lead_count:
            if off_width > 0:
                return False
            return self._can_complete_same_suit(same_prefix)
        return self._can_complete_exhausting_suit(
            same_prefix=same_prefix,
            off_prefix=off_prefix,
            last_face=last_face,
        )

    def _verify_follow_sub_play_priority(
        self,
        played_suit_cards: tuple[Card, ...],
    ) -> bool:
        if not played_suit_cards:
            return True

        played_subs = decompose(
            list(played_suit_cards), self.trump_suit, self.trump_rank
        )
        played_pair_count = sum(sub.pair_count for sub in played_subs)
        if played_pair_count < self.pair_floor:
            return False

        selected_pair_faces = played_pair_faces(played_suit_cards)
        return self.pair_planner.pair_selection_is_valid(
            selected_pair_faces
        )

    def _can_complete_exhausting_suit(
        self,
        *,
        same_prefix: tuple[FaceCount, ...],
        off_prefix: tuple[FaceCount, ...],
        last_face: CardFace | None,
    ) -> bool:
        same_counts = _face_count_map(same_prefix)
        for available in self.same_suit_faces:
            selected_count = same_counts.get(available.face, 0)
            if selected_count > available.count:
                return False
            if (
                last_face is not None
                and face_sort_key(available.face)
                <= face_sort_key(last_face)
                and selected_count != available.count
            ):
                return False

        required_off_count = (
            self.lead_count - self.required_same_suit_count
        )
        off_width = face_count_width(off_prefix)
        if off_width > required_off_count:
            return False
        remaining_off = required_off_count - off_width
        return (
            _remaining_capacity_after_last(
                available_faces=self.off_suit_faces,
                selected=_face_count_map(off_prefix),
                last_face=last_face,
                pair_counts_allowed=True,
            )
            >= remaining_off
        )

    def _can_complete_same_suit(
        self, prefix: tuple[FaceCount, ...]
    ) -> bool:
        target_width = self.lead_count
        prefix_width = face_count_width(prefix)
        if prefix_width > target_width:
            return False
        if prefix_width == target_width:
            return self.validate_face_counts(prefix)
        return self.pair_planner.can_complete(prefix)


def analyze_follow(
    *,
    hand: list[Card],
    lead_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> Ok[FollowAnalysis] | Rejected:
    """Return compiled follow-play constraints."""
    if not lead_cards:
        return EmptyLeadRejected()

    lead_effective_suit = effective_suit(
        lead_cards[0], trump_suit, trump_rank
    )
    same_suit_cards = tuple(
        card
        for card in hand
        if effective_suit(card, trump_suit, trump_rank)
        == lead_effective_suit
    )
    off_suit_cards = tuple(
        card
        for card in hand
        if effective_suit(card, trump_suit, trump_rank)
        != lead_effective_suit
    )
    hand_subs = (
        decompose(list(same_suit_cards), trump_suit, trump_rank)
        if same_suit_cards
        else []
    )
    lead_subs = decompose(lead_cards, trump_suit, trump_rank)
    lead_pair_count = sum(sub.pair_count for sub in lead_subs)
    hand_pair_count = sum(sub.pair_count for sub in hand_subs)
    lead_count = len(lead_cards)
    same_suit_faces = canonical_face_counts(same_suit_cards)
    return Ok(
        value=FollowAnalysis(
            hand_cards=tuple(hand),
            lead_cards=tuple(lead_cards),
            trump_suit=trump_suit,
            trump_rank=trump_rank,
            lead_count=lead_count,
            lead_effective_suit=lead_effective_suit,
            same_suit_cards=same_suit_cards,
            off_suit_cards=off_suit_cards,
            same_suit_faces=same_suit_faces,
            off_suit_faces=canonical_face_counts(off_suit_cards),
            required_same_suit_count=min(
                len(same_suit_cards), lead_count
            ),
            lead_pair_count=lead_pair_count,
            pair_floor=min(hand_pair_count, lead_pair_count),
            pair_planner=build_follow_pair_planner(
                hand_subs=hand_subs,
                same_suit_faces=same_suit_faces,
                lead_pair_count=lead_pair_count,
                pair_floor=min(hand_pair_count, lead_pair_count),
                target_width=lead_count,
            ),
        )
    )


def _cards_are_in_hand(
    played_cards: list[Card],
    hand_cards: tuple[Card, ...],
) -> bool:
    hand_ids = {card.id for card in hand_cards}
    return all(card.id in hand_ids for card in played_cards)


def _prefix_is_ordered(prefix: tuple[FaceCount, ...]) -> bool:
    seen: set[CardFace] = set()
    previous: CardFace | None = None
    for item in prefix:
        if item.face in seen:
            return False
        seen.add(item.face)
        if previous is not None and face_sort_key(item.face) <= (
            face_sort_key(previous)
        ):
            return False
        previous = item.face
    return True


def _counts_available(
    prefix: tuple[FaceCount, ...],
    hand_cards: tuple[Card, ...],
) -> bool:
    available = _face_count_map(canonical_face_counts(hand_cards))
    for item in prefix:
        if item.count > available.get(item.face, 0):
            return False
    return True


def _face_effective_suit(
    face: CardFace,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> EffectiveSuit:
    if face.suit == Suit.JOKER:
        return "trump"
    if face.rank == trump_rank:
        return "trump"
    if trump_suit is not None and face.suit == trump_suit:
        return "trump"
    return face.suit


def _face_count_map(face_counts: tuple[FaceCount, ...]) -> FaceCountMap:
    return {item.face: item.count for item in face_counts}


def _last_face(prefix: tuple[FaceCount, ...]) -> CardFace | None:
    if not prefix:
        return None
    return prefix[-1].face


def _remaining_capacity_after_last(
    *,
    available_faces: tuple[FaceCount, ...],
    selected: FaceCountMap,
    last_face: CardFace | None,
    pair_counts_allowed: bool,
) -> int:
    capacity = 0
    for available in available_faces:
        if selected.get(available.face, 0) > 0:
            continue
        if last_face is not None and face_sort_key(
            available.face
        ) <= face_sort_key(last_face):
            continue
        if pair_counts_allowed:
            capacity += available.count
        else:
            capacity += min(available.count, 1)
    return capacity
