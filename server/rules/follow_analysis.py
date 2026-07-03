"""Shared follow-play analysis for validation and action masking."""

from __future__ import annotations

from dataclasses import dataclass

from server.result import Ok, Rejected
from server.rules.card_faces import (
    CardFace,
    FaceCount,
    bind_face_counts,
    canonical_face_counts,
    card_face,
    face_count_width,
    face_sort_key,
)
from server.rules.cards import Card, Rank, Suit
from server.rules.decompose import decompose
from server.rules.ordering import effective_suit
from server.rules.rejections.play import EmptyLeadRejected
from server.rules.types import EffectiveSuit, SubPlay

type FaceCountMap = dict[CardFace, int]


@dataclass(frozen=True, slots=True)
class PairUnit:
    """One available same-suit pair tracked for prefix completion."""

    face: CardFace
    level: int
    tractor_index: int | None
    tractor_position: int | None


@dataclass(frozen=True, slots=True)
class TractorUnitRange:
    """Pair-unit index range owned by one tractor sub-play."""

    tractor_index: int
    start: int
    end: int


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
    pair_units: tuple[PairUnit, ...]
    tractor_ranges: tuple[TractorUnitRange, ...]
    hand_pairs_by_level: tuple[tuple[int, int], ...]

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

        played_pair_faces = _played_pair_faces(played_suit_cards)
        if not self._pair_priority_is_satisfied(played_pair_faces):
            return False
        return self._tractor_continuity_is_satisfied(played_pair_faces)

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

        prefix_counts = _face_count_map(prefix)
        last_face = _last_face(prefix)
        for mask in range(1 << len(self.pair_units)):
            if not self._pair_mask_is_compatible(
                mask=mask,
                prefix_counts=prefix_counts,
                last_face=last_face,
            ):
                continue
            if not self._pair_mask_priority_is_satisfied(mask):
                continue
            if self._pair_mask_can_fill_singles(
                mask=mask,
                prefix_counts=prefix_counts,
                last_face=last_face,
                target_width=target_width,
            ):
                return True
        return False

    def _pair_mask_is_compatible(
        self,
        *,
        mask: int,
        prefix_counts: FaceCountMap,
        last_face: CardFace | None,
    ) -> bool:
        pair_index_by_face = _pair_index_by_face(self.pair_units)
        for face, selected_count in prefix_counts.items():
            pair_index = pair_index_by_face.get(face)
            bit_set = pair_index is not None and (
                (mask >> pair_index) & 1
            )
            if selected_count == 2 and not bit_set:
                return False
            if selected_count == 1 and bit_set:
                return False
        if last_face is None:
            return True
        last_key = face_sort_key(last_face)
        for index, unit in enumerate(self.pair_units):
            if ((mask >> index) & 1) == 0:
                continue
            if face_sort_key(unit.face) > last_key:
                continue
            if prefix_counts.get(unit.face, 0) != 2:
                return False
        return True

    def _pair_mask_priority_is_satisfied(self, mask: int) -> bool:
        selected_indices = _selected_pair_indices(mask, self.pair_units)
        if len(selected_indices) < self.pair_floor:
            return False

        played_by_level: dict[int, int] = {}
        for index in selected_indices:
            level = self.pair_units[index].level
            played_by_level[level] = played_by_level.get(level, 0) + 1

        remaining_needed = self.lead_pair_count
        levels = sorted(
            {level for level, _count in self.hand_pairs_by_level}
            | set(played_by_level.keys()),
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
        return self._pair_mask_tractor_continuity_is_satisfied(mask)

    def _pair_mask_tractor_continuity_is_satisfied(
        self, mask: int
    ) -> bool:
        for tractor_range in self.tractor_ranges:
            positions = [
                index - tractor_range.start
                for index in range(
                    tractor_range.start, tractor_range.end
                )
                if ((mask >> index) & 1) == 1
            ]
            if not positions:
                continue
            if (
                len(positions)
                == tractor_range.end - tractor_range.start
            ):
                continue
            if positions[-1] - positions[0] != len(positions) - 1:
                return False
        return True

    def _pair_mask_can_fill_singles(
        self,
        *,
        mask: int,
        prefix_counts: FaceCountMap,
        last_face: CardFace | None,
        target_width: int,
    ) -> bool:
        final_counts = dict(prefix_counts)
        for index, unit in enumerate(self.pair_units):
            if ((mask >> index) & 1) == 0:
                continue
            current = final_counts.get(unit.face, 0)
            if current == 1:
                return False
            final_counts[unit.face] = 2

        fixed_width = sum(final_counts.values())
        if fixed_width > target_width:
            return False
        remaining = target_width - fixed_width
        return (
            _remaining_capacity_after_last(
                available_faces=self.same_suit_faces,
                selected=final_counts,
                last_face=last_face,
                pair_counts_allowed=False,
            )
            >= remaining
        )

    def _pair_priority_is_satisfied(
        self, played_pair_faces: set[CardFace]
    ) -> bool:
        pair_index_by_face = _pair_index_by_face(self.pair_units)
        mask = 0
        for face in played_pair_faces:
            pair_index = pair_index_by_face.get(face)
            if pair_index is None:
                continue
            mask |= 1 << pair_index
        return self._pair_mask_priority_is_satisfied(mask)

    def _tractor_continuity_is_satisfied(
        self, played_pair_faces: set[CardFace]
    ) -> bool:
        pair_index_by_face = _pair_index_by_face(self.pair_units)
        mask = 0
        for face in played_pair_faces:
            pair_index = pair_index_by_face.get(face)
            if pair_index is None:
                continue
            mask |= 1 << pair_index
        return self._pair_mask_tractor_continuity_is_satisfied(mask)


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
    pair_units, tractor_ranges = _pair_units(hand_subs)
    hand_pairs_by_level = _hand_pairs_by_level(hand_subs)
    lead_count = len(lead_cards)
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
            same_suit_faces=canonical_face_counts(same_suit_cards),
            off_suit_faces=canonical_face_counts(off_suit_cards),
            required_same_suit_count=min(
                len(same_suit_cards), lead_count
            ),
            lead_pair_count=lead_pair_count,
            pair_floor=min(hand_pair_count, lead_pair_count),
            pair_units=pair_units,
            tractor_ranges=tractor_ranges,
            hand_pairs_by_level=hand_pairs_by_level,
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


def _pair_units(
    hand_subs: list[SubPlay],
) -> tuple[tuple[PairUnit, ...], tuple[TractorUnitRange, ...]]:
    units: list[PairUnit] = []
    tractor_ranges: list[TractorUnitRange] = []
    tractor_index = 0
    for sub in hand_subs:
        if sub.pair_count == 0:
            continue
        faces = _sub_pair_faces(sub)
        if sub.pair_count >= 2:
            start = len(units)
            for position, face in enumerate(faces):
                units.append(
                    PairUnit(
                        face=face,
                        level=sub.pair_count,
                        tractor_index=tractor_index,
                        tractor_position=position,
                    )
                )
            tractor_ranges.append(
                TractorUnitRange(
                    tractor_index=tractor_index,
                    start=start,
                    end=len(units),
                )
            )
            tractor_index += 1
            continue
        for face in faces:
            units.append(
                PairUnit(
                    face=face,
                    level=1,
                    tractor_index=None,
                    tractor_position=None,
                )
            )
    return tuple(units), tuple(tractor_ranges)


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
    hand_subs: list[SubPlay],
) -> tuple[tuple[int, int], ...]:
    counts: dict[int, int] = {}
    for sub in hand_subs:
        counts[sub.pair_count] = (
            counts.get(sub.pair_count, 0) + sub.pair_count
        )
    return tuple(sorted(counts.items(), reverse=True))


def _played_pair_faces(
    played_suit_cards: tuple[Card, ...],
) -> set[CardFace]:
    counts: dict[CardFace, int] = {}
    result: set[CardFace] = set()
    for card in played_suit_cards:
        face = card_face(card)
        count = counts.get(face, 0) + 1
        counts[face] = count
        if count == 2:
            result.add(face)
    return result


def _pair_index_by_face(
    pair_units: tuple[PairUnit, ...],
) -> dict[CardFace, int]:
    return {unit.face: index for index, unit in enumerate(pair_units)}


def _selected_pair_indices(
    mask: int,
    pair_units: tuple[PairUnit, ...],
) -> tuple[int, ...]:
    return tuple(
        index
        for index in range(len(pair_units))
        if ((mask >> index) & 1) == 1
    )
