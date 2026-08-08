"""Analyze physical cards against the fixed tractor topology."""

from __future__ import annotations

from server.game.rules._ordering import EffectiveSuit, effective_suit
from server.game.rules.cards import Card, Rank, Suit
from server.game.rules.cards.faces import (
    CardFace,
    card_face,
    face_sort_key,
)

from .model import PairRun, PairTier, Pattern, PatternAnalysis
from .topology import pair_tiers


def analyze_patterns(
    cards: list[Card] | tuple[Card, ...],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> PatternAnalysis:
    """Analyze a non-empty card set deterministically."""
    assert cards
    assert len({card.id for card in cards}) == len(cards)
    effective_suits: set[EffectiveSuit] = {
        effective_suit(card, trump_suit, trump_rank) for card in cards
    }
    assert len(effective_suits) == 1
    analyzed_suit: EffectiveSuit = next(iter(effective_suits))

    cards_by_face: dict[CardFace, list[Card]] = {}
    for card in cards:
        cards_by_face.setdefault(card_face(card), []).append(card)
    assert all(1 <= len(group) <= 2 for group in cards_by_face.values())

    pair_faces = tuple(
        sorted(
            (
                face
                for face, group in cards_by_face.items()
                if len(group) == 2
            ),
            key=face_sort_key,
        )
    )
    topology = pair_tiers(analyzed_suit, trump_suit, trump_rank)
    tier_by_face = {
        face: index
        for index, tier in enumerate(topology)
        for face in tier
    }
    assert all(face in tier_by_face for face in pair_faces)
    occupied_tiers = tuple(
        tuple(face for face in tier if face in pair_faces)
        for tier in topology
    )
    tractor_runs = _tractor_runs(occupied_tiers)

    used_faces: set[CardFace] = set()
    patterns: list[Pattern] = []
    for run in tractor_runs:
        selected_faces = tuple(tier[0] for tier in run.tiers)
        used_faces.update(selected_faces)
        patterns.append(
            _paired_pattern(
                selected_faces,
                cards_by_face,
                analyzed_suit,
            )
        )
    for face in sorted(
        (face for face in pair_faces if face not in used_faces),
        key=lambda item: tier_by_face[item],
    ):
        patterns.append(
            _paired_pattern((face,), cards_by_face, analyzed_suit)
        )
    for face, group in sorted(
        cards_by_face.items(), key=lambda item: face_sort_key(item[0])
    ):
        if len(group) == 1:
            patterns.append(
                Pattern(
                    cards=(group[0],),
                    pair_faces=(),
                    effective_suit=analyzed_suit,
                )
            )
    patterns.sort(key=lambda pattern: pattern.level, reverse=True)
    return PatternAnalysis(
        effective_suit=analyzed_suit,
        patterns=tuple(patterns),
        pair_faces=pair_faces,
        tractor_runs=tractor_runs,
    )


def tractor_windows(
    pattern: Pattern,
    pair_count: int,
) -> tuple[tuple[Card, ...], ...]:
    """Return every contiguous pair window of one tractor pattern."""
    assert pattern.pair_count >= 2
    assert 0 <= pair_count <= pattern.pair_count
    if pair_count == 0:
        return ((),)
    by_face: dict[CardFace, tuple[Card, ...]] = {}
    for face in pattern.pair_faces:
        pair = tuple(
            card for card in pattern.cards if card_face(card) == face
        )
        assert len(pair) == 2
        by_face[face] = pair
    return tuple(
        tuple(
            card
            for face in pattern.pair_faces[start : start + pair_count]
            for card in by_face[face]
        )
        for start in range(pattern.pair_count - pair_count + 1)
    )


def _tractor_runs(
    occupied_tiers: tuple[PairTier, ...],
) -> tuple[PairRun, ...]:
    result: list[PairRun] = []
    current: list[PairTier] = []
    for tier in occupied_tiers:
        if tier:
            current.append(tier)
            continue
        if len(current) >= 2:
            result.append(PairRun(tiers=tuple(current)))
        current = []
    if len(current) >= 2:
        result.append(PairRun(tiers=tuple(current)))
    return tuple(result)


def _paired_pattern(
    faces: tuple[CardFace, ...],
    cards_by_face: dict[CardFace, list[Card]],
    analyzed_suit: EffectiveSuit,
) -> Pattern:
    cards = tuple(
        card for face in faces for card in cards_by_face[face]
    )
    return Pattern(
        cards=cards,
        pair_faces=faces,
        effective_suit=analyzed_suit,
    )
