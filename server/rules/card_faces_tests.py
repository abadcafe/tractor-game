"""Tests for rules.card_faces public interface."""

from __future__ import annotations

from server.player.test_helpers import card
from server.result import Ok, Rejected
from server.rules.card_faces import (
    CardFace,
    FaceCount,
    bind_face_counts,
    canonical_face_counts,
)


def test_canonical_face_counts_groups_duplicate_cards() -> None:
    first = card("hearts", "A", 1)
    second = card("hearts", "A", 2)
    spade = card("spades", "K", 1)

    result = canonical_face_counts((spade, second, first))

    assert result == (
        FaceCount(CardFace(first.suit, first.rank), 2),
        FaceCount(CardFace(spade.suit, spade.rank), 1),
    )


def test_bind_face_counts_uses_available_cards() -> None:
    first = card("hearts", "A", 1)
    second = card("hearts", "A", 2)

    result = bind_face_counts(
        (FaceCount(CardFace(first.suit, first.rank), 2),),
        (first, second),
    )

    assert isinstance(result, Ok)
    assert result.value == [first, second]


def test_bind_face_counts_rejects_missing_count() -> None:
    first = card("hearts", "A", 1)

    result = bind_face_counts(
        (FaceCount(CardFace(first.suit, first.rank), 2),),
        (first,),
    )

    assert isinstance(result, Rejected)
