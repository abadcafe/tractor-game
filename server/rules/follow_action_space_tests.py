"""Tests for rules.follow_action_space public interface."""

from __future__ import annotations

from itertools import combinations

from server.player.test_helpers import card
from server.result import Ok, Rejected
from server.rules.card_faces import (
    CardFace,
    FaceCount,
    canonical_face_counts,
)
from server.rules.cards import Card, Rank, Suit
from server.rules.follow import is_legal_follow
from server.rules.follow_action_space import (
    FollowActionSpace,
    build_follow_action_space,
)


def test_allowed_next_single_must_follow_suit() -> None:
    lead = [card("hearts", "A")]
    heart = card("hearts", "3")
    spade = card("spades", "K")
    space = _space([heart, spade], lead)

    assert space.allowed_next(()) == (_face_count(heart, 1),)


def test_allowed_next_single_no_suit_allows_any_face() -> None:
    lead = [card("hearts", "A")]
    spade = card("spades", "K")
    club = card("clubs", "Q")
    space = _space([spade, club], lead)

    assert space.allowed_next(()) == (
        _face_count(spade, 1),
        _face_count(club, 1),
    )


def test_allowed_next_pair_requires_pair_when_available() -> None:
    lead = [card("hearts", "A", 1), card("hearts", "A", 2)]
    heart_first = card("hearts", "3", 1)
    heart_second = card("hearts", "3", 2)
    heart_single = card("hearts", "K", 1)
    space = _space([heart_first, heart_second, heart_single], lead)

    assert space.allowed_next(()) == (_face_count(heart_first, 2),)


def test_allowed_next_pair_allows_singles_when_no_pair() -> None:
    lead = [card("hearts", "A", 1), card("hearts", "A", 2)]
    heart_first = card("hearts", "3", 1)
    heart_second = card("hearts", "K", 1)
    space = _space([heart_first, heart_second], lead)

    assert space.allowed_next(()) == (_face_count(heart_first, 1),)
    assert space.allowed_next((_face_count(heart_first, 1),)) == (
        _face_count(heart_second, 1),
    )


def test_allowed_next_tractor_partial_requires_contiguous_pairs() -> (
    None
):
    lead = [
        card("hearts", "A", 1),
        card("hearts", "A", 2),
        card("hearts", "K", 1),
        card("hearts", "K", 2),
    ]
    h3 = [card("hearts", "3", 1), card("hearts", "3", 2)]
    h4 = [card("hearts", "4", 1), card("hearts", "4", 2)]
    h5 = [card("hearts", "5", 1), card("hearts", "5", 2)]
    space = _space([*h3, *h4, *h5], lead)

    prefix = (_face_count(h3[0], 2),)

    assert _face_count(h4[0], 2) in space.allowed_next(prefix)
    assert _face_count(h5[0], 2) not in space.allowed_next(prefix)


def test_decode_accepts_full_legal_trace() -> None:
    lead = [card("hearts", "A")]
    heart = card("hearts", "3")
    space = _space([heart], lead)

    result = space.decode((_face_count(heart, 1),))

    assert isinstance(result, Ok)
    assert result.value == [heart]


def test_decode_rejects_illegal_trace() -> None:
    lead = [card("hearts", "A")]
    heart = card("hearts", "3")
    spade = card("spades", "K")
    space = _space([heart, spade], lead)

    result = space.decode((_face_count(spade, 1),))

    assert isinstance(result, Rejected)


def test_decode_requires_complete_trace() -> None:
    lead = [card("hearts", "A", 1), card("hearts", "A", 2)]
    heart = card("hearts", "3")
    space = _space([heart], lead)

    result = space.decode((_face_count(heart, 1),))

    assert isinstance(result, Rejected)


def test_allowed_next_fewer_suit_cards_requires_exhausting_suit() -> (
    None
):
    lead = [
        card("hearts", "A", 1),
        card("hearts", "A", 2),
        card("hearts", "K", 1),
    ]
    heart = card("hearts", "3")
    spade = card("spades", "K")
    club = card("clubs", "Q")
    space = _space([heart, spade, club], lead)

    assert space.allowed_next(()) == (_face_count(heart, 1),)
    assert space.allowed_next((_face_count(heart, 1),)) == (
        _face_count(spade, 1),
    )
    assert space.allowed_next(
        (_face_count(heart, 1), _face_count(spade, 1))
    ) == (_face_count(club, 1),)


def test_decode_matches_is_legal_follow_small_set() -> None:
    lead = [card("hearts", "A", 1), card("hearts", "A", 2)]
    hand = [
        card("hearts", "3", 1),
        card("hearts", "3", 2),
        card("hearts", "K", 1),
        card("spades", "Q", 1),
    ]
    space = _space(hand, lead)

    for selected_cards in combinations(hand, len(lead)):
        face_counts = canonical_face_counts(tuple(selected_cards))
        decoded = space.decode(face_counts)
        expected = is_legal_follow(
            hand,
            list(selected_cards),
            lead,
            Suit.SPADES,
            Rank.TWO,
        )
        assert isinstance(decoded, Ok) is expected


def test_allowed_next_wide_no_suit_is_bounded() -> None:
    lead = [
        card("hearts", "A", 1),
        card("hearts", "A", 2),
        card("hearts", "K", 1),
        card("hearts", "K", 2),
    ]
    hand = [
        card("spades", "3", 1),
        card("spades", "3", 2),
        card("spades", "4", 1),
        card("spades", "4", 2),
        card("clubs", "5", 1),
        card("clubs", "5", 2),
        card("diamonds", "6", 1),
        card("diamonds", "6", 2),
    ]
    space = _space(hand, lead)

    allowed = space.allowed_next(())

    assert allowed == (
        _face_count(hand[0], 1),
        _face_count(hand[0], 2),
        _face_count(hand[2], 1),
        _face_count(hand[2], 2),
        _face_count(hand[6], 2),
    )


def _space(hand: list[Card], lead: list[Card]) -> FollowActionSpace:
    result = build_follow_action_space(
        hand=hand,
        lead_cards=lead,
        trump_suit=Suit.SPADES,
        trump_rank=Rank.TWO,
    )
    assert isinstance(result, Ok)
    return result.value


def _face_count(card_value: Card, count: int) -> FaceCount:
    return FaceCount(
        CardFace(card_value.suit, card_value.rank),
        count,
    )
