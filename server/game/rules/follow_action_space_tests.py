"""Tests for rules.follow_action_space public interface."""

from __future__ import annotations

from itertools import combinations

from server.foundation.result import Ok, Rejected
from server.game.players.test_helpers import TestRank, TestSuit, card
from server.game.rules.card_faces import (
    CardFace,
    FaceCount,
    canonical_face_counts,
    face_count_width,
)
from server.game.rules.cards import Card, Rank, Suit
from server.game.rules.follow import is_legal_follow
from server.game.rules.follow_action_space import (
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


def test_allowed_next_three_pair_tractor_uses_pair_run_windows() -> (
    None
):
    lead = [
        card("hearts", "A", 1),
        card("hearts", "A", 2),
        card("hearts", "K", 1),
        card("hearts", "K", 2),
        card("hearts", "Q", 1),
        card("hearts", "Q", 2),
    ]
    pairs = [
        [card("hearts", rank, 1), card("hearts", rank, 2)]
        for rank in ("3", "4", "5", "6", "7")
    ]
    hand = [card_value for pair in pairs for card_value in pair]
    space = _space(hand, lead)

    assert space.allowed_next(()) == (
        _face_count(pairs[0][0], 2),
        _face_count(pairs[1][0], 2),
        _face_count(pairs[2][0], 2),
    )
    assert space.allowed_next((_face_count(pairs[1][0], 2),)) == (
        _face_count(pairs[2][0], 2),
    )


def test_allowed_traces_follow_tractor_priority_from_decompose() -> (
    None
):
    lead = [
        card("hearts", "A", 1),
        card("hearts", "A", 2),
        card("hearts", "Q", 1),
        card("hearts", "Q", 2),
    ]
    h3 = [card("hearts", "3", 1), card("hearts", "3", 2)]
    h4 = [card("hearts", "4", 1), card("hearts", "4", 2)]
    h5 = [card("hearts", "5", 1), card("hearts", "5", 2)]
    hk = [card("hearts", "K", 1), card("hearts", "K", 2)]
    hand = [*h3, *h4, *h5, *hk]
    space = _space(hand, lead)

    assert _semantic_traces(space) == {
        (_face_count(h3[0], 2), _face_count(h4[0], 2)),
        (_face_count(h4[0], 2), _face_count(h5[0], 2)),
    }


def test_decode_trump_rank_pair_is_independent_of_hand_order() -> None:
    clubs = [
        card("clubs", "3", 1),
        card("clubs", "3", 2),
    ]
    diamonds = [
        card("diamonds", "3", 1),
        card("diamonds", "3", 2),
    ]
    spades = [
        card("spades", "3", 1),
        card("spades", "3", 2),
    ]
    lead = [
        card("hearts", "3", 1),
        card("hearts", "3", 2),
    ]
    selected = canonical_face_counts(tuple(clubs))

    for hand in (
        [*clubs, *diamonds, *spades],
        [*spades, *clubs, *diamonds],
    ):
        space_result = build_follow_action_space(
            hand=hand,
            lead_cards=lead,
            trump_suit=Suit.HEARTS,
            trump_rank=Rank.THREE,
        )

        assert isinstance(space_result, Ok)
        assert isinstance(space_result.value.decode(selected), Ok)
        assert is_legal_follow(
            hand,
            clubs,
            lead,
            Suit.HEARTS,
            Rank.THREE,
        )


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


def test_allowed_traces_match_is_legal_follow_small_set() -> None:
    lead = [
        card("hearts", "A", 1),
        card("hearts", "A", 2),
        card("hearts", "K", 1),
        card("hearts", "K", 2),
    ]
    hand = [
        card("hearts", "3", 1),
        card("hearts", "3", 2),
        card("hearts", "4", 1),
        card("hearts", "4", 2),
        card("hearts", "5", 1),
        card("hearts", "5", 2),
        card("hearts", "Q", 1),
        card("hearts", "Q", 2),
    ]
    space = _space(hand, lead)
    expected: set[tuple[FaceCount, ...]] = set()

    for selected_cards in combinations(hand, len(lead)):
        if is_legal_follow(
            hand,
            list(selected_cards),
            lead,
            Suit.SPADES,
            Rank.TWO,
        ):
            expected.add(canonical_face_counts(tuple(selected_cards)))

    assert _semantic_traces(space) == expected


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


def test_wide_trump_tractor_mask_uses_compact_pair_plans() -> None:
    trump_pairs: list[tuple[TestSuit, TestRank]] = [
        ("spades", "3"),
        ("spades", "4"),
        ("spades", "5"),
        ("spades", "6"),
        ("spades", "7"),
        ("spades", "8"),
        ("spades", "9"),
        ("spades", "10"),
        ("spades", "J"),
        ("spades", "Q"),
        ("spades", "K"),
        ("spades", "A"),
        ("hearts", "2"),
        ("diamonds", "2"),
        ("clubs", "2"),
        ("spades", "2"),
    ]
    hand = [
        card(suit, rank, deck)
        for suit, rank in trump_pairs
        for deck in (1, 2)
    ]
    lead = [
        card("spades", "A", 1),
        card("spades", "A", 2),
        card("spades", "K", 1),
        card("spades", "K", 2),
    ]
    space = _space(hand, lead)

    assert len(space.analysis.pair_planner.pair_plans) <= 16
    allowed = space.allowed_next(())
    assert len(allowed) == 12
    assert space.allowed_next(()) is allowed


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


def _semantic_traces(
    space: FollowActionSpace,
    prefix: tuple[FaceCount, ...] = (),
) -> set[tuple[FaceCount, ...]]:
    if face_count_width(prefix) == space.analysis.lead_count:
        return {prefix}
    result: set[tuple[FaceCount, ...]] = set()
    for argument in space.allowed_next(prefix):
        result.update(_semantic_traces(space, (*prefix, argument)))
    return result
