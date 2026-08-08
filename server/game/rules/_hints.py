"""Private bounded legal play hint enumeration."""

from __future__ import annotations

from collections.abc import Iterator

from server.foundation.result import Ok, Rejected

from ._follow_space import FollowActionSpace, build_follow_action_space
from ._ordering import trump_order
from ._patterns import analyze_patterns
from ._rejections.hints import TooManyPlayHintsRejected
from .cards import Card, Rank, Suit
from .cards.faces import FaceCount, face_count_width

MAX_PLAY_ACTION_HINTS: int = 25
TOO_MANY_PLAY_HINTS: str = TooManyPlayHintsRejected.reason_text

type PlayActionHintSortKey = tuple[
    int, int, tuple[int, ...], tuple[str, ...]
]


def get_legal_play_hints(
    hand: list[Card],
    lead_cards: list[Card] | None,
    trump_suit: Suit | None,
    trump_rank: Rank,
    max_hints: int,
) -> Ok[list[list[Card]]] | Rejected:
    """Return bounded, complete semantic follow-play hints."""
    assert max_hints >= 0
    if lead_cards is None or not lead_cards:
        return Ok([])
    space_result = build_follow_action_space(
        hand=hand,
        lead_cards=lead_cards,
        trump_suit=trump_suit,
        trump_rank=trump_rank,
    )
    if isinstance(space_result, Rejected):
        return space_result
    return _collect_bounded_hints(
        space_result.value,
        len(lead_cards),
        max_hints,
    )


def sort_play_action_hints(
    hints: list[list[Card]],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> list[list[Card]]:
    """Sort player-facing play hints from weakest to strongest."""
    return sorted(
        [list(cards) for cards in hints],
        key=lambda cards: _play_action_hint_sort_key(
            cards, trump_suit, trump_rank
        ),
    )


def _play_action_hint_sort_key(
    cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> PlayActionHintSortKey:
    max_level = (
        0
        if not cards
        else max(
            pattern.level
            for pattern in analyze_patterns(
                cards, trump_suit, trump_rank
            ).patterns
        )
    )
    card_orders = tuple(
        sorted(
            trump_order(card, trump_suit, trump_rank) for card in cards
        )
    )
    card_ids = tuple(sorted(card.id for card in cards))
    return (len(cards), max_level, card_orders, card_ids)


def _collect_bounded_hints(
    space: FollowActionSpace,
    target_width: int,
    max_hints: int,
) -> Ok[list[list[Card]]] | Rejected:
    result: list[list[Card]] = []
    for hint in _iter_hints(space, target_width, ()):
        result.append(hint)
        if len(result) > max_hints:
            return TooManyPlayHintsRejected()
    return Ok(result)


def _iter_hints(
    space: FollowActionSpace,
    target_width: int,
    prefix: tuple[FaceCount, ...],
) -> Iterator[list[Card]]:
    if face_count_width(prefix) == target_width:
        decoded = space.decode(prefix)
        assert isinstance(decoded, Ok)
        yield decoded.value
        return
    for choice in space.allowed_next(prefix):
        yield from _iter_hints(space, target_width, (*prefix, choice))
