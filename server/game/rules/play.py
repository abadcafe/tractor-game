"""Complete play legality, comparison, and action-space rules."""

from __future__ import annotations

from dataclasses import dataclass

from server.foundation.result import Ok, Rejected

from ._compare import compare_plays
from ._follow import illegal_follow_rejection, is_legal_follow
from ._follow_space import (
    FollowActionSpace,
    FollowSelectionPlan,
    build_follow_action_space,
)
from ._hints import (
    MAX_PLAY_ACTION_HINTS,
    get_legal_play_hints,
    sort_play_action_hints,
)
from ._ordering import (
    effective_suit,
    sort_by_display_order,
    trump_order,
)
from ._throw import resolve_lead_throw
from .cards import Card, Rank, Suit
from .cards.faces import FaceCount, face_count_width

__all__ = [
    "FollowActionSpace",
    "FollowSelectionPlan",
    "LeadResolution",
    "build_follow_action_space",
    "choose_legal_play",
    "closed_hints",
    "compare",
    "effective_suit",
    "resolve_lead",
    "sort_hand",
    "trump_order",
    "validate_follow",
]


@dataclass(frozen=True, slots=True)
class LeadResolution:
    """The cards actually accepted after throw resolution."""

    cards: tuple[Card, ...]


def resolve_lead(
    hand: tuple[Card, ...],
    attempted: tuple[Card, ...],
    trump_suit: Suit | None,
    trump_rank: Rank,
    other_hands: tuple[tuple[Card, ...], ...],
) -> Ok[LeadResolution] | Rejected:
    """Validate a lead and resolve a failed throw if necessary."""
    match resolve_lead_throw(
        list(hand),
        list(attempted),
        trump_suit,
        trump_rank,
        [list(other) for other in other_hands],
    ):
        case Ok(value=resolution):
            return Ok(
                LeadResolution(cards=tuple(resolution.played_cards))
            )
        case Rejected() as rejected:
            return rejected


def validate_follow(
    hand: tuple[Card, ...],
    played: tuple[Card, ...],
    lead: tuple[Card, ...],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> Ok[None] | Rejected:
    """Validate one follow play against the lead."""
    if is_legal_follow(
        list(hand),
        list(played),
        list(lead),
        trump_suit,
        trump_rank,
    ):
        return Ok(None)
    return illegal_follow_rejection(
        list(hand),
        list(played),
        list(lead),
        trump_suit,
        trump_rank,
    )


def compare(
    candidate: tuple[Card, ...],
    incumbent: tuple[Card, ...],
    lead: tuple[Card, ...],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> int:
    """Compare two plays under the structure established by lead."""
    return compare_plays(
        list(candidate),
        list(incumbent),
        list(lead),
        trump_suit,
        trump_rank,
    )


def choose_legal_play(
    hand: tuple[Card, ...],
    lead: tuple[Card, ...] | None,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> tuple[Card, ...]:
    """Choose one deterministic legal play for automation."""
    if lead is None:
        assert hand
        return (
            sort_by_display_order(
                list(hand),
                trump_suit,
                trump_rank,
            )[0],
        )
    result = get_legal_play_hints(
        hand=list(hand),
        lead_cards=list(lead),
        trump_suit=trump_suit,
        trump_rank=trump_rank,
        max_hints=1,
    )
    if isinstance(result, Ok) and result.value:
        return tuple(result.value[0])
    space_result = build_follow_action_space(
        hand=list(hand),
        lead_cards=list(lead),
        trump_suit=trump_suit,
        trump_rank=trump_rank,
    )
    assert isinstance(space_result, Ok)
    action_space = space_result.value
    prefix: tuple[FaceCount, ...] = ()
    while face_count_width(prefix) < len(lead):
        choices = action_space.allowed_next(prefix)
        assert choices
        prefix = prefix + (choices[0],)
    decoded = action_space.decode(prefix)
    assert isinstance(decoded, Ok)
    return tuple(decoded.value)


def closed_hints(
    hand: tuple[Card, ...],
    lead: tuple[Card, ...] | None,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> tuple[tuple[Card, ...], ...]:
    """Return a complete bounded UI hint set, or no closed set."""
    result = get_legal_play_hints(
        hand=list(hand),
        lead_cards=None if lead is None else list(lead),
        trump_suit=trump_suit,
        trump_rank=trump_rank,
        max_hints=MAX_PLAY_ACTION_HINTS,
    )
    if isinstance(result, Rejected):
        return ()
    return tuple(
        tuple(cards)
        for cards in sort_play_action_hints(
            result.value,
            trump_suit,
            trump_rank,
        )
    )


def sort_hand(
    cards: tuple[Card, ...],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> tuple[Card, ...]:
    """Sort a hand in the canonical player-facing order."""
    return tuple(
        sort_by_display_order(
            list(cards),
            trump_suit,
            trump_rank,
        )
    )
