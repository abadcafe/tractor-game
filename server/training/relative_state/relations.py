"""Viewer-relative player, trick-position, and trump semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from server.game import Seat, next_seat, partner_seat, previous_seat
from server.game.rules.cards import Suit


class RelativeActor(str, Enum):
    """One player expressed relative to the policy viewer."""

    SELF = "self"
    NEXT_OPPONENT = "next_opponent"
    PARTNER = "partner"
    PREVIOUS_OPPONENT = "previous_opponent"


class TrickPosition(str, Enum):
    """Chronological position of an action inside a trick."""

    LEAD = "lead"
    FOLLOW_1 = "follow_1"
    FOLLOW_2 = "follow_2"
    FOLLOW_3 = "follow_3"


class TrumpMode(str, Enum):
    """Lifecycle state of the round trump suit."""

    UNSET = "unset"
    NO_TRUMP = "no_trump"
    SUITED = "suited"


@dataclass(frozen=True, slots=True)
class TrumpState:
    """Unambiguous round trump state."""

    mode: TrumpMode
    suit: Suit | None

    def __post_init__(self) -> None:
        assert (self.mode == TrumpMode.SUITED) == (
            self.suit is not None
        )
        if self.suit is not None:
            assert self.suit != Suit.JOKER


def relative_actor(viewer: Seat, actor: Seat) -> RelativeActor:
    """Convert one absolute actor at the sole absolute-position edge."""
    if actor == viewer:
        return RelativeActor.SELF
    if actor == next_seat(viewer):
        return RelativeActor.NEXT_OPPONENT
    if actor == partner_seat(viewer):
        return RelativeActor.PARTNER
    assert actor == previous_seat(viewer)
    return RelativeActor.PREVIOUS_OPPONENT


def trick_position(*, lead_actor: Seat, actor: Seat) -> TrickPosition:
    """Return chronological trick position from the game topology."""
    if actor == lead_actor:
        return TrickPosition.LEAD
    if actor == next_seat(lead_actor):
        return TrickPosition.FOLLOW_1
    if actor == partner_seat(lead_actor):
        return TrickPosition.FOLLOW_2
    assert actor == previous_seat(lead_actor)
    return TrickPosition.FOLLOW_3


__all__ = (
    "RelativeActor",
    "TrickPosition",
    "TrumpMode",
    "TrumpState",
)
