"""Viewer-relative player, trick-position, and trump semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from server.game.rules.cards import Suit
from server.game.state_machine.constants import PLAYER_COUNT


class RelativeActor(str, Enum):
    """One player expressed relative to the policy viewer."""

    SELF = "self"
    PARTNER = "partner"
    LEFT_ENEMY = "left_enemy"
    RIGHT_ENEMY = "right_enemy"


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


def relative_actor(viewer: int, actor: int) -> RelativeActor:
    """Convert one absolute actor at the sole absolute-position edge."""
    assert 0 <= viewer < PLAYER_COUNT
    assert 0 <= actor < PLAYER_COUNT
    offset = (actor - viewer) % PLAYER_COUNT
    if offset == 0:
        return RelativeActor.SELF
    if offset == 1:
        return RelativeActor.LEFT_ENEMY
    if offset == 2:
        return RelativeActor.PARTNER
    return RelativeActor.RIGHT_ENEMY


def trick_position(*, lead_player: int, actor: int) -> TrickPosition:
    """Return chronological trick position from the game topology."""
    assert 0 <= lead_player < PLAYER_COUNT
    assert 0 <= actor < PLAYER_COUNT
    offset = (actor - lead_player) % PLAYER_COUNT
    return (
        TrickPosition.LEAD,
        TrickPosition.FOLLOW_1,
        TrickPosition.FOLLOW_2,
        TrickPosition.FOLLOW_3,
    )[offset]


__all__ = (
    "RelativeActor",
    "TrickPosition",
    "TrumpMode",
    "TrumpState",
)
