"""Structured observation records for training policies.

The schema keeps model input semantic: card identity is represented by
face plus count, while public structure such as segment, actor, event
age, trick age, and play order is represented by separate components.
"""

from __future__ import annotations

from dataclasses import dataclass

from server.game.rules.card_faces import CardColor, FaceCount
from server.game.rules.cards import Rank, Suit
from server.training.token_context import (
    ObservationSegment,
    RelativeRole,
    TokenScalar,
    TrickRecordState,
)
from server.training.token_fields import (
    ActionQueryFieldName,
    GlobalFieldName,
    RoundEventFieldName,
    RoundFieldName,
    TrickResultFieldName,
)


@dataclass(frozen=True, slots=True)
class FaceCountToken:
    """A visible semantic card-face multiplicity plus context."""

    suit: Suit
    rank: Rank
    points: int
    color: CardColor
    count: int
    segment: ObservationSegment
    role: RelativeRole | None = None
    trick_age: int | None = None
    trick_state: TrickRecordState | None = None
    play_order: int | None = None
    play_width: int | None = None
    event_age: int | None = None


@dataclass(frozen=True, slots=True)
class GlobalFieldToken:
    """One global rules/context field."""

    field: GlobalFieldName
    value: TokenScalar


@dataclass(frozen=True, slots=True)
class RoundFieldToken:
    """One current-round public field."""

    field: RoundFieldName
    value: TokenScalar


@dataclass(frozen=True, slots=True)
class RoundEventFieldToken:
    """One field inside an ordered public round event."""

    field: RoundEventFieldName
    value: TokenScalar
    event_age: int


@dataclass(frozen=True, slots=True)
class TrickResultFieldToken:
    """One result field for a completed trick."""

    field: TrickResultFieldName
    value: TokenScalar
    trick_age: int


@dataclass(frozen=True, slots=True)
class ActionQueryFieldToken:
    """One field in the current structured decision request."""

    field: ActionQueryFieldName
    value: TokenScalar


type ObservationToken = (
    FaceCountToken
    | GlobalFieldToken
    | RoundFieldToken
    | RoundEventFieldToken
    | TrickResultFieldToken
    | ActionQueryFieldToken
)


def token_name(token: ObservationToken) -> str:
    """Return a stable short name for tests, metrics, and debugging."""
    if isinstance(token, FaceCountToken):
        return "FACE_COUNT"
    if isinstance(token, GlobalFieldToken):
        return "GLOBAL_FIELD"
    if isinstance(token, RoundFieldToken):
        return "ROUND_FIELD"
    if isinstance(token, RoundEventFieldToken):
        return "ROUND_EVENT_FIELD"
    if isinstance(token, TrickResultFieldToken):
        return "TRICK_RESULT_FIELD"
    return "ACTION_QUERY_FIELD"


def face_count_token(
    face_count: FaceCount,
    *,
    segment: ObservationSegment,
    role: RelativeRole | None = None,
    trick_age: int | None = None,
    trick_state: TrickRecordState | None = None,
    play_order: int | None = None,
    play_width: int | None = None,
    event_age: int | None = None,
) -> FaceCountToken:
    """Create one model token for a visible semantic card group."""
    face = face_count.face
    return FaceCountToken(
        suit=face.suit,
        rank=face.rank,
        points=face.points,
        color=face.color,
        count=face_count.count,
        segment=segment,
        role=role,
        trick_age=trick_age,
        trick_state=trick_state,
        play_order=play_order,
        play_width=play_width,
        event_age=event_age,
    )
