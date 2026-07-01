"""Structured observation token records for training policies.

The schema keeps one business item as one model token.  Card face
identity is represented by card components, while public context such as
segment, actor, event age, trick age, and play order is represented by
separate token components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from server.rules.cards import Card, Rank, Suit
from server.sm.constants import PLAYER_COUNT

type RelativeRole = Literal[
    "self",
    "partner",
    "left_enemy",
    "right_enemy",
]
type CardColor = Literal["red", "black", "none"]
type TokenScalar = str | int | bool | None
type ObservationSegment = Literal[
    "global_context",
    "round_context",
    "round_event",
    "stir_event",
    "self_hand",
    "visible_bottom",
    "own_exchange_pickup",
    "own_exchange_discard",
    "own_exchange_resulting_bottom",
    "play_record",
    "trick_result",
    "action_query",
]
type TrickRecordState = Literal["open", "completed"]
type GlobalFieldName = Literal[
    "team_layout",
    "left_player_role",
    "right_player_role",
    "partner_role",
    "deck_count",
    "player_count",
    "bottom_card_count",
    "required_level",
    "final_target",
    "rules_version",
]
type RoundFieldName = Literal[
    "phase",
    "awaiting_action",
    "dealer_role",
    "dealer_team",
    "self_team_level",
    "enemy_team_level",
    "self_team_required_level",
    "enemy_team_required_level",
    "self_team_distance_to_required_level",
    "enemy_team_distance_to_required_level",
    "trump_suit",
    "level_rank",
    "level_card_revealer_role",
    "current_score",
    "remaining_cards_self",
    "remaining_cards_partner",
    "remaining_cards_left_enemy",
    "remaining_cards_right_enemy",
    "winning_team",
]
type RoundEventFieldName = Literal[
    "event_kind",
    "actor",
    "bid_kind",
    "stir_kind",
    "suit",
    "joker_type",
    "count",
    "priority",
    "trigger",
    "stir_event_age",
]
type TrickResultFieldName = Literal["winner", "points"]
type ActionQueryFieldName = Literal[
    "kind",
    "pass_allowed",
    "min_select",
    "max_select",
    "exact_select",
    "selection_source",
    "action_play_order",
    "current_trick_width",
    "lead_actor",
    "discard_count",
    "trump_suit",
    "level_rank",
    "current_best_bid_role",
]


@dataclass(frozen=True, slots=True)
class CardToken:
    """A visible physical card plus explicit public structure."""

    card_id: str
    suit: Suit
    rank: Rank
    points: int
    color: CardColor
    segment: ObservationSegment
    role: RelativeRole | None = None
    trick_age: int | None = None
    trick_state: TrickRecordState | None = None
    play_order: int | None = None
    card_order: int | None = None
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
    CardToken
    | GlobalFieldToken
    | RoundFieldToken
    | RoundEventFieldToken
    | TrickResultFieldToken
    | ActionQueryFieldToken
)


def relative_role(viewer: int, actor: int) -> RelativeRole:
    """Map absolute player index to viewer-relative role."""
    if actor == viewer:
        return "self"
    if actor == (viewer + 2) % PLAYER_COUNT:
        return "partner"
    if actor == (viewer + 1) % PLAYER_COUNT:
        return "left_enemy"
    return "right_enemy"


def token_name(token: ObservationToken) -> str:
    """Return a stable short name for tests, metrics, and debugging."""
    if isinstance(token, CardToken):
        return "CARD"
    if isinstance(token, GlobalFieldToken):
        return "GLOBAL_FIELD"
    if isinstance(token, RoundFieldToken):
        return "ROUND_FIELD"
    if isinstance(token, RoundEventFieldToken):
        return "ROUND_EVENT_FIELD"
    if isinstance(token, TrickResultFieldToken):
        return "TRICK_RESULT_FIELD"
    return "ACTION_QUERY_FIELD"


def card_token(
    card: Card,
    *,
    segment: ObservationSegment,
    role: RelativeRole | None = None,
    trick_age: int | None = None,
    trick_state: TrickRecordState | None = None,
    play_order: int | None = None,
    card_order: int | None = None,
    play_width: int | None = None,
    event_age: int | None = None,
) -> CardToken:
    """Create one model token for a visible physical card."""
    return CardToken(
        card_id=card.id,
        suit=card.suit,
        rank=card.rank,
        points=card.points,
        color=_card_color(card),
        segment=segment,
        role=role,
        trick_age=trick_age,
        trick_state=trick_state,
        play_order=play_order,
        card_order=card_order,
        play_width=play_width,
        event_age=event_age,
    )


def _card_color(card: Card) -> CardColor:
    if card.suit in (Suit.HEARTS, Suit.DIAMONDS):
        return "red"
    if card.suit in (Suit.SPADES, Suit.CLUBS):
        return "black"
    return "none"
