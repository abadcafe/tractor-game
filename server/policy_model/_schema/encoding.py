"""Closed numeric ids for compact typed-token tensors."""

from __future__ import annotations

from enum import IntEnum


class TokenVariant(IntEnum):
    """Global payload variants; zero means inactive."""

    MANDATORY_LEVEL = 1
    DECLARER_ACTOR = 2
    OWN_LEVEL = 3
    OPPONENT_LEVEL = 4
    OWN_TARGET = 5
    OPPONENT_TARGET = 6
    OWN_DISTANCE = 7
    OPPONENT_DISTANCE = 8
    TRUMP_STATE = 9
    LEVEL_RANK = 10
    DEFENDER_POINTS = 11
    REMAINING_CARDS = 12
    TRICK = 13
    ACTION = 14
    CARD = 15


class SemanticState(IntEnum):
    """Disjoint non-optional semantic states; zero means inactive."""

    UNSET = 1
    NO_TRUMP = 2
    SUITED_TRUMP = 3
    WIN_TARGET = 4
    OPEN_TRICK = 5
    COMPLETED_TRICK = 6
    ACTION_FACT = 7
    ACTION_QUERY = 8


CATEGORY_COUNT: int = 11

FAMILY_INDEX: int = 0
VARIANT_INDEX: int = 1
ACTOR_INDEX: int = 2
RANK_INDEX: int = 3
SUIT_INDEX: int = 4
EFFECTIVE_SUIT_INDEX: int = 5
ACTION_KIND_INDEX: int = 6
STATE_INDEX: int = 7
DISPOSITION_INDEX: int = 8
TRICK_POSITION_INDEX: int = 9
PAYLOAD_ROLE_INDEX: int = 10

TOKEN_VARIANT_COUNT: int = max(TokenVariant) + 1
SEMANTIC_STATE_COUNT: int = max(SemanticState) + 1

ACTOR_COUNT: int = 5
RANK_COUNT: int = 16
SUIT_COUNT: int = 6
EFFECTIVE_SUIT_COUNT: int = 7
ACTION_KIND_COUNT: int = 5
DISPOSITION_COUNT: int = 3
TRICK_POSITION_COUNT: int = 5
PAYLOAD_ROLE_COUNT: int = 9

__all__ = (
    "ACTION_KIND_COUNT",
    "ACTOR_COUNT",
    "CATEGORY_COUNT",
    "DISPOSITION_COUNT",
    "EFFECTIVE_SUIT_COUNT",
    "PAYLOAD_ROLE_COUNT",
    "RANK_COUNT",
    "SEMANTIC_STATE_COUNT",
    "SUIT_COUNT",
    "TOKEN_VARIANT_COUNT",
    "TRICK_POSITION_COUNT",
    "SemanticState",
    "TokenVariant",
)
