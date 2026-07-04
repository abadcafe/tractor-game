"""Shared context dimensions for training observation tokens."""

from __future__ import annotations

from typing import Literal

from server.sm.constants import PLAYER_COUNT

type RelativeRole = Literal[
    "self",
    "partner",
    "left_enemy",
    "right_enemy",
]
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
    "play_record",
    "failed_throw_attempted",
    "failed_throw_forced",
    "trick_result",
    "action_query",
]
type TrickRecordState = Literal["open", "completed"]


def relative_role(viewer: int, actor: int) -> RelativeRole:
    """Map absolute player index to viewer-relative role."""
    if actor == viewer:
        return "self"
    if actor == (viewer + 2) % PLAYER_COUNT:
        return "partner"
    if actor == (viewer + 1) % PLAYER_COUNT:
        return "left_enemy"
    return "right_enemy"
