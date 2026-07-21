"""Build player-visible semantic action queries from snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from server.game.protocol import StateSnapshot
from server.game.rules.card_faces import (
    FaceCount,
    canonical_face_counts,
    face_count_width,
)
from server.game.rules.cards import Rank, Suit
from server.training.relative_state import RelativeActor
from server.training.relative_state.relations import relative_actor

type DecisionKind = Literal[
    "bid", "stir", "discard", "lead_play", "follow_play"
]


@dataclass(frozen=True, slots=True)
class ActionQuery:
    """Player-visible decision shape for semantic action decoding."""

    kind: DecisionKind | None
    hand_faces: tuple[FaceCount, ...]
    pass_allowed: bool
    min_select: int
    max_select: int
    exact_select: int | None
    action_play_order: int | None
    current_trick_width: int | None
    lead_actor: RelativeActor | None
    discard_count: int | None
    trump_suit: Suit | None
    level_rank: Rank
    current_best_bid_role: RelativeActor | None


def build_action_query(
    *,
    player_index: int,
    snapshot: StateSnapshot,
) -> ActionQuery:
    """Build the structured player-visible decision request."""
    kind = _decision_kind(snapshot)
    hand_faces = canonical_face_counts(snapshot.player_hand)
    hand_size = face_count_width(hand_faces)
    pass_allowed = kind in ("bid", "stir")
    min_select, max_select, exact_select = _selection_shape(
        kind=kind,
        hand_size=hand_size,
        snapshot=snapshot,
    )
    action_play_order = _action_play_order(snapshot)
    current_trick_width = _current_trick_width(snapshot)
    return ActionQuery(
        kind=kind,
        hand_faces=hand_faces,
        pass_allowed=pass_allowed,
        min_select=min_select,
        max_select=max_select,
        exact_select=exact_select,
        action_play_order=action_play_order,
        current_trick_width=current_trick_width,
        lead_actor=_lead_actor(player_index, snapshot),
        discard_count=_discard_count(snapshot)
        if kind == "discard"
        else None,
        trump_suit=snapshot.trump_suit,
        level_rank=snapshot.trump_rank,
        current_best_bid_role=_current_best_bid_role(
            player_index, snapshot
        ),
    )


def _decision_kind(snapshot: StateSnapshot) -> DecisionKind | None:
    if snapshot.awaiting_action == "bid":
        return "bid"
    if snapshot.awaiting_action == "stir":
        return "stir"
    if snapshot.awaiting_action == "discard":
        return "discard"
    if snapshot.awaiting_action == "play":
        order = _action_play_order(snapshot)
        if order is None or order == 0:
            return "lead_play"
        return "follow_play"
    return None


def _selection_shape(
    *,
    kind: DecisionKind | None,
    hand_size: int,
    snapshot: StateSnapshot,
) -> tuple[int, int, int | None]:
    if kind is None:
        return 0, 0, None
    if kind in ("bid", "stir"):
        return (1 if hand_size > 0 else 0), min(4, hand_size), None
    if kind == "discard":
        count = min(_discard_count(snapshot), hand_size)
        return count, count, count
    if kind == "follow_play":
        width = _current_trick_width(snapshot)
        assert width is not None
        exact = min(width, hand_size)
        return exact, exact, exact
    return (1 if hand_size > 0 else 0), hand_size, None


def _discard_count(snapshot: StateSnapshot) -> int:
    stir = snapshot.stirring_state
    if stir is not None and stir.exchange_count is not None:
        return stir.exchange_count
    return 8


def _current_trick_width(snapshot: StateSnapshot) -> int | None:
    trick = snapshot.trick
    if trick is None:
        return None
    for slot in trick.slots:
        if slot.player == trick.lead_player and slot.cards:
            return len(slot.cards)
    return None


def _action_play_order(snapshot: StateSnapshot) -> int | None:
    trick = snapshot.trick
    if snapshot.awaiting_action != "play" or trick is None:
        return None
    return _play_order(
        lead_player=trick.lead_player, player=trick.current_player
    )


def _lead_actor(
    player_index: int, snapshot: StateSnapshot
) -> RelativeActor | None:
    trick = snapshot.trick
    if snapshot.awaiting_action != "play" or trick is None:
        return None
    return relative_actor(player_index, trick.lead_player)


def _current_best_bid_role(
    player_index: int, snapshot: StateSnapshot
) -> RelativeActor | None:
    winner = snapshot.bid_winner
    if winner is None:
        return None
    return relative_actor(player_index, winner.player)


def _play_order(*, lead_player: int, player: int) -> int:
    if player >= lead_player:
        return player - lead_player
    return 4 - lead_player + player
