"""Stir legal action space."""

from __future__ import annotations

from server.game.rules import bidding
from server.game.snapshots import PlayerSnapshot
from server.training.legal_actions.complete_trace import (
    CompleteTraceLegalActionIndex,
    pass_action,
    selection_action,
)
from server.training.semantic_actions.query import ActionQuery
from server.training.semantic_actions.values import GeneratedAction


def build_stir_index(
    *,
    player_index: int,
    snapshot: PlayerSnapshot,
    query: ActionQuery,
) -> CompleteTraceLegalActionIndex:
    """Build legal semantic traces for a stir decision."""
    actions: list[GeneratedAction] = [pass_action("stir")]
    if _last_stir_player(snapshot) != player_index:
        for declaration in bidding.legal_reveals(
            snapshot.player_hand,
            snapshot.trump_rank,
            _current_declaration(snapshot),
        ):
            if len(declaration.cards) != 2:
                continue
            actions.append(selection_action("stir", declaration.cards))
    return CompleteTraceLegalActionIndex(query, tuple(actions))


def _last_stir_player(snapshot: PlayerSnapshot) -> int | None:
    for event in reversed(snapshot.stir_events):
        if event.kind == "stir":
            return event.player
    return None


def _current_declaration(
    snapshot: PlayerSnapshot,
) -> bidding.Declaration | None:
    for event in reversed(snapshot.stir_events):
        if event.kind == "stir":
            return bidding.Declaration(event.cards)
    if snapshot.bid_winner is None:
        return None
    return bidding.Declaration(snapshot.bid_winner.cards)
