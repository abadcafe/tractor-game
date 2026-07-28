"""Stir legal action space."""

from __future__ import annotations

from server.game import Seat
from server.game.rules import bidding
from server.game.snapshots import PlayerSnapshot
from server.policy_model.actions.legality.complete_trace import (
    CompleteTraceLegalActionSpace,
    pass_action,
    selection_action,
)
from server.policy_model.actions.semantic.values import GeneratedAction
from server.policy_model.observation.query import ActionQuery


def build_stir_space(
    *,
    viewer: Seat,
    snapshot: PlayerSnapshot,
    query: ActionQuery,
) -> CompleteTraceLegalActionSpace:
    """Build legal semantic traces for a stir decision."""
    actions: list[GeneratedAction] = [pass_action("stir")]
    if _last_stir_actor(snapshot) != viewer:
        for declaration in bidding.legal_reveals(
            snapshot.hand,
            snapshot.trump_rank,
            _current_declaration(snapshot),
        ):
            if len(declaration.cards) != 2:
                continue
            actions.append(selection_action("stir", declaration.cards))
    return CompleteTraceLegalActionSpace(query, tuple(actions))


def _last_stir_actor(snapshot: PlayerSnapshot) -> Seat | None:
    for event in reversed(snapshot.stir_events):
        if event.kind == "stir":
            return event.actor
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
