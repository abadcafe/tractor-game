"""Bid legal action space."""

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


def build_bid_space(
    *,
    viewer: Seat,
    snapshot: PlayerSnapshot,
    query: ActionQuery,
) -> CompleteTraceLegalActionSpace:
    """Build legal semantic traces for a bid decision."""
    actions: list[GeneratedAction] = [pass_action("bid")]
    if snapshot.bid_winner is None or (
        snapshot.bid_winner.actor != viewer
    ):
        current = (
            None
            if snapshot.bid_winner is None
            else bidding.Declaration(snapshot.bid_winner.cards)
        )
        for declaration in bidding.legal_reveals(
            snapshot.hand,
            snapshot.trump_rank,
            current,
        ):
            actions.append(selection_action("bid", declaration.cards))
    return CompleteTraceLegalActionSpace(query, tuple(actions))
