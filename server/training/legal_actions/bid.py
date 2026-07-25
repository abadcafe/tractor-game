"""Bid legal action space."""

from __future__ import annotations

from server.game import Seat
from server.game.rules import bidding
from server.game.snapshots import PlayerSnapshot
from server.training.legal_actions.complete_trace import (
    CompleteTraceLegalActionIndex,
    pass_action,
    selection_action,
)
from server.training.semantic_actions.query import ActionQuery
from server.training.semantic_actions.values import GeneratedAction


def build_bid_index(
    *,
    viewer: Seat,
    snapshot: PlayerSnapshot,
    query: ActionQuery,
) -> CompleteTraceLegalActionIndex:
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
    return CompleteTraceLegalActionIndex(query, tuple(actions))
