"""Bid legal action space."""

from __future__ import annotations

from server.game.protocol import StateSnapshot
from server.game.rules import bid as bid_rules
from server.training.legal_actions.complete_trace import (
    CompleteTraceLegalActionIndex,
    pass_action,
    selection_action,
)
from server.training.semantic_actions.query import ActionQuery
from server.training.semantic_actions.values import GeneratedAction


def build_bid_index(
    *,
    player_index: int,
    snapshot: StateSnapshot,
    query: ActionQuery,
) -> CompleteTraceLegalActionIndex:
    """Build legal semantic traces for a bid decision."""
    actions: list[GeneratedAction] = [pass_action("bid")]
    if snapshot.bid_winner is None or (
        snapshot.bid_winner.player != player_index
    ):
        current_cards = (
            None
            if snapshot.bid_winner is None
            else snapshot.bid_winner.cards
        )
        for cards in bid_rules.legal_bid_hints(
            snapshot.player_hand,
            snapshot.trump_rank,
            current_cards,
        ):
            actions.append(selection_action("bid", cards))
    return CompleteTraceLegalActionIndex(query, tuple(actions))
