"""Factory for semantic legal action indexes."""

from __future__ import annotations

from server.protocol import StateSnapshot
from server.training.legal_actions.bid import build_bid_index
from server.training.legal_actions.contract import (
    EmptyLegalActionIndex,
    LegalActionIndex,
)
from server.training.legal_actions.discard import (
    DiscardLegalActionIndex,
)
from server.training.legal_actions.follow import build_follow_index
from server.training.legal_actions.lead import LeadPlayLegalActionIndex
from server.training.legal_actions.stir import build_stir_index
from server.training.semantic_actions import (
    ActionQuery,
    build_action_query,
)


def build_legal_action_index(
    *,
    player_index: int,
    snapshot: StateSnapshot,
    query: ActionQuery | None = None,
) -> LegalActionIndex:
    """Build the rule-complete action index for a snapshot."""
    action_query = (
        build_action_query(player_index=player_index, snapshot=snapshot)
        if query is None
        else query
    )
    if action_query.kind is None:
        return EmptyLegalActionIndex(action_query)
    if action_query.kind == "bid":
        return build_bid_index(
            player_index=player_index,
            snapshot=snapshot,
            query=action_query,
        )
    if action_query.kind == "stir":
        return build_stir_index(
            player_index=player_index,
            snapshot=snapshot,
            query=action_query,
        )
    if action_query.kind == "discard":
        return DiscardLegalActionIndex(action_query)
    if action_query.kind == "lead_play":
        return LeadPlayLegalActionIndex(
            action_query, tuple(snapshot.player_hand)
        )
    if action_query.kind == "follow_play":
        return build_follow_index(snapshot=snapshot, query=action_query)
    assert False
