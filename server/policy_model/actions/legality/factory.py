"""Factory for semantic legal action spaces."""

from __future__ import annotations

from server.game import Seat
from server.game.snapshots import PlayerSnapshot
from server.policy_model.actions.legality.bid import build_bid_space
from server.policy_model.actions.legality.contract import (
    EmptyLegalActionSpace,
    LegalActionSpace,
)
from server.policy_model.actions.legality.discard import (
    DiscardLegalActionSpace,
)
from server.policy_model.actions.legality.follow import (
    build_follow_space,
)
from server.policy_model.actions.legality.lead import (
    LeadPlayLegalActionSpace,
)
from server.policy_model.actions.legality.stir import build_stir_space
from server.policy_model.observation.query import (
    ActionQuery,
    build_action_query,
)


def build_legal_action_space(
    *,
    viewer: Seat,
    snapshot: PlayerSnapshot,
    query: ActionQuery | None = None,
) -> LegalActionSpace:
    """Build the rule-complete action space for a snapshot."""
    action_query = (
        build_action_query(viewer=viewer, snapshot=snapshot)
        if query is None
        else query
    )
    if action_query.kind is None:
        return EmptyLegalActionSpace(action_query)
    if action_query.kind == "bid":
        return build_bid_space(
            viewer=viewer,
            snapshot=snapshot,
            query=action_query,
        )
    if action_query.kind == "stir":
        return build_stir_space(
            viewer=viewer,
            snapshot=snapshot,
            query=action_query,
        )
    if action_query.kind == "discard":
        return DiscardLegalActionSpace(action_query)
    if action_query.kind == "lead_play":
        return LeadPlayLegalActionSpace(
            action_query, tuple(snapshot.hand)
        )
    if action_query.kind == "follow_play":
        return build_follow_space(snapshot=snapshot, query=action_query)
    assert False
