"""Stir legal action space."""

from __future__ import annotations

from server.protocol import StateSnapshot
from server.rules import bid as bid_rules
from server.rules.ordering import bid_value
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
    snapshot: StateSnapshot,
    query: ActionQuery,
) -> CompleteTraceLegalActionIndex:
    """Build legal semantic traces for a stir decision."""
    actions: list[GeneratedAction] = [pass_action("stir")]
    if _last_stir_player(snapshot) != player_index:
        current_priority = _current_stir_priority(snapshot)
        for candidate in bid_rules.bid_card_candidates(
            snapshot.player_hand,
            snapshot.trump_rank,
        ):
            if len(candidate) != 2:
                continue
            if bid_value(candidate, snapshot.trump_rank) <= (
                current_priority
            ):
                continue
            actions.append(selection_action("stir", candidate))
    return CompleteTraceLegalActionIndex(query, tuple(actions))


def _last_stir_player(snapshot: StateSnapshot) -> int | None:
    for event in reversed(snapshot.stir_events):
        if event.kind == "stir":
            return event.player
    return None


def _current_stir_priority(snapshot: StateSnapshot) -> int:
    current_priority = (
        0
        if snapshot.bid_winner is None
        else bid_value(snapshot.bid_winner.cards, snapshot.trump_rank)
    )
    for event in snapshot.stir_events:
        if event.kind == "pass":
            continue
        assert event.kind == "stir"
        assert event.priority is not None
        current_priority = max(current_priority, event.priority)
    return current_priority
