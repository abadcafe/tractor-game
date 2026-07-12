"""Follow-play legal action space."""

from __future__ import annotations

from dataclasses import dataclass

from server.foundation.result import Ok, Rejected
from server.game.protocol import StateSnapshot, TrickSnapshot
from server.game.rules.cards import Card
from server.game.rules.follow_action_space import (
    FollowActionSpace,
    build_follow_action_space,
)
from server.training.legal_actions.contract import LegalActionIndex
from server.training.legal_actions.selection import (
    trace_is_selection_only,
)
from server.training.semantic_actions.arguments import (
    InvalidSemanticActionRejected,
    SemanticArgumentPrefix,
    SemanticArgumentTrace,
    semantic_prefix_state,
)
from server.training.semantic_actions.query import ActionQuery
from server.training.semantic_actions.values import GeneratedAction


@dataclass(slots=True)
class FollowPlayLegalActionIndex(LegalActionIndex):
    """Following action space using the full follow-rule validator."""

    _query: ActionQuery
    _space: FollowActionSpace

    @property
    def query(self) -> ActionQuery:
        return self._query

    @property
    def space(self) -> FollowActionSpace:
        """Return compiled follow constraints for this decision."""
        return self._space

    def decode(
        self, trace: SemanticArgumentTrace
    ) -> Ok[GeneratedAction] | Rejected:
        if not trace_is_selection_only(trace):
            return InvalidSemanticActionRejected(
                "exact-count 动作不能包含终止参数"
            )
        selected_result = semantic_prefix_state(
            SemanticArgumentPrefix(arguments=trace.arguments)
        )
        if isinstance(selected_result, Rejected):
            return selected_result
        selected = selected_result.value
        decoded = self._space.decode(selected)
        if isinstance(decoded, Rejected):
            return decoded
        return Ok(
            value=GeneratedAction(
                action_kind="play",
                message_type="play",
                face_counts=selected,
                semantic_trace=trace,
                is_pass=False,
            )
        )


def build_follow_index(
    *,
    snapshot: StateSnapshot,
    query: ActionQuery,
) -> FollowPlayLegalActionIndex:
    """Build a legal action index for a follow-play decision."""
    lead_cards = _lead_cards(snapshot.trick)
    assert lead_cards
    space_result = build_follow_action_space(
        hand=snapshot.player_hand,
        lead_cards=lead_cards,
        trump_suit=query.trump_suit,
        trump_rank=query.level_rank,
    )
    assert isinstance(space_result, Ok)
    return FollowPlayLegalActionIndex(query, space_result.value)


def _lead_cards(trick: TrickSnapshot | None) -> list[Card]:
    if trick is None:
        return []
    for slot in trick.slots:
        if slot.player == trick.lead_player:
            return list(slot.cards)
    return []
