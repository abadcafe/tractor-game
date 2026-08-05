"""Follow-play legal action space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

from server.foundation.result import Ok, Rejected
from server.game.rules import play
from server.game.rules.cards import Card
from server.game.snapshots import PlayerSnapshot
from server.game.snapshots.tricks import TrickSnapshot
from server.policy_model._schema.actions import (
    ActionPrefix,
    ActionTrace,
    InvalidActionRejected,
    action_prefix_cards,
)
from server.policy_model.actions.legality.contract import (
    LegalActionSpace,
)
from server.policy_model.actions.legality.selection import (
    trace_is_selection_only,
)
from server.policy_model.actions.semantic.values import GeneratedAction
from server.policy_model.observation.query import ActionQuery


@dataclass(slots=True)
class FollowPlayLegalActionSpace(LegalActionSpace):
    """Following action space using the full follow-rule validator."""

    _query: ActionQuery
    _space: play.FollowActionSpace

    @property
    @override
    def query(self) -> ActionQuery:
        return self._query

    @property
    def space(self) -> play.FollowActionSpace:
        """Return compiled follow constraints for this decision."""
        return self._space

    @override
    def decode(
        self, trace: ActionTrace
    ) -> Ok[GeneratedAction] | Rejected:
        if not trace_is_selection_only(trace):
            return InvalidActionRejected(
                "exact-count 动作不能包含终止参数"
            )
        selected_result = action_prefix_cards(
            ActionPrefix(choices=trace.choices)
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
                trace=trace,
                is_pass=False,
            )
        )


def build_follow_space(
    *,
    snapshot: PlayerSnapshot,
    query: ActionQuery,
) -> FollowPlayLegalActionSpace:
    """Build a legal action space for a follow-play decision."""
    lead_cards = _lead_cards(snapshot.trick)
    assert lead_cards
    space_result = play.build_follow_action_space(
        hand=list(snapshot.hand),
        lead_cards=list(lead_cards),
        trump_suit=query.trump_suit,
        trump_rank=query.level_rank,
    )
    assert isinstance(space_result, Ok)
    return FollowPlayLegalActionSpace(query, space_result.value)


def _lead_cards(trick: TrickSnapshot | None) -> tuple[Card, ...]:
    if trick is None:
        return ()
    for slot in trick.slots:
        if slot.actor == trick.lead_actor:
            return slot.cards
    return ()
