"""Discard legal action space."""

from __future__ import annotations

from dataclasses import dataclass

from server.foundation.result import Ok, Rejected
from server.game.rules.card_faces import face_count_width
from server.training.legal_actions.contract import LegalActionIndex
from server.training.legal_actions.selection import (
    trace_is_selection_only,
)
from server.training.semantic_actions.choices import (
    ActionPrefix,
    ActionTrace,
    InvalidActionRejected,
    action_prefix_cards,
)
from server.training.semantic_actions.query import ActionQuery
from server.training.semantic_actions.values import GeneratedAction


@dataclass(slots=True)
class DiscardLegalActionIndex(LegalActionIndex):
    """Exact-card-count discard action space."""

    _query: ActionQuery

    @property
    def query(self) -> ActionQuery:
        return self._query

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
        if face_count_width(selected) != self._required_count():
            return InvalidActionRejected("埋牌数量不满足规则")
        return Ok(
            value=GeneratedAction(
                action_kind="discard",
                message_type="discard",
                face_counts=selected,
                trace=trace,
                is_pass=False,
            )
        )

    def _required_count(self) -> int:
        assert self._query.exact_select is not None
        return self._query.exact_select
