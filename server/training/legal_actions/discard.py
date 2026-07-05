"""Discard legal action space."""

from __future__ import annotations

from dataclasses import dataclass

from server.result import Ok, Rejected
from server.rules.card_faces import face_count_width
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
class DiscardLegalActionIndex(LegalActionIndex):
    """Exact-card-count discard action space."""

    _query: ActionQuery

    @property
    def query(self) -> ActionQuery:
        return self._query

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
        if face_count_width(selected) != self._required_count():
            return InvalidSemanticActionRejected("埋牌数量不满足规则")
        return Ok(
            value=GeneratedAction(
                action_kind="discard",
                message_type="discard",
                face_counts=selected,
                semantic_trace=trace,
                is_pass=False,
            )
        )

    def _required_count(self) -> int:
        assert self._query.exact_select is not None
        return self._query.exact_select
