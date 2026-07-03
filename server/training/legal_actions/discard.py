"""Discard legal action space."""

from __future__ import annotations

from dataclasses import dataclass

from server.result import Ok, Rejected
from server.rules.card_faces import FaceCount, face_count_width
from server.training.legal_actions.contract import LegalActionIndex
from server.training.legal_actions.selection import (
    remaining_count_after_selected,
    select_arguments,
    trace_is_selection_only,
)
from server.training.semantic_actions import (
    ActionQuery,
    GeneratedAction,
    InvalidSemanticActionRejected,
    SemanticArgument,
    SemanticArgumentPrefix,
    SemanticArgumentTrace,
    semantic_prefix_state,
)


@dataclass(slots=True)
class DiscardLegalActionIndex(LegalActionIndex):
    """Exact-card-count discard action space."""

    _query: ActionQuery

    @property
    def query(self) -> ActionQuery:
        return self._query

    def allowed_next(
        self, prefix: SemanticArgumentPrefix
    ) -> tuple[SemanticArgument, ...]:
        selected_result = semantic_prefix_state(prefix)
        if isinstance(selected_result, Rejected):
            return ()
        selected = selected_result.value
        selected_count = face_count_width(selected)
        if selected_count >= self._required_count():
            return ()
        return select_arguments(
            query=self._query,
            selected=selected,
            can_complete=self._discard_can_complete,
        )

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

    def _discard_can_complete(
        self, selected: tuple[FaceCount, ...]
    ) -> bool:
        selected_count = face_count_width(selected)
        if selected_count > self._required_count():
            return False
        if selected_count == self._required_count():
            return True
        return (
            remaining_count_after_selected(
                hand_faces=self._query.hand_faces,
                selected=selected,
            )
            >= self._required_count() - selected_count
        )

    def _required_count(self) -> int:
        assert self._query.exact_select is not None
        return self._query.exact_select
