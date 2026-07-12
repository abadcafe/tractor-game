"""Legal action index contract shared by training policies."""

from __future__ import annotations

from dataclasses import dataclass

from server.foundation.result import Ok, Rejected
from server.training.semantic_actions.arguments import (
    InvalidSemanticActionRejected,
    SemanticArgumentTrace,
)
from server.training.semantic_actions.query import ActionQuery
from server.training.semantic_actions.values import GeneratedAction


class LegalActionIndex:
    """Rule-complete next-argument mask for one player decision."""

    @property
    def query(self) -> ActionQuery:
        """Return the action query this legal index answers."""
        raise NotImplementedError

    def decode(
        self, trace: SemanticArgumentTrace
    ) -> Ok[GeneratedAction] | Rejected:
        """Decode a complete legal trace into a generated action."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class EmptyLegalActionIndex(LegalActionIndex):
    """No action is legal because the snapshot awaits nothing."""

    _query: ActionQuery

    @property
    def query(self) -> ActionQuery:
        return self._query

    def decode(
        self, trace: SemanticArgumentTrace
    ) -> Ok[GeneratedAction] | Rejected:
        return InvalidSemanticActionRejected("当前没有动作请求")
