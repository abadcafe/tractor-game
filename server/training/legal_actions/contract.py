"""Legal action index contract shared by training policies."""

from __future__ import annotations

from dataclasses import dataclass

from server.result import Ok, Rejected
from server.training.semantic_actions import (
    ActionQuery,
    GeneratedAction,
    InvalidSemanticActionRejected,
    SemanticArgument,
    SemanticArgumentPrefix,
    SemanticArgumentTrace,
)


class LegalActionIndex:
    """Rule-complete next-argument mask for one player decision."""

    @property
    def query(self) -> ActionQuery:
        """Return the action query this legal index answers."""
        raise NotImplementedError

    def allowed_next(
        self, prefix: SemanticArgumentPrefix
    ) -> tuple[SemanticArgument, ...]:
        """Return legal next semantic arguments after the prefix."""
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

    def allowed_next(
        self, prefix: SemanticArgumentPrefix
    ) -> tuple[SemanticArgument, ...]:
        return ()

    def decode(
        self, trace: SemanticArgumentTrace
    ) -> Ok[GeneratedAction] | Rejected:
        return InvalidSemanticActionRejected("当前没有动作请求")
