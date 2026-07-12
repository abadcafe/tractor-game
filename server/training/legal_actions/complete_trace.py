"""Legal indexes backed by complete semantic traces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from server.foundation.result import Ok, Rejected
from server.game.rules.card_faces import (
    FaceCount,
    canonical_face_counts,
)
from server.game.rules.cards import Card
from server.training.legal_actions.contract import (
    LegalActionIndex,
)
from server.training.semantic_actions.arguments import (
    InvalidSemanticActionRejected,
    SemanticArgument,
    SemanticArgumentTrace,
)
from server.training.semantic_actions.query import ActionQuery
from server.training.semantic_actions.values import (
    GeneratedAction,
    PlayerActionKind,
)


@dataclass(frozen=True, slots=True)
class CompleteTraceLegalActionIndex(LegalActionIndex):
    """Legal index backed by a closed set of complete action traces."""

    _query: ActionQuery
    _actions: tuple[GeneratedAction, ...]

    @property
    def query(self) -> ActionQuery:
        return self._query

    @property
    def actions(self) -> tuple[GeneratedAction, ...]:
        """Return the closed complete action set."""
        return self._actions

    def decode(
        self, trace: SemanticArgumentTrace
    ) -> Ok[GeneratedAction] | Rejected:
        for action in self._actions:
            if action.semantic_trace == trace:
                return Ok(value=action)
        return InvalidSemanticActionRejected(
            "动作不在当前规则合法集合内"
        )


def pass_action(
    message_type: Literal["bid", "stir"],
) -> GeneratedAction:
    """Return the complete pass action for bid or stir."""
    trace = SemanticArgumentTrace(arguments=(SemanticArgument("pass"),))
    return GeneratedAction(
        action_kind="pass",
        message_type=message_type,
        face_counts=(),
        semantic_trace=trace,
        is_pass=True,
    )


def selection_action(
    message_type: Literal["bid", "stir"], cards: Sequence[Card]
) -> GeneratedAction:
    """Return a complete face-count selection for bid or stir."""
    face_counts = canonical_face_counts(tuple(cards))
    return GeneratedAction(
        action_kind=_action_kind(message_type),
        message_type=message_type,
        face_counts=face_counts,
        semantic_trace=trace_for_selection(
            face_counts, include_stop=True
        ),
        is_pass=False,
    )


def trace_for_selection(
    face_counts: tuple[FaceCount, ...], *, include_stop: bool
) -> SemanticArgumentTrace:
    """Return the canonical trace for selected face counts."""
    arguments = [
        SemanticArgument("select_face_count", face_count)
        for face_count in face_counts
    ]
    if include_stop:
        arguments.append(SemanticArgument("stop"))
    return SemanticArgumentTrace(arguments=tuple(arguments))


def _action_kind(
    message_type: Literal["bid", "stir"],
) -> PlayerActionKind:
    if message_type == "bid":
        return "bid"
    return "stir"
