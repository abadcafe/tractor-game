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
from server.training.semantic_actions.choices import (
    ActionChoice,
    ActionTrace,
    InvalidActionRejected,
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
        self, trace: ActionTrace
    ) -> Ok[GeneratedAction] | Rejected:
        for action in self._actions:
            if action.trace == trace:
                return Ok(value=action)
        return InvalidActionRejected("动作不在当前规则合法集合内")


def pass_action(
    message_type: Literal["bid", "stir"],
) -> GeneratedAction:
    """Return the complete pass action for bid or stir."""
    trace = ActionTrace(choices=(ActionChoice("pass"),))
    return GeneratedAction(
        action_kind="pass",
        message_type=message_type,
        face_counts=(),
        trace=trace,
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
        trace=trace_for_selection(face_counts, include_finish=False),
        is_pass=False,
    )


def trace_for_selection(
    face_counts: tuple[FaceCount, ...], *, include_finish: bool
) -> ActionTrace:
    """Return the canonical trace for selected face counts."""
    choices = [
        ActionChoice("card", face_count) for face_count in face_counts
    ]
    if include_finish:
        choices.append(ActionChoice("finish"))
    return ActionTrace(choices=tuple(choices))


def _action_kind(
    message_type: Literal["bid", "stir"],
) -> PlayerActionKind:
    if message_type == "bid":
        return "bid"
    return "stir"
