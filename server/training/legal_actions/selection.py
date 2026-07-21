"""Face-count selection grammar for semantic legal actions."""

from __future__ import annotations

from collections.abc import Sequence

from server.foundation.result import Ok, Rejected
from server.game.rules.card_faces import FaceCount
from server.game.rules.cards import Card
from server.training.semantic_actions.choices import (
    ActionTrace,
    InvalidActionRejected,
)


def cards_for_face_counts(
    face_counts: tuple[FaceCount, ...],
    hand_cards: Sequence[Card],
) -> Ok[list[Card]] | Rejected:
    """Bind selected semantic faces to concrete hand cards."""
    result: list[Card] = []
    used_ids: set[str] = set()
    for requested in face_counts:
        matching = [
            card
            for card in hand_cards
            if card.id not in used_ids
            and card.suit == requested.face.suit
            and card.rank == requested.face.rank
        ]
        if len(matching) < requested.count:
            return InvalidActionRejected("当前手牌没有足够的指定牌面")
        selected = matching[: requested.count]
        result.extend(selected)
        used_ids.update(card.id for card in selected)
    return Ok(value=result)


def trace_is_selection_only(trace: ActionTrace) -> bool:
    """Return whether a trace contains only face-count selections."""
    return all(choice.kind == "card" for choice in trace.choices)
