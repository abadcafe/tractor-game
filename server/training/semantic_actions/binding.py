"""Bind semantic generated actions to physical card ids."""

from __future__ import annotations

from server.foundation.result import Ok, Rejected
from server.game.rules.card_faces import (
    CardFace,
    FaceCount,
    bind_face_counts,
)
from server.game.rules.cards import Card
from server.training.semantic_actions.arguments import (
    InvalidSemanticActionRejected,
)
from server.training.semantic_actions.values import (
    BoundAction,
    GeneratedAction,
)


def bind_generated_action(
    action: GeneratedAction,
    hand_cards: list[Card] | tuple[Card, ...],
) -> Ok[BoundAction] | Rejected:
    """Bind a semantic action to current physical card ids."""
    if action.is_pass:
        return Ok(
            value=BoundAction(
                raw={"type": action.message_type, "pass": True},
            )
        )
    card_ids_result = _card_ids_for_face_counts(
        action.face_counts, hand_cards
    )
    if isinstance(card_ids_result, Rejected):
        return card_ids_result
    return Ok(
        value=BoundAction(
            raw={
                "type": action.message_type,
                "cards": list(card_ids_result.value),
            },
        )
    )


def _card_ids_for_face_counts(
    face_counts: tuple[FaceCount, ...],
    hand_cards: list[Card] | tuple[Card, ...],
) -> Ok[tuple[str, ...]] | Rejected:
    unique_check = _validate_unique_faces(face_counts)
    if isinstance(unique_check, Rejected):
        return unique_check
    bound = bind_face_counts(face_counts, hand_cards)
    if isinstance(bound, Rejected):
        return InvalidSemanticActionRejected(bound.reason)
    return Ok(value=tuple(card.id for card in bound.value))


def _validate_unique_faces(
    face_counts: tuple[FaceCount, ...],
) -> Ok[None] | Rejected:
    seen: set[CardFace] = set()
    for item in face_counts:
        if item.face in seen:
            return InvalidSemanticActionRejected("同一牌面重复选择")
        seen.add(item.face)
    return Ok(value=None)
