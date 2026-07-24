"""Bind semantic generated actions to physical card ids."""

from __future__ import annotations

from server.foundation.result import Ok, Rejected
from server.game import commands
from server.game.rules.cards import Card, CardId
from server.game.rules.cards.faces import (
    CardFace,
    FaceCount,
    bind_face_counts,
)
from server.training.semantic_actions.choices import (
    InvalidActionRejected,
)
from server.training.semantic_actions.values import GeneratedAction


def bind_generated_action(
    action: GeneratedAction,
    hand_cards: list[Card] | tuple[Card, ...],
) -> Ok[commands.Command] | Rejected:
    """Bind a semantic action to one typed game command."""
    if action.is_pass:
        assert action.action_kind == "pass"
        if action.message_type == "bid":
            return Ok(value=commands.PassBid())
        assert action.message_type == "stir"
        return Ok(value=commands.PassStir())
    card_ids_result = _card_ids_for_face_counts(
        action.face_counts, hand_cards
    )
    if isinstance(card_ids_result, Rejected):
        return card_ids_result
    card_ids = card_ids_result.value
    if action.message_type == "bid":
        return Ok(value=commands.RevealBid(card_ids))
    if action.message_type == "stir":
        return Ok(value=commands.Stir(card_ids))
    if action.message_type == "discard":
        return Ok(value=commands.Bury(card_ids))
    assert action.message_type == "play"
    return Ok(value=commands.Play(card_ids))


def _card_ids_for_face_counts(
    face_counts: tuple[FaceCount, ...],
    hand_cards: list[Card] | tuple[Card, ...],
) -> Ok[tuple[CardId, ...]] | Rejected:
    unique_check = _validate_unique_faces(face_counts)
    if isinstance(unique_check, Rejected):
        return unique_check
    bound = bind_face_counts(face_counts, hand_cards)
    if isinstance(bound, Rejected):
        return InvalidActionRejected(bound.reason)
    return Ok(value=tuple(CardId(card.id) for card in bound.value))


def _validate_unique_faces(
    face_counts: tuple[FaceCount, ...],
) -> Ok[None] | Rejected:
    seen: set[CardFace] = set()
    for item in face_counts:
        if item.face in seen:
            return InvalidActionRejected("同一牌面重复选择")
        seen.add(item.face)
    return Ok(value=None)
