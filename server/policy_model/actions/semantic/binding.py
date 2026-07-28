"""Bind semantic generated actions to physical card ids."""

from __future__ import annotations

from server.foundation.result import Ok, Rejected
from server.game import commands
from server.game.rules.cards import Card, CardId
from server.game.rules.cards.faces import (
    CardFace,
    FaceCount,
    bind_face_counts,
    canonical_face_counts,
)
from server.policy_model._schema.actions import (
    ActionChoice,
    ActionTrace,
    InvalidActionRejected,
)
from server.policy_model.actions.legality.complete_trace import (
    trace_for_selection,
)
from server.policy_model.actions.legality.contract import (
    LegalActionSpace,
)
from server.policy_model.actions.semantic.values import GeneratedAction


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


def semantic_action(
    *,
    command: commands.Command,
    hand: tuple[Card, ...],
    legal_actions: LegalActionSpace,
) -> Ok[GeneratedAction] | Rejected:
    """Resolve one physical command through the model action grammar."""
    if isinstance(command, (commands.PassBid, commands.PassStir)):
        trace = ActionTrace((ActionChoice("pass"),))
    elif isinstance(
        command,
        (
            commands.RevealBid,
            commands.Stir,
            commands.Bury,
            commands.Play,
        ),
    ):
        cards_result = _command_cards(command=command, hand=hand)
        if isinstance(cards_result, Rejected):
            return cards_result
        trace = trace_for_selection(
            canonical_face_counts(cards_result.value),
            include_finish=legal_actions.query.kind == "lead_play",
        )
    else:
        return Rejected(reason="command is not a model action")
    return legal_actions.decode(trace)


def physical_command(
    *,
    action: GeneratedAction,
    hand: tuple[Card, ...],
) -> Ok[commands.Command] | Rejected:
    """Bind one semantic model action to physical cards."""
    return bind_generated_action(action, hand)


def _command_cards(
    *,
    command: (
        commands.RevealBid
        | commands.Stir
        | commands.Bury
        | commands.Play
    ),
    hand: tuple[Card, ...],
) -> Ok[tuple[Card, ...]] | Rejected:
    by_id = {card.id: card for card in hand}
    result: list[Card] = []
    seen: set[str] = set()
    for card_id in command.card_ids:
        if card_id in seen:
            return Rejected(reason="command repeats one physical card")
        card = by_id.get(card_id)
        if card is None:
            return Rejected(reason="command card is absent from hand")
        seen.add(card_id)
        result.append(card)
    return Ok(tuple(result))


def _validate_unique_faces(
    face_counts: tuple[FaceCount, ...],
) -> Ok[None] | Rejected:
    seen: set[CardFace] = set()
    for item in face_counts:
        if item.face in seen:
            return InvalidActionRejected("同一牌面重复选择")
        seen.add(item.face)
    return Ok(value=None)
