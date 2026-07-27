"""Conversion between engine commands and semantic model actions."""

from __future__ import annotations

from server.foundation.result import Ok, Rejected
from server.game import commands
from server.game.rules.cards import Card
from server.game.rules.cards.faces import canonical_face_counts
from server.training.legal_actions import LegalActionIndex
from server.training.legal_actions.complete_trace import (
    trace_for_selection,
)
from server.training.semantic_actions import GeneratedAction
from server.training.semantic_actions.binding import (
    bind_generated_action,
)
from server.training.semantic_actions.choices import (
    ActionChoice,
    ActionTrace,
)


def semantic_action(
    *,
    command: commands.Command,
    hand: tuple[Card, ...],
    legal_actions: LegalActionIndex,
) -> Ok[GeneratedAction] | Rejected:
    """Resolve one physical command through the model action grammar."""
    if isinstance(command, commands.PassBid):
        trace = ActionTrace((ActionChoice("pass"),))
    elif isinstance(command, commands.PassStir):
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
        include_finish = legal_actions.query.kind == "lead_play"
        trace = trace_for_selection(
            canonical_face_counts(cards_result.value),
            include_finish=include_finish,
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


__all__ = ("physical_command", "semantic_action")
