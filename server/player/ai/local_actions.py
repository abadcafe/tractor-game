"""Local non-strategic decisions for AIPlayer."""

from __future__ import annotations

from server.messages import PlayerMessage
from server.player.ai.client import AIToolCall
from server.player.ai.tools import tool_call_to_message
from server.sm.card_model import Card
from server.sm.comparator import bid_value
from server.sm.result import Ok, Rejected
from server.snapshot import StateSnapshot

type LocalDecision = Ok[PlayerMessage] | Rejected | None


def local_message(seq: int, snapshot: StateSnapshot) -> LocalDecision:
    """Return a local protocol decision, or None when LLM should decide."""
    awaiting = snapshot.awaiting_action
    if awaiting == "next_round":
        return tool_call_to_message(seq, snapshot, AIToolCall(
            name="confirm_next_round",
            arguments={"reason": "local confirm"},
        ))
    if awaiting == "bid":
        if snapshot.action_hints:
            cards = _smallest_bid_hint(snapshot)
            return tool_call_to_message(seq, snapshot, AIToolCall(
                name="bid_trump",
                arguments={"card_ids": [card.id for card in cards], "reason": "local smallest bid hint"},
            ))
        return tool_call_to_message(seq, snapshot, AIToolCall(
            name="pass_bid",
            arguments={"reason": "local pass bid"},
        ))
    if awaiting == "stir" and not snapshot.action_hints:
        return tool_call_to_message(seq, snapshot, AIToolCall(
            name="pass_stir",
            arguments={"reason": "local pass stir without hints"},
        ))
    return None


def _smallest_bid_hint(snapshot: StateSnapshot) -> list[Card]:
    return min(
        snapshot.action_hints,
        key=lambda cards: (
            bid_value(cards, snapshot.trump_rank),
            tuple(sorted(card.id for card in cards)),
        ),
    )
