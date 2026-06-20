"""Local non-strategic decisions for AIPlayer."""

from __future__ import annotations

from server.messages import PlayerMessage
from server.player.ai.client import AIToolCall
from server.player.ai.tools import tool_call_to_message
from server.sm.card_model import Card
from server.sm.comparator import bid_value
from server.result import Ok, Rejected
from server.snapshot import StateSnapshot

type LocalDecision = Ok[PlayerMessage] | Rejected | None


def local_message(seq: int, snapshot: StateSnapshot) -> LocalDecision:
    """Return a local protocol decision, or None when LLM should decide."""
    awaiting = snapshot.awaiting_action
    if awaiting == "next_round":
        return tool_call_to_message(seq, snapshot, AIToolCall(
            name="confirm_next_round",
            arguments={"reason": "本地确认"},
        ))
    if awaiting == "bid":
        if snapshot.action_hints:
            cards = _smallest_bid_hint(snapshot)
            return tool_call_to_message(seq, snapshot, AIToolCall(
                name="bid_trump",
                arguments={"card_ids": [card.id for card in cards], "reason": "本地选择最小抢主提示"},
            ))
        return tool_call_to_message(seq, snapshot, AIToolCall(
            name="pass_bid",
            arguments={"reason": "本地不抢"},
        ))
    if awaiting == "stir" and not snapshot.action_hints:
        return tool_call_to_message(seq, snapshot, AIToolCall(
            name="pass_stir",
            arguments={"reason": "本地不反"},
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
