"""Local non-strategic decisions for AIPlayer."""

from __future__ import annotations

from server.protocol import PlayerMessage, StateSnapshot
from server.result import Ok, Rejected

type LocalDecision = Ok[PlayerMessage] | Rejected | None


def local_message(seq: int, snapshot: StateSnapshot) -> LocalDecision:
    """
    Return a local protocol decision, or None when LLM should decide.
    """
    awaiting = snapshot.awaiting_action
    if awaiting == "next_round":
        return Ok(PlayerMessage(seq=seq, raw={"type": "next_round"}))
    if awaiting == "bid":
        if snapshot.action_hints:
            cards = snapshot.action_hints[0]
            return Ok(
                PlayerMessage(
                    seq=seq,
                    raw={
                        "type": "bid",
                        "cards": [card.id for card in cards],
                    },
                )
            )
        return Ok(
            PlayerMessage(seq=seq, raw={"type": "bid", "pass": True})
        )
    if awaiting == "stir" and not snapshot.action_hints:
        return Ok(
            PlayerMessage(seq=seq, raw={"type": "stir", "pass": True})
        )
    return None
