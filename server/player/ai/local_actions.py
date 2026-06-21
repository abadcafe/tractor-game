"""Local non-strategic decisions for AIPlayer."""

from __future__ import annotations

from server.protocol import PlayerMessage, StateSnapshot
from server.result import Ok, Rejected

type LocalDecision = Ok[PlayerMessage] | Rejected | None

BID_LLM_DECISION_HAND_COUNTS: frozenset[int] = frozenset(
    {5, 11, 17, 23}
)


def local_message(seq: int, snapshot: StateSnapshot) -> LocalDecision:
    """
    Return a local protocol decision, or None when LLM should decide.
    """
    awaiting = snapshot.awaiting_action
    if awaiting == "next_round":
        return Ok(PlayerMessage(seq=seq, raw={"type": "next_round"}))
    if awaiting == "bid":
        if _should_ask_llm_for_bid(snapshot):
            return None
        return Ok(
            PlayerMessage(seq=seq, raw={"type": "bid", "pass": True})
        )
    if awaiting == "stir" and not snapshot.action_hints:
        return Ok(
            PlayerMessage(seq=seq, raw={"type": "stir", "pass": True})
        )
    return None


def _should_ask_llm_for_bid(snapshot: StateSnapshot) -> bool:
    hand_count = len(snapshot.player_hand)
    return hand_count in BID_LLM_DECISION_HAND_COUNTS and bool(
        snapshot.action_hints
    )
