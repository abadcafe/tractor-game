"""Tool schemas and validation for AIPlayer."""

from __future__ import annotations

from server.messages import PlayerMessage
from server.player.ai.client import AIToolCall, AIToolSpec, JSONValue
from server.sm.result import Ok, Rejected
from server.snapshot import StateSnapshot


def allowed_tool_specs(snapshot: StateSnapshot) -> list[AIToolSpec]:
    awaiting = snapshot.awaiting_action
    if awaiting == "next_round":
        return [_no_card_tool("confirm_next_round", "Confirm readiness for game start or next round.")]
    if awaiting == "bid":
        allowed_ids = _hint_card_ids(snapshot) or _hand_card_ids(snapshot)
        return [
            _card_tool("bid_trump", "Reveal cards to bid for trump.", allowed_ids),
            _no_card_tool("pass_bid", "Pass during deal-bid phase."),
        ]
    if awaiting == "stir":
        allowed_ids = _hint_card_ids(snapshot) or _hand_card_ids(snapshot)
        return [
            _card_tool("stir_trump", "Reveal cards to stir/change trump.", allowed_ids),
            _no_card_tool("pass_stir", "Pass during stirring phase."),
        ]
    if awaiting == "discard":
        return [_card_tool("discard_bottom", "Discard cards into the bottom pile.", _hand_card_ids(snapshot))]
    if awaiting == "play":
        allowed_ids = _hint_card_ids(snapshot) or _hand_card_ids(snapshot)
        return [_card_tool(
            "play_cards",
            "Play one or more cards in the current trick. "
            "If action_hints are present, card_ids must exactly match one complete hint group.",
            allowed_ids,
        )]
    return []


def tool_call_to_message(
    seq: int,
    snapshot: StateSnapshot,
    call: AIToolCall,
) -> Ok[PlayerMessage] | Rejected:
    awaiting = snapshot.awaiting_action
    if call.name == "confirm_next_round":
        if awaiting != "next_round":
            return Rejected("confirm_next_round is not allowed now")
        return Ok(PlayerMessage(seq=seq, raw={"type": "next_round"}))

    if call.name == "pass_bid":
        if awaiting != "bid":
            return Rejected("pass_bid is not allowed now")
        return Ok(PlayerMessage(seq=seq, raw={"type": "bid", "pass": True}))

    if call.name == "pass_stir":
        if awaiting != "stir":
            return Rejected("pass_stir is not allowed now")
        return Ok(PlayerMessage(seq=seq, raw={"type": "stir", "pass": True}))

    card_ids_result = _card_ids(call)
    if isinstance(card_ids_result, Rejected):
        return card_ids_result
    card_ids = card_ids_result.value
    hand_result = _validate_card_ids_in_hand(snapshot, card_ids)
    if isinstance(hand_result, Rejected):
        return hand_result

    if call.name == "bid_trump":
        if awaiting != "bid":
            return Rejected("bid_trump is not allowed now")
        if snapshot.action_hints and not _matches_hint(card_ids, snapshot):
            return Rejected("bid_trump card_ids do not match action_hints")
        return Ok(PlayerMessage(seq=seq, raw={"type": "bid", "cards": card_ids}))

    if call.name == "stir_trump":
        if awaiting != "stir":
            return Rejected("stir_trump is not allowed now")
        if snapshot.action_hints and not _matches_hint(card_ids, snapshot):
            return Rejected("stir_trump card_ids do not match action_hints")
        return Ok(PlayerMessage(seq=seq, raw={"type": "stir", "cards": card_ids}))

    if call.name == "discard_bottom":
        if awaiting != "discard":
            return Rejected("discard_bottom is not allowed now")
        expected = snapshot.stirring_state.exchange_count if snapshot.stirring_state is not None else None
        if expected is None:
            return Rejected("discard_bottom missing exchange_count")
        if len(card_ids) != expected:
            return Rejected(f"discard_bottom requires {expected} cards")
        return Ok(PlayerMessage(seq=seq, raw={"type": "discard", "cards": card_ids}))

    if call.name == "play_cards":
        if awaiting != "play":
            return Rejected("play_cards is not allowed now")
        if not card_ids:
            return Rejected("play_cards requires at least one card")
        if snapshot.action_hints and not _matches_hint(card_ids, snapshot):
            return Rejected("play_cards card_ids do not match action_hints")
        return Ok(PlayerMessage(seq=seq, raw={"type": "play", "cards": card_ids}))

    return Rejected(f"unknown AI tool: {call.name}")


def _no_card_tool(name: str, description: str) -> AIToolSpec:
    return AIToolSpec(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Short reason for logs."},
            },
            "required": ["reason"],
            "additionalProperties": False,
        },
    )


def _card_tool(name: str, description: str, allowed_card_ids: list[str]) -> AIToolSpec:
    item_schema: dict[str, JSONValue] = {"type": "string"}
    if allowed_card_ids:
        enum_values: list[JSONValue] = [card_id for card_id in allowed_card_ids]
        item_schema["enum"] = enum_values
    return AIToolSpec(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "card_ids": {
                    "type": "array",
                    "items": item_schema,
                    "description": "Card ids copied exactly from the current legal card id enum.",
                },
                "reason": {"type": "string", "description": "Short reason for logs."},
            },
            "required": ["card_ids", "reason"],
            "additionalProperties": False,
        },
    )


def _hand_card_ids(snapshot: StateSnapshot) -> list[str]:
    return [card.id for card in snapshot.player_hand]


def _hint_card_ids(snapshot: StateSnapshot) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for hint in snapshot.action_hints:
        for card in hint:
            if card.id not in seen:
                seen.add(card.id)
                result.append(card.id)
    return result


def _card_ids(call: AIToolCall) -> Ok[list[str]] | Rejected:
    raw = call.arguments.get("card_ids")
    if not isinstance(raw, list):
        return Rejected(f"{call.name} requires card_ids")
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            return Rejected("card_ids must contain only strings")
        if item in seen:
            return Rejected(f"duplicate card id: {item}")
        seen.add(item)
        result.append(item)
    return Ok(result)


def _validate_card_ids_in_hand(snapshot: StateSnapshot, card_ids: list[str]) -> Ok[None] | Rejected:
    hand_ids = {card.id for card in snapshot.player_hand}
    for card_id in card_ids:
        if card_id not in hand_ids:
            return Rejected(f"card id not in hand: {card_id}")
    return Ok(None)


def _matches_hint(card_ids: list[str], snapshot: StateSnapshot) -> bool:
    key = tuple(sorted(card_ids))
    return any(tuple(sorted(card.id for card in hint)) == key for hint in snapshot.action_hints)
