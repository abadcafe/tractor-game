"""Tool schemas and validation for AIPlayer."""

from __future__ import annotations

from server.player.ai.client import AIToolCall, AIToolSpec, JSONValue
from server.player.ai.rejections import format_rejected
from server.protocol import AwaitingAction, PlayerMessage, StateSnapshot
from server.result import Ok, Rejected

_AWAITING_TEXT: dict[AwaitingAction, str] = {
    "bid": "抢主或不抢",
    "stir": "反主或不反",
    "discard": "埋底牌",
    "play": "出牌",
    "next_round": "确认进入下一轮",
}


def allowed_tool_specs(snapshot: StateSnapshot) -> list[AIToolSpec]:
    awaiting = snapshot.awaiting_action
    if awaiting == "next_round":
        return []
    if awaiting == "bid":
        return []
    if awaiting == "stir":
        allowed_ids = _hint_card_ids(snapshot) or _hand_card_ids(
            snapshot
        )
        return [
            _card_tool(
                "stir_trump",
                "炒地皮阶段反主：card_ids 为空或省略表示不反；"
                "要反主时，card_ids 必须完整等于 action_hints "
                "中的一组。",
                allowed_ids,
                require_card_ids=False,
                require_reason=False,
            ),
        ]
    if awaiting == "discard":
        return [
            _card_tool(
                "discard_bottom",
                "埋底：选择指定数量的牌放入底牌。",
                _hand_card_ids(snapshot),
            )
        ]
    if awaiting == "play":
        allowed_ids = _hint_card_ids(snapshot) or _hand_card_ids(
            snapshot
        )
        return [
            _card_tool(
                "play_cards",
                "出牌。若 action_hints 非空，card_ids 必须完整等于"
                "其中一组，不能只取一部分。",
                allowed_ids,
            )
        ]
    return []


def tool_call_to_message(
    seq: int,
    snapshot: StateSnapshot,
    call: AIToolCall,
) -> Ok[PlayerMessage] | Rejected:
    awaiting = snapshot.awaiting_action
    known_card_tools = {"stir_trump", "discard_bottom", "play_cards"}
    if call.name not in known_card_tools:
        return format_rejected(
            f"未知工具 {call.name}：当前 tools 列表里没有这个工具。",
            "只调用当前 tools 列表中存在的工具，"
            "不要使用上一轮状态里的工具名。",
        )

    card_ids_result = _card_ids(
        call, missing_as_empty=call.name == "stir_trump"
    )
    if isinstance(card_ids_result, Rejected):
        return card_ids_result
    card_ids = card_ids_result.value
    hand_result = _validate_card_ids_in_hand(snapshot, card_ids)
    if isinstance(hand_result, Rejected):
        return hand_result

    if call.name == "stir_trump":
        if awaiting != "stir":
            return _tool_not_allowed(call.name, awaiting, "stir")
        if not card_ids:
            return Ok(
                PlayerMessage(
                    seq=seq, raw={"type": "stir", "pass": True}
                )
            )
        if snapshot.action_hints and not _matches_hint(
            card_ids, snapshot
        ):
            return _action_hint_rejected()
        return Ok(
            PlayerMessage(
                seq=seq, raw={"type": "stir", "cards": card_ids}
            )
        )

    if call.name == "discard_bottom":
        if awaiting != "discard":
            return _tool_not_allowed(call.name, awaiting, "discard")
        expected = (
            snapshot.stirring_state.exchange_count
            if snapshot.stirring_state is not None
            else None
        )
        if expected is None:
            return format_rejected(
                "当前状态没有 stirring_state.exchange_count，"
                "不能判断埋底数量。",
                "重新读取当前状态；只有当前需要你埋底，且 "
                "exchange_count 存在时，才能调用 discard_bottom。",
            )
        if len(card_ids) != expected:
            return format_rejected(
                f"埋底数量错误：需要选择 {expected} 张牌，"
                f"你选择了 {len(card_ids)} 张。",
                f"调用 discard_bottom 时传入正好 {expected} 个 "
                "card_ids。",
            )
        return Ok(
            PlayerMessage(
                seq=seq, raw={"type": "discard", "cards": card_ids}
            )
        )

    if call.name == "play_cards":
        if awaiting != "play":
            return _tool_not_allowed(call.name, awaiting, "play")
        if not card_ids:
            return format_rejected(
                "出牌至少要选择一张牌。",
                "从当前手牌中选择要出的 card_ids；如果 "
                "action_hints 非空，完整复制其中一组。",
            )
        if snapshot.action_hints and not _matches_hint(
            card_ids, snapshot
        ):
            return _action_hint_rejected()
        return Ok(
            PlayerMessage(
                seq=seq, raw={"type": "play", "cards": card_ids}
            )
        )

    raise AssertionError(f"unreachable known tool: {call.name}")


def _card_tool(
    name: str,
    description: str,
    allowed_card_ids: list[str],
    *,
    require_card_ids: bool = True,
    require_reason: bool = True,
) -> AIToolSpec:
    item_schema: dict[str, JSONValue] = {"type": "string"}
    if allowed_card_ids:
        enum_values: list[JSONValue] = [
            card_id for card_id in allowed_card_ids
        ]
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
                    "description": (
                        "从当前可用 card id 枚举中逐字复制。"
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "简短说明，便于 debug。",
                },
            },
            "required": _required_tool_fields(
                require_card_ids=require_card_ids,
                require_reason=require_reason,
            ),
            "additionalProperties": False,
        },
    )


def _required_tool_fields(
    *, require_card_ids: bool, require_reason: bool
) -> list[JSONValue]:
    result: list[JSONValue] = []
    if require_card_ids:
        result.append("card_ids")
    if require_reason:
        result.append("reason")
    return result


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


def _card_ids(
    call: AIToolCall, *, missing_as_empty: bool
) -> Ok[list[str]] | Rejected:
    raw = call.arguments.get("card_ids")
    if raw is None and missing_as_empty:
        return Ok([])
    if not isinstance(raw, list):
        return format_rejected(
            f"{call.name} 必须提供 card_ids 数组。",
            "按工具参数格式传入 card_ids，且 card_ids "
            "必须是字符串数组。",
        )
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            return format_rejected(
                "card_ids 只能包含字符串 card id。",
                "从你的手牌或 action_hints 中逐字复制 card id 字符串。",
            )
        if item in seen:
            return format_rejected(
                f"card_ids 里重复选择了 {item}。",
                "每张牌 id 最多出现一次；如果你有两张同牌面，"
                "也必须使用两个不同 card id。",
            )
        seen.add(item)
        result.append(item)
    return Ok(result)


def _validate_card_ids_in_hand(
    snapshot: StateSnapshot, card_ids: list[str]
) -> Ok[None] | Rejected:
    hand_ids = {card.id for card in snapshot.player_hand}
    for card_id in card_ids:
        if card_id not in hand_ids:
            return format_rejected(
                f"牌 {card_id} 不在你的当前手牌里。",
                "card_ids 只能从你的手牌或 action_hints 中逐字复制；"
                "不要根据牌面自行编造"
                "id。",
            )
    return Ok(None)


def _matches_hint(card_ids: list[str], snapshot: StateSnapshot) -> bool:
    key = tuple(sorted(card_ids))
    return any(
        tuple(sorted(card.id for card in hint)) == key
        for hint in snapshot.action_hints
    )


def _tool_not_allowed(
    tool_name: str,
    awaiting_action: AwaitingAction | None,
    expected_awaiting_action: AwaitingAction,
) -> Rejected:
    actual = _awaiting_text(awaiting_action)
    expected = _AWAITING_TEXT[expected_awaiting_action]
    return format_rejected(
        f"当前不能调用 {tool_name}：现在需要你{actual}，"
        f"这个工具只允许在需要你{expected}时调用。",
        "只调用当前 tools 列表中的工具；"
        "不要使用上一轮状态里的工具调用。",
    )


def _action_hint_rejected() -> Rejected:
    return format_rejected(
        "card_ids 必须完整等于 action_hints 里的某一组："
        "不能只选其中一部分，也不能混合多组。",
        "从 legal_action_hint_groups 中复制一整组 card_ids。",
    )


def _awaiting_text(awaiting_action: AwaitingAction | None) -> str:
    if awaiting_action is None:
        return "不行动"
    return _AWAITING_TEXT[awaiting_action]
