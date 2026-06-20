"""Prompt context construction for AIPlayer."""

from __future__ import annotations

from server.player.ai.client import AIDecisionPrompt
from server.player.ai.formatting import card_points, card_text
from server.player.ai.memory import AIMemory
from server.player.ai.rules import RuleBook
from server.protocol import StateSnapshot, TrickSnapshot
from server.rules.cards import Card


def build_decision_prompt(
    *,
    player_index: int,
    snapshot: StateSnapshot,
    memory: AIMemory,
    rules: RuleBook,
) -> AIDecisionPrompt:
    """Build the one-shot prompt for a single AI action decision."""
    system = "\n\n".join([
        "你是升级/拖拉机游戏里的 AI 玩家。",
        "你只能根据当前 player 可见的信息决策，不能假设其他玩家手牌。",
        "你必须且只能调用一个当前允许的动作 tool。",
        "tool 参数里的 card_ids 只能从“你的手牌”或 action_hints 中逐字复制。",
        "如果当前 action_hints 非空，card_ids 必须完整等于其中一个 hint 组，不能只取 hint 组的一部分。",
        "当前墩、历史记忆、叫牌记录里的牌只用于判断局势，不能作为 card_ids。",
        "不要输出自然语言动作；reason 字段只用于日志。",
        rules.select(snapshot),
    ])

    user_parts = [
        _state_summary(player_index, snapshot),
        _hand_summary(snapshot.player_hand),
        _trick_summary(snapshot.trick),
        memory.summary(),
        _hints_summary(snapshot),
    ]
    return AIDecisionPrompt(system=system, user="\n\n".join(part for part in user_parts if part))


def _state_summary(player_index: int, snapshot: StateSnapshot) -> str:
    return "\n".join([
        "当前状态:",
        f"- 你是 player {player_index}",
        f"- phase: {snapshot.phase}",
        f"- awaiting_action: {snapshot.awaiting_action}",
        f"- trump_rank: {snapshot.trump_rank}",
        f"- trump_suit: {snapshot.trump_suit if snapshot.trump_suit is not None else 'no_trump'}",
        f"- declarer_player: {snapshot.declarer_player}",
        f"- declarer_team: {snapshot.declarer_team}",
        f"- defender_points: {snapshot.defender_points}",
        f"- player_hand_counts: {snapshot.player_hand_counts}",
    ])


def _hand_summary(hand: list[Card]) -> str:
    lines = ["你的手牌:"]
    if not hand:
        lines.append("- empty")
    for card in hand:
        lines.append(f"- {card.id}: {card_text(card)}, points={card_points(card)}")
    return "\n".join(lines)


def _trick_summary(trick: TrickSnapshot | None) -> str:
    if trick is None:
        return "当前墩: none"
    lines = [
        "当前墩:",
        f"- lead_player: {trick.lead_player}",
        f"- current_player: {trick.current_player}",
    ]
    for slot in trick.slots:
        cards = ", ".join(card_text(card) for card in slot.cards)
        lines.append(f"- player {slot.player}: {cards if cards else 'not_played'}")
    return "\n".join(lines)


def _hints_summary(snapshot: StateSnapshot) -> str:
    if not snapshot.action_hints:
        return "action_hints: empty"
    lines = ["action_hints:"]
    for index, hint in enumerate(snapshot.action_hints):
        cards = ", ".join(f"{card.id}:{card_text(card)}" for card in hint)
        lines.append(f"- hint {index}: {cards}")
    return "\n".join(lines)
