"""Prompt context construction for AIPlayer."""

from __future__ import annotations

from server.player.ai.client import AIDecisionPrompt
from server.player.ai.formatting import card_points, card_text
from server.player.ai.memory import AIMemory
from server.player.ai.rules import RuleBook
from server.protocol import (
    AwaitingAction,
    RoundPhase,
    StateSnapshot,
    TrickSnapshot,
)
from server.rules.cards import Card, Rank, Suit

_PHASE_TEXT: dict[RoundPhase, str] = {
    "DEAL_BID": "抓牌抢主阶段",
    "STIRRING": "炒地皮阶段",
    "PLAYING": "出牌阶段",
    "SCORING": "结算阶段",
    "WAITING": "等待下一轮阶段",
}

_AWAITING_TEXT: dict[AwaitingAction, str] = {
    "bid": "抢主或不抢",
    "stir": "反主或不反",
    "discard": "埋底牌",
    "play": "出牌",
    "next_round": "确认进入下一轮",
}

_SUIT_TEXT: dict[Suit, str] = {
    Suit.HEARTS: "红桃",
    Suit.SPADES: "黑桃",
    Suit.DIAMONDS: "方片",
    Suit.CLUBS: "梅花",
    Suit.JOKER: "王",
}

_RANK_TEXT: dict[Rank, str] = {
    Rank.TWO: "2",
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "10",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
    Rank.SMALL_JOKER: "小王",
    Rank.BIG_JOKER: "大王",
}


def build_decision_prompt(
    *,
    player_index: int,
    snapshot: StateSnapshot,
    memory: AIMemory,
    rules: RuleBook,
) -> AIDecisionPrompt:
    """Build the one-shot prompt for a single AI action decision."""
    system = "\n\n".join(
        [
            "你是升级/拖拉机游戏里的 AI 玩家。",
            "你只能根据当前玩家可见的信息决策，不能假设其他玩家手牌。",
            "花色对应关系：♠/spades=黑桃，♥/hearts=红桃，"
            "♣/clubs=梅花，♦/diamonds=方片，joker=王。",
            "你必须且只能调用一个当前允许的动作工具（tool）。",
            "tool 参数里的 card_ids 只能从“你的手牌”或 "
            "action_hints 中逐字复制。",
            "如果当前 action_hints 非空，且你提供了 card_ids，"
            "card_ids 必须完整等于其中一个 hint 组，"
            "不能只取 hint 组的一部分。",
            "当前墩、历史记忆、叫牌记录里的牌只用于判断局势，"
            "不能作为 card_ids。",
            "不要输出自然语言动作；reason 字段只用于日志。",
            rules.select(snapshot),
        ]
    )

    user_parts = [
        _state_summary(player_index, snapshot),
        _hand_summary(snapshot.player_hand),
        _trick_summary(snapshot.trick),
        memory.summary(),
        _hints_summary(snapshot),
    ]
    return AIDecisionPrompt(
        system=system,
        user="\n\n".join(part for part in user_parts if part),
    )


def _state_summary(player_index: int, snapshot: StateSnapshot) -> str:
    declarer_player = _optional_player_text(snapshot.declarer_player)
    declarer_team = _optional_team_text(snapshot.declarer_team)
    return "\n".join(
        [
            "当前状态:",
            f"- 你是：{_player_text(player_index)}",
            f"- 阶段：{_phase_text(snapshot.phase)}",
            f"- 当前需要你：{_awaiting_text(snapshot.awaiting_action)}",
            f"- 主级牌：{_rank_text(snapshot.trump_rank)}",
            f"- 主花色：{_optional_suit_text(snapshot.trump_suit)}",
            f"- 庄家：{declarer_player}",
            f"- 庄家队伍：{declarer_team}",
            f"- 防守方得分：{snapshot.defender_points}",
            f"- 各玩家剩余手牌数：{_hand_counts_text(snapshot)}",
        ]
    )


def _hand_summary(hand: list[Card]) -> str:
    lines = ["你的手牌:"]
    if not hand:
        lines.append("- 无")
    for card in hand:
        lines.append(
            f"- {card.id}: {card_text(card)}，分值={card_points(card)}"
        )
    return "\n".join(lines)


def _trick_summary(trick: TrickSnapshot | None) -> str:
    if trick is None:
        return "当前墩：无"
    lines = [
        "当前墩:",
        f"- 首出玩家：{_player_text(trick.lead_player)}",
        f"- 当前出牌玩家：{_player_text(trick.current_player)}",
    ]
    for slot in trick.slots:
        cards = ", ".join(card_text(card) for card in slot.cards)
        played_cards = cards if cards else "未出牌"
        lines.append(f"- {_player_text(slot.player)}：{played_cards}")
    return "\n".join(lines)


def _hints_summary(snapshot: StateSnapshot) -> str:
    if not snapshot.action_hints:
        return "可选提示（action_hints）：无"
    lines = ["可选提示（action_hints）："]
    for index, hint in enumerate(snapshot.action_hints):
        cards = ", ".join(
            f"{card.id}:{card_text(card)}" for card in hint
        )
        lines.append(f"- 提示 {index}: {cards}")
    return "\n".join(lines)


def _phase_text(phase: RoundPhase) -> str:
    return _PHASE_TEXT[phase]


def _awaiting_text(awaiting: AwaitingAction | None) -> str:
    if awaiting is None:
        return "不需要你行动"
    return _AWAITING_TEXT[awaiting]


def _rank_text(rank: Rank) -> str:
    return _RANK_TEXT[rank]


def _optional_suit_text(suit: Suit | None) -> str:
    if suit is None:
        return "无主"
    return _SUIT_TEXT[suit]


def _player_text(player: int) -> str:
    return f"玩家 {player}"


def _optional_player_text(player: int | None) -> str:
    if player is None:
        return "未确定"
    return _player_text(player)


def _optional_team_text(team: int | None) -> str:
    if team is None:
        return "未确定"
    return f"{team} 队"


def _hand_counts_text(snapshot: StateSnapshot) -> str:
    return "，".join(
        f"{_player_text(index)}={count} 张"
        for index, count in enumerate(snapshot.player_hand_counts)
    )
