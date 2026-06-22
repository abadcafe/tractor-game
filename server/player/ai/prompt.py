"""Prompt construction for AIPlayer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from server.player.ai.client import AIDecisionPrompt
from server.player.ai.formatting import card_points, card_text
from server.player.ai.memory import AIMemory
from server.protocol import (
    AwaitingAction,
    RoundPhase,
    StateSnapshot,
    TrickSnapshot,
)
from server.rules.cards import Card, Rank, Suit
from server.rules.ordering import is_trump_card, sort_by_display_order

COMMON_RULES: Final[str] = """
- 你是升级/拖拉机游戏里的玩家，你要以游戏**最终**胜利为目标，根据当前状
  态，只调用一个当前允 许的动作工具（tool）
- tool 参数里的 card_ids 只能从**你的手牌**逐字复制
- 如果当前 legal_action_groups 数组非空，那么它枚举了你手牌中的所有合法
  出牌选项；此时你如果要提供 card_ids ，那它必须完整等于其中一个元素
- 当前墩、历史记忆、叫牌记录里的牌只用于判断局势，不能作为 card_ids。
- 不要输出自然语言动作；reason 字段只用于日志。
- player 0 - 3 一共4个人玩：0 和 2 是队友, 1 和 3 是队友，要认清自己队友
- 花色有5种：
  ♠=黑桃（spades），♥=红桃（hearts），♣=梅花（clubs），♦=方片（diamonds），
  以及大小王（joker）
  除去级牌后，每个花色一共有24张牌，因此若某花色多于6张就算比较多了
- 亮主后，大小王，主副级牌，其他主牌（空主时不含）
  三者花色一致，都是主牌。
""".strip()

BID_RULES: Final[str] = """
- 若打算抢主，则只能从 legal_action_groups 里选抢主的牌型。选择大又多的
  那门花色抢
- 如果庄家未定，抢主胜者会成为庄家，能拿到8张底牌与手牌交换优化手牌
  - 越早抢主越有更大概率当上庄家，但也越有可能错抢一门少又小的花色当主
- 如果某一门花色多又大，就值得抢该门花色的主
- 抢主是同时进行的，你如果不抢，那别人同一时刻可能先抢了
- 如果一门花色有一对主，可以考虑不抢主而是在别人埋牌以后再反主，因为别人
  埋牌通常是尽量埋空一两门花色，这样你拿到的底牌花色更集中
- 你一共有4次抢主机会，每次间隔6张牌
- 抢主不是必须的，觉得手牌不合适就 pass
""".strip()

STIR_RULES: Final[str] = """
- 反主可以把 8 张底牌拿到手里进行交换，从而优化手牌
- 若打算反主，则只能从 legal_action_groups 里挑反主的牌型，
  一般选择大又多的那门花色反
- 反主不是必须的，觉得可反的那门花色手牌太少了也可以 pass
""".strip()

DISCARD_RULES: Final[str] = """
- 埋底牌必须正好埋 8 张不能多也不能少。
- 要加强优势花色。尽量多保留主牌和多又大的那门副牌，埋空一门少又小的副牌
- 拖拉机和大的对子（10以上的）都非常稀有，被对手压住的概率低，尽量保留
- 最后一墩牌的赢方会拿到并根据牌型放大底牌分（抠底），因此要根据手牌思考
  己方是否有把握在最后一墩拿到底牌
- 对手如果没拿过底牌，那他们大概率是每种花色都有的，
  因此庄家埋牌时可以保留A和对手大概率压不住的其他牌型
  例如，A，对K等。但单张K就很危险，对手有单张A从而压住单张K的可能性很大
""".strip()

PLAY_RULES: Final[str] = """
- 根据自己的手牌和整体出牌记录，推测队友和对手手里分别有什么牌，才能配合
  队友并压制对手
- 每次只能出一门花色的牌，不同花色不可混出
- 谨慎出分牌，要大概率该分能被己方得到才出分牌
- 如果对手大概率能赢，尽量垫低价值牌

""".strip()

PLAY_LEAD_RULES: Final[str] = """
- 如果你判断某几张牌的组合在该门花色中一定是最大的，可以一起出，这叫甩牌
  - 如果甩的牌中某种牌型在其他玩家手中有更大的，甩牌会失败
  - 失败会暴露所有尝试甩出的牌，并被迫打出比别人小的那组牌型
- 组合牌型的牌（对子，拖拉机，甩牌等）越长，被对手压制的概率更小
  - 但一旦被压制，对手就可以乘机上很多分
- 如果某种花色你有大概率最大的牌，并通过出牌记录和手牌分析所有对手很可能
  也有该花色的牌，那你可以把最大的牌一起甩出来，对手就必须跟你同花色的牌，
  队友也因此有机会出分牌
- 副牌没有好牌时，可以选择出很小的主牌，谓之调主，低风险地转移牌权给队友
""".strip()

PLAY_FOLLOW_RULES: Final[str] = """
- 同门花色有一样牌型时必须优先跟该牌型，没有一样牌型时必须优先跟同门花色
  - 如果都不满足，则不讲究牌型和花色随意跟牌（优先小牌），谨慎跟主牌分牌
  - 跟牌数量必须与首出牌数量一致
- 如果队友的牌明显大于对手已出和估计要出的牌，应当多出分牌
- 如果队友出的副牌花色你全部没有，可以用主牌杀，从而转移牌权给自己
  - 如果拿到牌权也不知道出什么牌，那也不一定非要杀
""".strip()

SCORING_RULES: Final[str] = """
- 5、10、K 是分牌，分别代表5，10，10分，出牌时要关注 5、10、K 的归属
- 庄家的唯一目的是阻止对手拿分，另一队的唯一目的是尽量多拿分
- 最后一墩的赢队将额外拿到底牌的分数x一定的放大倍数，因此输赢非常重要
- 时刻关注场上得分和出过的分牌，满80分庄家就输了
- 每40分是一级，40, 80, 120等40倍数分数都很重要，尽量阻止对手再过一级
""".strip()

DEFAULT_RULE_SECTIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "common": COMMON_RULES,
        "bid": BID_RULES,
        "stir": STIR_RULES,
        "discard": DISCARD_RULES,
        "play": PLAY_RULES,
        "play_lead": PLAY_LEAD_RULES,
        "play_follow": PLAY_FOLLOW_RULES,
        "scoring": SCORING_RULES,
    }
)

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

_SIDE_SUIT_ORDER: Final[tuple[Suit, ...]] = (
    Suit.SPADES,
    Suit.HEARTS,
    Suit.CLUBS,
    Suit.DIAMONDS,
)


@dataclass(frozen=True, slots=True)
class RuleBook:
    sections: dict[str, str]

    @classmethod
    def from_default(cls) -> "RuleBook":
        return cls(dict(DEFAULT_RULE_SECTIONS))

    def select(self, snapshot: StateSnapshot) -> str:
        keys = ["common"]
        awaiting = snapshot.awaiting_action
        if awaiting == "bid":
            keys.append("bid")
        elif awaiting == "stir":
            keys.append("stir")
        elif awaiting == "discard":
            keys.extend(["discard", "scoring"])
        elif awaiting == "play":
            keys.append("play")
            keys.append(
                "play_lead" if _is_leading(snapshot) else "play_follow"
            )
            keys.append("scoring")
        selected: list[str] = []
        for key in keys:
            section = self.sections.get(key)
            if section:
                selected.append(f"规则: {key}\n{section}")
        return "\n\n".join(selected)


def build_decision_prompt(
    *,
    player_index: int,
    snapshot: StateSnapshot,
    memory: AIMemory,
    rules: RuleBook,
) -> AIDecisionPrompt:
    """Build the one-shot prompt for a single AI action decision."""
    system = rules.select(snapshot)

    user_parts = [
        _state_summary(player_index, snapshot),
        _hand_summary(
            snapshot.player_hand,
            snapshot.trump_suit,
            snapshot.trump_rank,
        ),
        _trick_summary(snapshot.trick),
        memory.summary(),
        _action_constraints_summary(snapshot),
    ]
    return AIDecisionPrompt(
        system=system,
        user="\n\n".join(part for part in user_parts if part),
    )


def _state_summary(player_index: int, snapshot: StateSnapshot) -> str:
    declarer_player = _optional_player_text(snapshot.declarer_player)
    declarer_team = _optional_team_text(snapshot.declarer_team)
    return f"""
当前状态:
- 你是：{_player_text(player_index)}
- 队友：{_player_text(_teammate_index(player_index))}
- 阶段：{_phase_text(snapshot.phase)}
- 当前需要你：{_awaiting_text(snapshot.awaiting_action)}
- 主级牌：{_rank_text(snapshot.trump_rank)}
- 主花色：{_optional_suit_text(snapshot.trump_suit)}
- 庄家：{declarer_player}
- 庄家队伍：{declarer_team}
- 防守方得分：{snapshot.defender_points}
- 各玩家剩余手牌数：{_hand_counts_text(snapshot)}
""".strip()


def _hand_summary(
    hand: list[Card], trump_suit: Suit | None, trump_rank: Rank
) -> str:
    lines = ["你的手牌:"]
    for label, cards in _hand_groups(hand, trump_suit, trump_rank):
        lines.append(f"- {label}:")
        if not cards:
            lines.append("  - 无")
            continue
        for card in cards:
            lines.append(
                f"  - {card.id}: {card_text(card)}，"
                f"分值={card_points(card)}"
            )
    return "\n".join(lines)


def _hand_groups(
    hand: list[Card], trump_suit: Suit | None, trump_rank: Rank
) -> list[tuple[str, list[Card]]]:
    sorted_hand = sort_by_display_order(hand, trump_suit, trump_rank)
    trump_cards = [
        card
        for card in sorted_hand
        if is_trump_card(card, trump_suit, trump_rank)
    ]
    groups: list[tuple[str, list[Card]]] = [("主牌", trump_cards)]
    for suit in _SIDE_SUIT_ORDER:
        if trump_suit is not None and suit == trump_suit:
            continue
        suit_cards = [
            card
            for card in sorted_hand
            if card.suit == suit
            and not is_trump_card(card, trump_suit, trump_rank)
        ]
        groups.append((f"{_SUIT_TEXT[suit]}副牌", suit_cards))
    return groups


def _trick_summary(trick: TrickSnapshot | None) -> str:
    if trick is None:
        return "当前墩：无"
    header = f"""
当前墩:
- 首出玩家：{_player_text(trick.lead_player)}
- 当前出牌玩家：{_player_text(trick.current_player)}
""".strip()
    plays: list[str] = []
    for slot in trick.slots:
        cards = ", ".join(card_text(card) for card in slot.cards)
        played_cards = cards if cards else "未出牌"
        plays.append(f"- {_player_text(slot.player)}：{played_cards}")
    return f"{header}\n" + "\n".join(plays)


def _action_constraints_summary(snapshot: StateSnapshot) -> str:
    if not snapshot.action_hints:
        return "legal_action_groups：无"
    lines = ["legal_action_groups："]
    for index, hint in enumerate(snapshot.action_hints):
        cards = ", ".join(
            f"{card.id}:{card_text(card)}" for card in hint
        )
        lines.append(f"- 约束 {index}: {cards}")
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


def _teammate_index(player: int) -> int:
    return (player + 2) % 4


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


def _is_leading(snapshot: StateSnapshot) -> bool:
    if snapshot.trick is None:
        return True
    lead_slot = snapshot.trick.slots[snapshot.trick.lead_player]
    return len(lead_slot.cards) == 0
