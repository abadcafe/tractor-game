"""State-machine rejection types with centralized user-facing reasons.

Business validation errors use concrete subclasses so callers can preserve
the rejection type while still reading a single ``reason`` string.
"""

from __future__ import annotations

from typing import assert_never

from server.actions import GameActionKind
from server.result import Rejected

from .types import DealBidPhase, RoundPhase


def _game_action_text(action: GameActionKind) -> str:
    match action:
        case "bid":
            return "抢主"
        case "skip_bid":
            return "不抢"
        case "stir":
            return "反主"
        case "skip_stir":
            return "不反"
        case "discard":
            return "换底牌"
        case "play":
            return "出牌"
        case "next_round":
            return "下一轮"
    assert_never(action)

def _round_phase_text(phase: RoundPhase) -> str:
    match phase:
        case "DEAL_BID":
            return "抓牌抢主"
        case "STIRRING":
            return "炒地皮"
        case "PLAYING":
            return "出牌"
        case "SCORING":
            return "结算"
        case "WAITING":
            return "等待下一轮"
    assert_never(phase)

def _deal_bid_phase_text(phase: DealBidPhase) -> str:
    match phase:
        case "DEALING":
            return "发牌"
        case "COMPLETE":
            return "发牌完成"
        case "NO_BID":
            return "无人抢主"
    assert_never(phase)

def _round_phase_rejection(action_text: str, phase: RoundPhase) -> str:
    return f"不能在{_round_phase_text(phase)}阶段{action_text}。"

def _deal_bid_phase_rejection(action_text: str, phase: DealBidPhase) -> str:
    return f"不能在{_deal_bid_phase_text(phase)}阶段{action_text}。"

class MissingActionTypeRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("缺少动作类型：raw.type 必须是字符串。")

class UnknownActionTypeRejected(Rejected):
    def __init__(self, action_type: str) -> None:
        super().__init__(f"未知动作类型：{action_type}。")

class GameNotStartedRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("游戏尚未开始")

class MissingCardIdRejected(Rejected):
    def __init__(self, item: object) -> None:
        super().__init__(f"牌格式错误：对象缺少字符串 id 字段：{item}")

class InvalidCardFormatRejected(Rejected):
    def __init__(self, item: object) -> None:
        super().__init__(f"牌格式错误：cards 只能包含 card id 字符串或带 id 的对象：{item}")

class InvalidPlayerIndexRejected(Rejected):
    def __init__(self, player_index: int) -> None:
        super().__init__(f"玩家索引无效：{player_index}")

class WrongTurnRejected(Rejected):
    def __init__(self, current_player: int | None = None) -> None:
        if current_player is None:
            super().__init__("不是你的回合")
        else:
            super().__init__(f"不是你的回合，当前是玩家 {current_player} 的回合")

class WrongBidTurnRejected(Rejected):
    def __init__(self, current_bidder: int) -> None:
        super().__init__(f"不是你的抢主回合（当前抢主者：{current_bidder}）")

class DuplicateNextRoundConfirmationRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("你已经确认过了")

class PlayerActionNotAllowedInRoundPhaseRejected(Rejected):
    def __init__(self, action: GameActionKind, phase: RoundPhase) -> None:
        super().__init__(
            f"不能在{_round_phase_text(phase)}阶段执行{_game_action_text(action)}。"
        )

class CannotStartGameRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("不能开始游戏：游戏已经开始或已经结束。")

class CannotProcessRoundResultRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("不能处理回合结果：游戏尚未开始或已经结束。")

class DealCardNotAllowedInRoundPhaseRejected(Rejected):
    def __init__(self, phase: RoundPhase) -> None:
        super().__init__(_round_phase_rejection("发牌", phase))

class BidNotAllowedInRoundPhaseRejected(Rejected):
    def __init__(self, phase: RoundPhase) -> None:
        super().__init__(_round_phase_rejection("抢主", phase))

class FinalizeDealNotAllowedInRoundPhaseRejected(Rejected):
    def __init__(self, phase: RoundPhase) -> None:
        super().__init__(_round_phase_rejection("结束发牌", phase))

class SkipStirNotAllowedInRoundPhaseRejected(Rejected):
    def __init__(self, phase: RoundPhase) -> None:
        super().__init__(_round_phase_rejection("不反", phase))

class StirNotAllowedInRoundPhaseRejected(Rejected):
    def __init__(self, phase: RoundPhase) -> None:
        super().__init__(_round_phase_rejection("反主", phase))

class DiscardNotAllowedInRoundPhaseRejected(Rejected):
    def __init__(self, phase: RoundPhase) -> None:
        super().__init__(_round_phase_rejection("换底牌", phase))

class PlayNotAllowedInRoundPhaseRejected(Rejected):
    def __init__(self, phase: RoundPhase) -> None:
        super().__init__(_round_phase_rejection("出牌", phase))

class RoundMissingDealBidStateRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("回合状态异常：当前需要发牌抢主状态，但 deal_bid_state 为空。")

class RoundMissingStirringStateRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("回合状态异常：当前需要炒地皮状态，但 stirring_state 为空。")

class RoundMissingTrickStateRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("回合状态异常：当前需要出牌状态，但 trick_state 为空。")

class StirringMissingExchangeStateRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("炒地皮状态异常：当前需要换底牌状态，但 exchange_state 为空。")

class DealCardNotAllowedInDealBidPhaseRejected(Rejected):
    def __init__(self, phase: DealBidPhase) -> None:
        super().__init__(_deal_bid_phase_rejection("发牌", phase))

class BidNotAllowedInDealBidPhaseRejected(Rejected):
    def __init__(self, phase: DealBidPhase) -> None:
        super().__init__(_deal_bid_phase_rejection("抢主", phase))

class AllCardsDealtRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("所有牌已发完")

class DealNotCompleteRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("还有牌未发完，不能结束发牌")

class CannotPassStirWhileExchangingRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("正在换底牌，不能跳过反主")

class CannotStirNowRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("当前不能反主")

class CannotStirConsecutivelyRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("不能连续反主")

class StirMustBePairRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("反主必须出对子")

class JokerCannotPairWithNormalRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("王和普通牌不能配对")

class StirCardNotTrumpRankRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("牌不是主牌等级")

class PairSuitMismatchRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("对子必须同花色")

class StirPriorityTooLowRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("优先级不足，不能反主")

class NotStirExchangePhaseRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("当前不在换底牌阶段")

class NotStirringExchangerRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("只有炒主者可以换底牌")

class InvalidExchangeCountRejected(Rejected):
    def __init__(self, required_count: int, actual_count: int) -> None:
        super().__init__(f"埋牌数量错误：需要 {required_count} 张，实际 {actual_count} 张")

class TrickResolvedRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("该轮已结束")
