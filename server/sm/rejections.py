"""State-machine rejection types with centralized user-facing reasons.

Business validation errors use concrete subclasses so callers can preserve
the rejection type while still reading a single ``reason`` string.
"""

from __future__ import annotations

from typing import ClassVar, assert_never

from server.actions import GameActionKind
from server.result import Rejected

from .card_model import Rank
from .card_model import Suit
from .types import (
    DealBidPhase,
    EffectiveSuit,
    GamePhase,
    PlayShapeInfo,
    PublicGamePhase,
    RoundPhase,
)


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


def _game_phase_text(phase: GamePhase) -> str:
    match phase:
        case "IDLE":
            return "未开始"
        case "IN_ROUND":
            return "一轮进行中"
        case "GAME_OVER":
            return "游戏结束"
    assert_never(phase)


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


def _public_game_phase_text(phase: PublicGamePhase) -> str:
    if phase == "GAME_OVER":
        return "游戏结束"
    return _round_phase_text(phase)


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


class CardNotInHandRejected(Rejected):
    def __init__(self, card_id: str, *, player_index: int | None = None, current: bool = False) -> None:
        if player_index is not None and current:
            super().__init__(f"牌 {card_id} 不在玩家 {player_index} 的当前手牌里。")
        elif current:
            super().__init__(f"牌 {card_id} 不在你的当前手牌里。")
        elif player_index is not None:
            super().__init__(f"牌 {card_id} 不在玩家 {player_index} 的手牌中")
        else:
            super().__init__(f"牌 {card_id} 不在手牌中")


class CardsNotInHandRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("出的牌不在手牌中")


class DuplicateCardRejected(Rejected):
    def __init__(self, card_id: str) -> None:
        super().__init__(f"牌 {card_id} 重复出现")


class DuplicateBidCardsRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("牌张重复，不能使用同一张牌两次")


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


class PlayerActionNotAllowedInGamePhaseRejected(Rejected):
    def __init__(self, action: GameActionKind, phase: PublicGamePhase) -> None:
        super().__init__(
            f"不能在{_public_game_phase_text(phase)}阶段执行{_game_action_text(action)}。"
        )


class CannotStartGameRejected(Rejected):
    def __init__(self, current_phase: GamePhase) -> None:
        super().__init__(f"不能在{_game_phase_text(current_phase)}阶段开始游戏，需要未开始阶段。")


class CannotProcessRoundResultRejected(Rejected):
    def __init__(self, current_phase: GamePhase) -> None:
        super().__init__(f"不能在{_game_phase_text(current_phase)}阶段处理回合结果，需要一轮进行中。")


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


class EmptyBidRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("抢主必须至少亮出一张牌。")


class MissingBidSuitRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("主牌抢主必须指定花色")


class BidCardWrongRankRejected(Rejected):
    def __init__(self, card_id: str, trump_rank: Rank) -> None:
        super().__init__(f"牌 {card_id} 不是主牌等级 {trump_rank.value}")


class BidCardSuitMismatchRejected(Rejected):
    def __init__(self, card_suit: Suit, declared_suit: Suit) -> None:
        super().__init__(f"牌花色 {effective_suit_name(card_suit)} 与声明花色 {effective_suit_name(declared_suit)} 不一致")


class BidCountRejected(Rejected):
    def __init__(self, count: int) -> None:
        super().__init__(f"抢主数量必须为1或2，实际 {count}")


class BidCardsCountMismatchRejected(Rejected):
    def __init__(self, actual_card_count: int, declared_count: int) -> None:
        super().__init__(f"牌张数量 {actual_card_count} 与声明数量 {declared_count} 不一致")


class JokerBidMustBePairRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("王抢主必须出对子")


class JokerBidCountRejected(Rejected):
    def __init__(self, actual_count: int) -> None:
        super().__init__(f"王抢主必须出2张，实际 {actual_count} 张")


class NotJokerRejected(Rejected):
    def __init__(self, card_id: str) -> None:
        super().__init__(f"牌 {card_id} 不是王")


class MixedJokerPairRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("两种王不能配对")


class JokerBidSuitRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("王抢主不能指定花色")


class ZeroBidValueRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("抢主无效：牌张价值为零")


class BidPriorityTooLowRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("抢主优先级不足")


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


class EmptyPlayRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("必须至少出一张牌")


class TrickResolvedRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("该轮已结束")


class MixedLeadSuitRejected(Rejected):
    def __init__(self, suits: set[EffectiveSuit]) -> None:
        suit_names = "、".join(sorted(effective_suit_name(suit) for suit in suits))
        super().__init__(f"首出必须是同一门牌：你选择的牌里混合了{suit_names}。")


class TooManyPlayHintsRejected(Rejected):
    reason_text: ClassVar[str] = "too many play hints"

    def __init__(self) -> None:
        super().__init__(self.reason_text)


class EmptyLeadRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("首出牌为空，无法判断跟牌规则。")


class WrongFollowCountRejected(Rejected):
    def __init__(self, lead_count: int) -> None:
        super().__init__(f"跟牌张数错误：首出 {lead_count} 张，你也必须出 {lead_count} 张。")


class EmptyFollowRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("跟牌必须至少出一张牌。")


class MustFollowTrumpRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("必须跟主牌：首出是主牌，你手里还有主牌。")


class MustFollowLeadSuitRejected(Rejected):
    def __init__(self, lead_suit: EffectiveSuit) -> None:
        suit_name = effective_suit_name(lead_suit)
        super().__init__(f"必须跟首出花色：首出是{suit_name}，你手里还有{suit_name}。")


class MustExhaustTrumpRejected(Rejected):
    def __init__(self, count: int) -> None:
        super().__init__(f"必须先把主牌跟完：首出是主牌，你手里只有 {count} 张主牌，必须全部出出来。")


class MustExhaustLeadSuitRejected(Rejected):
    def __init__(self, lead_suit: EffectiveSuit, count: int) -> None:
        suit_name = effective_suit_name(lead_suit)
        super().__init__(f"必须先把首出花色跟完：首出是{suit_name}，你手里只有 {count} 张{suit_name}，必须全部出出来。")


class MustFollowPairsRejected(Rejected):
    def __init__(
        self,
        *,
        lead_pair_count: int,
        lead_suit: EffectiveSuit,
        hand_pair_count: int,
        pair_floor: int,
    ) -> None:
        suit_name = effective_suit_name(lead_suit)
        super().__init__(
            f"必须跟对子：首出包含 {lead_pair_count} 个{suit_name}对子，"
            f"你手里有 {hand_pair_count} 个{suit_name}对子，至少要跟 {pair_floor} 个对子。"
        )


class MustFollowHigherPatternRejected(Rejected):
    def __init__(self, lead_shape: PlayShapeInfo) -> None:
        super().__init__(
            f"必须优先跟更大的牌型：首出是{_format_play_shape(lead_shape)}，"
            "你手里有拖拉机或对子时不能先拆成更小牌型。"
        )


class IllegalFollowShapeRejected(Rejected):
    def __init__(self, lead_shape: PlayShapeInfo) -> None:
        super().__init__(
            f"跟牌牌型不符合首出牌型：首出是{_format_play_shape(lead_shape)}，"
            "你手里有对应牌型时必须优先跟。"
        )


def _format_play_shape(shape: PlayShapeInfo) -> str:
    match shape.kind:
        case "empty":
            return "空牌"
        case "single":
            assert shape.suit is not None
            return f"{effective_suit_name(shape.suit)}单张"
        case "pair":
            assert shape.suit is not None
            return f"{effective_suit_name(shape.suit)}对子"
        case "tractor":
            assert shape.suit is not None
            assert shape.pair_count is not None
            return f"{effective_suit_name(shape.suit)}{shape.pair_count}连对"
        case "cards":
            assert shape.suit is not None
            return f"{shape.card_count}张{effective_suit_name(shape.suit)}牌"
    assert_never(shape.kind)


def effective_suit_name(suit: EffectiveSuit) -> str:
    if suit == "trump":
        return "主牌"
    if suit == Suit.HEARTS:
        return "红桃"
    if suit == Suit.SPADES:
        return "黑桃"
    if suit == Suit.CLUBS:
        return "梅花"
    if suit == Suit.DIAMONDS:
        return "方片"
    if suit == Suit.JOKER:
        return "王牌"
    assert_never(suit)
