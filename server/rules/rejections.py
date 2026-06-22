"""Rule-level rejection types and user-facing reasons."""

from __future__ import annotations

from typing import ClassVar, assert_never

from server.result import Rejected

from .cards import Rank, Suit
from .types import EffectiveSuit, PlayShapeInfo


class CardNotInHandRejected(Rejected):
    def __init__(
        self,
        card_id: str,
        *,
        player_index: int | None = None,
        current: bool = False,
    ) -> None:
        if player_index is not None and current:
            super().__init__(
                f"牌 {card_id} 不在玩家 {player_index} 的当前手牌里。"
            )
        elif current:
            super().__init__(f"牌 {card_id} 不在你的当前手牌里。")
        elif player_index is not None:
            super().__init__(
                f"牌 {card_id} 不在玩家 {player_index} 的手牌中"
            )
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


class EmptyBidRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("抢主必须至少亮出一张牌。")


class MissingBidSuitRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("主牌抢主必须指定花色")


class BidCardWrongRankRejected(Rejected):
    def __init__(self, card_id: str, trump_rank: Rank) -> None:
        super().__init__(
            f"牌 {card_id} 不是主牌等级 {trump_rank.value}"
        )


class BidCardSuitMismatchRejected(Rejected):
    def __init__(self, card_suit: Suit, declared_suit: Suit) -> None:
        super().__init__(
            f"牌花色 {effective_suit_name(card_suit)} 与声明花色"
            f"{effective_suit_name(declared_suit)} 不一致"
        )


class BidCountRejected(Rejected):
    def __init__(self, count: int) -> None:
        super().__init__(f"抢主数量必须为1或2，实际 {count}")


class BidCardsCountMismatchRejected(Rejected):
    def __init__(
        self, actual_card_count: int, declared_count: int
    ) -> None:
        super().__init__(
            f"牌张数量 {actual_card_count} 与声明数量 "
            f"{declared_count} 不一致"
        )


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


class CurrentBidWinnerCannotRebidRejected(Rejected):
    def __init__(self) -> None:
        super().__init__(
            "当前抢主胜者不能再次抢自己的主；被别人抢走后才能再抢回来。"
        )


class EmptyPlayRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("必须至少出一张牌")


class MixedLeadSuitRejected(Rejected):
    def __init__(self, suits: set[EffectiveSuit]) -> None:
        suit_names = "、".join(
            sorted(effective_suit_name(suit) for suit in suits)
        )
        super().__init__(
            f"首出必须是同一门牌：你选择的牌里混合了{suit_names}。"
        )


class TooManyPlayHintsRejected(Rejected):
    reason_text: ClassVar[str] = "too many play hints"

    def __init__(self) -> None:
        super().__init__(self.reason_text)


class EmptyLeadRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("首出牌为空，无法判断跟牌规则。")


class WrongFollowCountRejected(Rejected):
    def __init__(self, lead_count: int) -> None:
        super().__init__(
            f"跟牌张数错误：首出 {lead_count} 张，"
            f"你也必须出 {lead_count} 张。"
        )


class EmptyFollowRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("跟牌必须至少出一张牌。")


class MustFollowTrumpRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("必须跟主牌：首出是主牌，你手里还有主牌。")


class MustFollowLeadSuitRejected(Rejected):
    def __init__(self, lead_suit: EffectiveSuit) -> None:
        suit_name = effective_suit_name(lead_suit)
        super().__init__(
            f"必须跟首出花色：首出是{suit_name}，你手里还有{suit_name}。"
        )


class MustExhaustTrumpRejected(Rejected):
    def __init__(self, count: int) -> None:
        super().__init__(
            f"必须先把主牌跟完：首出是主牌，"
            f"你手里只有 {count} 张主牌，必须全部出出来。"
        )


class MustExhaustLeadSuitRejected(Rejected):
    def __init__(self, lead_suit: EffectiveSuit, count: int) -> None:
        suit_name = effective_suit_name(lead_suit)
        super().__init__(
            f"必须先把首出花色跟完：首出是{suit_name}，"
            f"你手里只有 {count}"
            f"张{suit_name}，必须全部出出来。"
        )


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
            f"必须跟对子：首出包含 {lead_pair_count} 个"
            f"{suit_name}对子，"
            f"你手里有 {hand_pair_count} 个{suit_name}对子，"
            f"至少要跟 {pair_floor}"
            f"个对子。"
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
            suit_name = effective_suit_name(shape.suit)
            return f"{suit_name}{shape.pair_count}连对"
        case "cards":
            assert shape.suit is not None
            suit_name = effective_suit_name(shape.suit)
            return f"{shape.card_count}张{suit_name}牌"
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
