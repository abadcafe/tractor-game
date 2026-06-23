"""Lead and follow play rule rejections."""

from __future__ import annotations

from server.result import Rejected
from server.rules.rejections.text import (
    effective_suit_name,
    play_shape_text,
)
from server.rules.types import EffectiveSuit, PlayShapeInfo


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
            f"必须优先跟更大的牌型：首出是{play_shape_text(lead_shape)}，"
            "你手里有拖拉机或对子时不能先拆成更小牌型。"
        )


class IllegalFollowShapeRejected(Rejected):
    def __init__(self, lead_shape: PlayShapeInfo) -> None:
        super().__init__(
            f"跟牌牌型不符合首出牌型：首出是{play_shape_text(lead_shape)}，"
            "你手里有对应牌型时必须优先跟。"
        )
