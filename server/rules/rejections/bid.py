"""Bid declaration rule rejections."""

from __future__ import annotations

from server.result import Rejected
from server.rules.cards import Rank, Suit
from server.rules.rejections.text import effective_suit_name


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
