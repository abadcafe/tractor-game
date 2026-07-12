"""Stirring and bottom-exchange rejection types."""

from __future__ import annotations

from server.foundation.result import Rejected


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


class StirringMissingExchangeStateRejected(Rejected):
    def __init__(self) -> None:
        super().__init__(
            "炒地皮状态异常：当前需要换底牌状态，"
            "但 exchange_state 为空。"
        )


class InvalidExchangeCountRejected(Rejected):
    def __init__(self, required_count: int, actual_count: int) -> None:
        super().__init__(
            f"埋牌数量错误：需要 {required_count} 张，"
            f"实际 {actual_count} 张"
        )
