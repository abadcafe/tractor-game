"""Round orchestration rejection types."""

from __future__ import annotations

from server.result import Rejected
from server.sm.rejections.text import (
    deal_bid_phase_rejection,
    round_phase_rejection,
)
from server.sm.types import DealBidPhase, RoundPhase


class DealCardNotAllowedInRoundPhaseRejected(Rejected):
    def __init__(self, phase: RoundPhase) -> None:
        super().__init__(round_phase_rejection("发牌", phase))


class BidNotAllowedInRoundPhaseRejected(Rejected):
    def __init__(self, phase: RoundPhase) -> None:
        super().__init__(round_phase_rejection("抢主", phase))


class FinalizeDealNotAllowedInRoundPhaseRejected(Rejected):
    def __init__(self, phase: RoundPhase) -> None:
        super().__init__(round_phase_rejection("结束发牌", phase))


class SkipStirNotAllowedInRoundPhaseRejected(Rejected):
    def __init__(self, phase: RoundPhase) -> None:
        super().__init__(round_phase_rejection("不反", phase))


class StirNotAllowedInRoundPhaseRejected(Rejected):
    def __init__(self, phase: RoundPhase) -> None:
        super().__init__(round_phase_rejection("反主", phase))


class DiscardNotAllowedInRoundPhaseRejected(Rejected):
    def __init__(self, phase: RoundPhase) -> None:
        super().__init__(round_phase_rejection("换底牌", phase))


class PlayNotAllowedInRoundPhaseRejected(Rejected):
    def __init__(self, phase: RoundPhase) -> None:
        super().__init__(round_phase_rejection("出牌", phase))


class RoundMissingDealBidStateRejected(Rejected):
    def __init__(self) -> None:
        super().__init__(
            "回合状态异常：当前需要发牌抢主状态，"
            "但 deal_bid_state 为空。"
        )


class RoundMissingStirringStateRejected(Rejected):
    def __init__(self) -> None:
        super().__init__(
            "回合状态异常：当前需要炒地皮状态，但 stirring_state 为空。"
        )


class RoundMissingTrickStateRejected(Rejected):
    def __init__(self) -> None:
        super().__init__(
            "回合状态异常：当前需要出牌状态，但 trick_state 为空。"
        )


class DealCardNotAllowedInDealBidPhaseRejected(Rejected):
    def __init__(self, phase: DealBidPhase) -> None:
        super().__init__(deal_bid_phase_rejection("发牌", phase))


class BidNotAllowedInDealBidPhaseRejected(Rejected):
    def __init__(self, phase: DealBidPhase) -> None:
        super().__init__(deal_bid_phase_rejection("抢主", phase))


class AllCardsDealtRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("所有牌已发完")


class DealNotCompleteRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("还有牌未发完，不能结束发牌")
