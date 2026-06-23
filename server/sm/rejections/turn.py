"""Turn and player-visible action rejection types."""

from __future__ import annotations

from server.actions import GameActionKind
from server.result import Rejected
from server.sm.rejections.text import game_action_text, round_phase_text
from server.sm.types import RoundPhase


class WrongTurnRejected(Rejected):
    def __init__(self, current_player: int | None = None) -> None:
        if current_player is None:
            super().__init__("不是你的回合")
        else:
            super().__init__(
                f"不是你的回合，当前是玩家 {current_player} 的回合"
            )


class WrongBidTurnRejected(Rejected):
    def __init__(self, current_bidder: int) -> None:
        super().__init__(
            f"不是你的抢主回合（当前抢主者：{current_bidder}）"
        )


class DuplicateNextRoundConfirmationRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("你已经确认过了")


class PlayerActionNotAllowedInRoundPhaseRejected(Rejected):
    def __init__(
        self, action: GameActionKind, phase: RoundPhase
    ) -> None:
        super().__init__(
            f"不能在{round_phase_text(phase)}阶段执行"
            f"{game_action_text(action)}。"
        )
