"""Card ownership and identity rule rejections."""

from __future__ import annotations

from server.foundation.result import Rejected


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
