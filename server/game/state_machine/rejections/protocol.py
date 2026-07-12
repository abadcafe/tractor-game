"""Player action protocol rejection types."""

from __future__ import annotations

from server.foundation.result import Rejected


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
        super().__init__(
            "牌格式错误：cards 只能包含 card id 字符串或"
            f"带 id 的对象：{item}"
        )


class InvalidPlayerIndexRejected(Rejected):
    def __init__(self, player_index: int) -> None:
        super().__init__(f"玩家索引无效：{player_index}")
