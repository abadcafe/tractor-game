"""Top-level game state-machine rejection types."""

from __future__ import annotations

from server.result import Rejected


class CannotStartGameRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("不能开始游戏：游戏已经开始或已经结束。")


class CannotProcessRoundResultRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("不能处理回合结果：游戏尚未开始或已经结束。")
