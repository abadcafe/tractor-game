"""Trick state-machine rejection types."""

from __future__ import annotations

from server.result import Rejected


class TrickResolvedRejected(Rejected):
    def __init__(self) -> None:
        super().__init__("该轮已结束")
