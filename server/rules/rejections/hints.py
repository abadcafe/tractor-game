"""Hint-generation rule rejections."""

from __future__ import annotations

from typing import ClassVar

from server.result import Rejected


class TooManyPlayHintsRejected(Rejected):
    reason_text: ClassVar[str] = "too many play hints"

    def __init__(self) -> None:
        super().__init__(self.reason_text)
