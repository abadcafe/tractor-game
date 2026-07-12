"""Bot player construction for room player occupants."""

from __future__ import annotations

import os
from typing import Literal

from server.game.players import AIPlayer, AutoPlayer

type BotKind = Literal["ai", "auto"]
type BotPlayer = AIPlayer | AutoPlayer


def bot_kind_from_env() -> BotKind:
    return (
        bot_kind_from_str(os.environ.get("TRACTOR_BOT_PLAYER", "auto"))
        or "auto"
    )


def bot_kind_from_str(value: str | None) -> BotKind | None:
    match (value or "").strip().lower():
        case "ai":
            return "ai"
        case "auto":
            return "auto"
        case _:
            return None


def create_bot_player(index: int, kind: BotKind) -> BotPlayer:
    match kind:
        case "ai":
            return AIPlayer(index)
        case "auto":
            return AutoPlayer(index)
