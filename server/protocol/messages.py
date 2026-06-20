"""Wire protocol message envelopes for players and state pushes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .snapshot import StateSnapshot


@dataclass(frozen=True, slots=True)
class PlayerMessage:
    """Raw player-to-server message envelope.

    Game.receive() checks seq before interpreting raw action fields.
    """

    seq: int
    raw: dict[str, object]


class StateMessage(BaseModel):
    """Server-to-player state envelope."""

    model_config = ConfigDict(frozen=True)

    type: Literal["state"] = "state"
    seq: int
    state: StateSnapshot
    error: str | None = None
