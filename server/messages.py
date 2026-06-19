"""Wire protocol message envelopes for players and state pushes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

from server.snapshot import SnapshotDict, StateSnapshot


@dataclass(frozen=True, slots=True)
class PlayerMessage:
    """Raw player-to-server message envelope.

    Game.receive() checks seq before interpreting raw action fields.
    """

    seq: int
    raw: dict[str, object]


class StateMessageDict(TypedDict):
    """JSON-serializable server-to-player state message."""

    type: Literal["state"]
    seq: int
    awaiting: str | None
    state: SnapshotDict
    error: NotRequired[str]


@dataclass(frozen=True, slots=True)
class StateMessage:
    """Server-to-player state envelope."""

    seq: int
    awaiting: str | None
    state: StateSnapshot
    error: str | None = None

    def to_dict(self) -> StateMessageDict:
        result: StateMessageDict = {
            "type": "state",
            "seq": self.seq,
            "awaiting": self.awaiting,
            "state": self.state.to_dict(),
        }
        if self.error is not None:
            result["error"] = self.error
        return result
