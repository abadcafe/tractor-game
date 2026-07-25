"""The single public rejection type of the game aggregate."""

from dataclasses import dataclass

from server.foundation.result import Rejected
from server.game.seating import Seat


@dataclass(frozen=True, slots=True)
class CommandRejected:
    """A command rejected as normal domain input."""

    reason: str


def wrong_turn(current: Seat) -> Rejected:
    """Reject a command from a seat other than the current actor."""

    return Rejected(f"当前应由座位 {current.value} 行动")
