"""Fixed partnership and counterclockwise seat topology."""

from server.foundation.result import Rejected
from server.game.config import Seat

SEATS: tuple[Seat, Seat, Seat, Seat] = (
    Seat.NORTH,
    Seat.WEST,
    Seat.SOUTH,
    Seat.EAST,
)


def team(seat: Seat) -> int:
    return int(seat) % 2


def next_seat(seat: Seat) -> Seat:
    return Seat((int(seat) + 1) % len(SEATS))


def partner(seat: Seat) -> Seat:
    return Seat((int(seat) + 2) % len(SEATS))


def wrong_turn(current: Seat) -> Rejected:
    return Rejected(f"当前应由玩家 {int(current)} 行动")
