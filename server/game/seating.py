"""Fixed four-seat play order and opposite partnerships."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Literal, assert_never

type SeatId = Literal["a", "b", "c", "d"]
type PartnershipId = Literal["first", "second"]


class Seat(Enum):
    """Stable seat identity without presentation direction."""

    A = "a"
    B = "b"
    C = "c"
    D = "d"


class Partnership(Enum):
    """One of the two fixed opposite-seat partnerships."""

    FIRST = "first"
    SECOND = "second"


_SEATS: tuple[Seat, Seat, Seat, Seat] = (
    Seat.A,
    Seat.B,
    Seat.C,
    Seat.D,
)


def seats() -> tuple[Seat, Seat, Seat, Seat]:
    """Return every seat in play order."""
    return _SEATS


def seat_id(seat: Seat) -> SeatId:
    """Serialize a seat to its closed external identifier."""
    match seat:
        case Seat.A:
            return "a"
        case Seat.B:
            return "b"
        case Seat.C:
            return "c"
        case Seat.D:
            return "d"
    assert_never(seat)


def seat_from_id(value: str) -> Seat | None:
    """Parse an external stable seat identifier."""

    match value:
        case "a":
            return Seat.A
        case "b":
            return Seat.B
        case "c":
            return Seat.C
        case "d":
            return Seat.D
        case _:
            return None


def partnership_id(partnership: Partnership) -> PartnershipId:
    """Serialize a partnership to its closed external identifier."""
    match partnership:
        case Partnership.FIRST:
            return "first"
        case Partnership.SECOND:
            return "second"
    assert_never(partnership)


def next_seat(seat: Seat) -> Seat:
    """Return the actor immediately after ``seat`` in play order."""
    match seat:
        case Seat.A:
            return Seat.B
        case Seat.B:
            return Seat.C
        case Seat.C:
            return Seat.D
        case Seat.D:
            return Seat.A
    assert_never(seat)


def previous_seat(seat: Seat) -> Seat:
    """Return the actor immediately before ``seat`` in play order."""
    match seat:
        case Seat.A:
            return Seat.D
        case Seat.B:
            return Seat.A
        case Seat.C:
            return Seat.B
        case Seat.D:
            return Seat.C
    assert_never(seat)


def partner_seat(seat: Seat) -> Seat:
    """Return the opposite-seat partner."""
    match seat:
        case Seat.A:
            return Seat.C
        case Seat.B:
            return Seat.D
        case Seat.C:
            return Seat.A
        case Seat.D:
            return Seat.B
    assert_never(seat)


def partnership_of(seat: Seat) -> Partnership:
    """Return the fixed partnership containing ``seat``."""
    match seat:
        case Seat.A | Seat.C:
            return Partnership.FIRST
        case Seat.B | Seat.D:
            return Partnership.SECOND
    assert_never(seat)


@dataclass(frozen=True, slots=True, init=False)
class SeatMap[T]:
    """Immutable total mapping containing one value for every seat."""

    _a: T
    _b: T
    _c: T
    _d: T

    def __init__(self, *, a: T, b: T, c: T, d: T) -> None:
        object.__setattr__(self, "_a", a)
        object.__setattr__(self, "_b", b)
        object.__setattr__(self, "_c", c)
        object.__setattr__(self, "_d", d)

    def at(self, seat: Seat) -> T:
        """Return the value belonging to ``seat``."""
        match seat:
            case Seat.A:
                return self._a
            case Seat.B:
                return self._b
            case Seat.C:
                return self._c
            case Seat.D:
                return self._d
        assert_never(seat)

    def replace(self, seat: Seat, value: T) -> SeatMap[T]:
        """Return a map with exactly one seat value replaced."""
        match seat:
            case Seat.A:
                return SeatMap(a=value, b=self._b, c=self._c, d=self._d)
            case Seat.B:
                return SeatMap(a=self._a, b=value, c=self._c, d=self._d)
            case Seat.C:
                return SeatMap(a=self._a, b=self._b, c=value, d=self._d)
            case Seat.D:
                return SeatMap(a=self._a, b=self._b, c=self._c, d=value)
        assert_never(seat)

    def items(
        self,
    ) -> tuple[
        tuple[Seat, T],
        tuple[Seat, T],
        tuple[Seat, T],
        tuple[Seat, T],
    ]:
        """Return every seat and value in play order."""
        return (
            (Seat.A, self._a),
            (Seat.B, self._b),
            (Seat.C, self._c),
            (Seat.D, self._d),
        )

    def values(self) -> tuple[T, T, T, T]:
        """Return every value in play order."""
        return (self._a, self._b, self._c, self._d)

    def map[U](self, transform: Callable[[T], U]) -> SeatMap[U]:
        """Transform every value while preserving its seat."""
        return SeatMap(
            a=transform(self._a),
            b=transform(self._b),
            c=transform(self._c),
            d=transform(self._d),
        )


@dataclass(frozen=True, slots=True, init=False)
class PartnershipMap[T]:
    """Immutable total mapping for both partnerships."""

    _first: T
    _second: T

    def __init__(self, *, first: T, second: T) -> None:
        object.__setattr__(self, "_first", first)
        object.__setattr__(self, "_second", second)

    def at(self, partnership: Partnership) -> T:
        """Return the value belonging to ``partnership``."""
        match partnership:
            case Partnership.FIRST:
                return self._first
            case Partnership.SECOND:
                return self._second
        assert_never(partnership)

    def replace(
        self, partnership: Partnership, value: T
    ) -> PartnershipMap[T]:
        """Return a map with exactly one partnership value replaced."""
        match partnership:
            case Partnership.FIRST:
                return PartnershipMap(first=value, second=self._second)
            case Partnership.SECOND:
                return PartnershipMap(first=self._first, second=value)
        assert_never(partnership)

    def items(
        self,
    ) -> tuple[
        tuple[Partnership, T],
        tuple[Partnership, T],
    ]:
        """Return both partnerships and values."""
        return (
            (Partnership.FIRST, self._first),
            (Partnership.SECOND, self._second),
        )

    def values(self) -> tuple[T, T]:
        """Return both values in partnership order."""
        return (self._first, self._second)

    def map[U](self, transform: Callable[[T], U]) -> PartnershipMap[U]:
        """Transform both values while preserving partnerships."""
        return PartnershipMap(
            first=transform(self._first),
            second=transform(self._second),
        )


__all__ = (
    "Partnership",
    "PartnershipId",
    "PartnershipMap",
    "Seat",
    "SeatId",
    "SeatMap",
    "next_seat",
    "partner_seat",
    "partnership_id",
    "partnership_of",
    "previous_seat",
    "seat_id",
    "seats",
)
