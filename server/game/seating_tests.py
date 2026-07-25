"""Black-box tests for the fixed Tractor seating topology."""

from server.game import (
    Partnership,
    PartnershipMap,
    Seat,
    SeatMap,
    next_seat,
    partner_seat,
    partnership_of,
    previous_seat,
    seat_from_id,
    seat_id,
    seats,
)


def test_seats_returns_each_stable_identity_once() -> None:
    assert seats() == (Seat.A, Seat.B, Seat.C, Seat.D)


def test_seat_from_id_accepts_only_stable_ids() -> None:
    assert (
        tuple(seat_from_id(seat_id(seat)) for seat in seats())
        == seats()
    )
    assert seat_from_id("0") is None
    assert seat_from_id("A") is None


def test_next_seat_forms_complete_play_order_cycle() -> None:
    assert tuple(next_seat(seat) for seat in seats()) == (
        Seat.B,
        Seat.C,
        Seat.D,
        Seat.A,
    )


def test_previous_seat_is_inverse_play_order() -> None:
    assert all(
        previous_seat(next_seat(seat)) == seat for seat in seats()
    )


def test_partner_seat_is_opposite_and_involutive() -> None:
    assert tuple(partner_seat(seat) for seat in seats()) == (
        Seat.C,
        Seat.D,
        Seat.A,
        Seat.B,
    )
    assert all(
        partner_seat(partner_seat(seat)) == seat for seat in seats()
    )


def test_partnership_of_groups_opposite_seats() -> None:
    assert tuple(partnership_of(seat) for seat in seats()) == (
        Partnership.FIRST,
        Partnership.SECOND,
        Partnership.FIRST,
        Partnership.SECOND,
    )


def test_seat_map_reads_replaces_and_iterates_by_seat() -> None:
    values = SeatMap(a=1, b=2, c=3, d=4)

    replaced = values.replace(Seat.C, 30)

    assert tuple(values.items()) == (
        (Seat.A, 1),
        (Seat.B, 2),
        (Seat.C, 3),
        (Seat.D, 4),
    )
    assert tuple(replaced.values()) == (1, 2, 30, 4)
    assert replaced.at(Seat.C) == 30
    assert values.at(Seat.C) == 3


def test_seat_map_maps_every_value_without_exposing_storage() -> None:
    values = SeatMap(a=1, b=2, c=3, d=4)

    mapped = values.map(str)

    assert mapped == SeatMap(a="1", b="2", c="3", d="4")


def test_partnership_map_reads_replaces_and_iterates() -> None:
    values = PartnershipMap(first="2", second="3")

    replaced = values.replace(Partnership.SECOND, "4")

    assert tuple(values.items()) == (
        (Partnership.FIRST, "2"),
        (Partnership.SECOND, "3"),
    )
    assert tuple(replaced.values()) == ("2", "4")
    assert replaced.at(Partnership.SECOND) == "4"
    assert values.at(Partnership.SECOND) == "3"
