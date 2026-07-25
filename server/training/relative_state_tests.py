"""Black-box tests for viewer-relative policy state."""

from __future__ import annotations

from server.foundation.result import Ok
from server.game import Seat, seats
from server.game.snapshots.events import (
    BottomExchangeSnapshot,
    StirDeclarationEventSnapshot,
)
from server.game.snapshots.tricks import (
    FailedThrowSnapshot,
    TrickSlotSnapshot,
    TrickSnapshot,
)
from server.training.observation_memory import ObservationMemoryView
from server.training.relative_state import (
    RelativeActor,
    TrumpMode,
    project_relative_observation,
)
from tests.support import card, partnership_levels, seat_values
from tests.support import snapshot as make_snapshot


def test_project_removes_viewer_and_absolute_team_identity() -> None:
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        declarer=Seat.D,
        partnership_levels=partnership_levels("10", "K"),
        hand=[card("spades", "A")],
        remaining_cards=seat_values(1, 4, 2, 3),
        trick=_open_trick(lead_actor=Seat.B, current_actor=Seat.A),
    )

    result = project_relative_observation(
        viewer=Seat.A,
        snapshot=snapshot,
        memory=ObservationMemoryView(
            bid_actions=(), completed_tricks=()
        ),
    )

    assert isinstance(result, Ok)
    observation = result.value
    assert not hasattr(observation, "player_index")
    assert not hasattr(
        observation.round_context, "declarer_partnership"
    )
    assert observation.round_context.declarer_actor == (
        RelativeActor.PREVIOUS_OPPONENT
    )
    assert observation.round_context.own_level.value == "10"
    assert observation.round_context.opponent_level.value == "K"


def test_project_rotated_seats_produce_equal_relative_state() -> None:
    original = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        declarer=Seat.C,
        partnership_levels=partnership_levels("J", "A"),
        hand=[card("clubs", "5")],
        remaining_cards=seat_values(1, 3, 2, 4),
        trick=_open_trick(lead_actor=Seat.B, current_actor=Seat.A),
    )
    rotated = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        declarer=Seat.D,
        partnership_levels=partnership_levels("A", "J"),
        hand=[card("clubs", "5")],
        remaining_cards=seat_values(4, 1, 3, 2),
        trick=_open_trick(lead_actor=Seat.C, current_actor=Seat.B),
    )
    empty = ObservationMemoryView(bid_actions=(), completed_tricks=())

    first = project_relative_observation(
        viewer=Seat.A, snapshot=original, memory=empty
    )
    second = project_relative_observation(
        viewer=Seat.B, snapshot=rotated, memory=empty
    )

    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert first.value == second.value


def test_project_distinguishes_unset_from_no_trump() -> None:
    empty = ObservationMemoryView(bid_actions=(), completed_tricks=())
    unset = project_relative_observation(
        viewer=Seat.A,
        snapshot=make_snapshot(
            phase="DEAL_BID",
            awaiting_action="bid",
            trump_suit=None,
            hand=[card("clubs", "2")],
            remaining_cards=seat_values(1, 0, 0, 0),
        ),
        memory=empty,
    )
    no_trump = project_relative_observation(
        viewer=Seat.A,
        snapshot=make_snapshot(
            phase="PLAYING",
            awaiting_action="play",
            trump_suit=None,
            hand=[card("clubs", "2")],
            remaining_cards=seat_values(1, 0, 0, 0),
            trick=_open_trick(lead_actor=Seat.A, current_actor=Seat.A),
        ),
        memory=empty,
    )

    assert isinstance(unset, Ok)
    assert isinstance(no_trump, Ok)
    assert unset.value.round_context.trump.mode == TrumpMode.UNSET
    assert no_trump.value.round_context.trump.mode == TrumpMode.NO_TRUMP


def test_project_failed_throw_keeps_only_revealed_extra() -> None:
    forced = card("hearts", "Q")
    extra = card("hearts", "K")
    trick = TrickSnapshot(
        lead_actor=Seat.B,
        current_actor=Seat.C,
        slots=(
            TrickSlotSnapshot(actor=Seat.A, cards=()),
            TrickSlotSnapshot(actor=Seat.B, cards=(forced,)),
            TrickSlotSnapshot(actor=Seat.C, cards=()),
            TrickSlotSnapshot(actor=Seat.D, cards=()),
        ),
        failed_throw=FailedThrowSnapshot(
            actor=Seat.B,
            attempted_cards=(forced, extra),
            forced_cards=(forced,),
        ),
    )

    result = project_relative_observation(
        viewer=Seat.A,
        snapshot=make_snapshot(
            phase="PLAYING",
            awaiting_action=None,
            remaining_cards=seat_values(0, 0, 0, 0),
            trick=trick,
        ),
        memory=ObservationMemoryView(
            bid_actions=(), completed_tricks=()
        ),
    )

    assert isinstance(result, Ok)
    action = result.value.tricks[-1].actions[0]
    assert action.actor == RelativeActor.NEXT_OPPONENT
    assert [
        (item.face.rank.value, item.count) for item in action.played
    ] == [("Q", 1)]
    assert [
        (item.face.rank.value, item.count)
        for item in action.revealed_extra
    ] == [("K", 1)]


def test_round_timeline_has_unique_ordinals_and_query_tail() -> None:
    bottom = BottomExchangeSnapshot(
        actor=Seat.A,
        trigger="initial",
        stir_event_index=None,
        picked_up_bottom_cards=(card("clubs", "2"),),
        discarded_bottom_cards=(card("diamonds", "3"),),
        resulting_bottom_cards=(card("diamonds", "3"),),
    )
    snapshot = make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        own_initial_bottom_exchange=bottom,
        stir_events=[
            StirDeclarationEventSnapshot(
                actor=Seat.B,
                kind="stir",
                cards=(card("hearts", "2"),),
                new_suit=card("hearts", "2").suit,
                priority=1,
                own_bottom_exchange=bottom,
            )
        ],
    )

    result = project_relative_observation(
        viewer=Seat.A,
        snapshot=snapshot,
        memory=ObservationMemoryView(
            bid_actions=(), completed_tricks=()
        ),
    )

    assert isinstance(result, Ok)
    ordinals = tuple(
        action.event_ordinal.value
        for action in result.value.round_actions
    )
    assert ordinals == (101, 102, 103)
    query = result.value.query
    assert query is not None
    assert query.round_event is not None
    assert query.round_event.value == 104
    assert query.round_event.value not in ordinals


def test_round_timeline_uses_only_visible_private_exchanges() -> None:
    public_event = StirDeclarationEventSnapshot(
        actor=Seat.B,
        kind="pass",
        cards=(),
        new_suit=None,
        priority=None,
        own_bottom_exchange=None,
    )
    result = project_relative_observation(
        viewer=Seat.A,
        snapshot=make_snapshot(
            phase="STIRRING",
            awaiting_action="stir",
            own_initial_bottom_exchange=None,
            stir_events=[public_event],
        ),
        memory=ObservationMemoryView(
            bid_actions=(), completed_tricks=()
        ),
    )

    assert isinstance(result, Ok)
    assert tuple(
        action.event_ordinal.value
        for action in result.value.round_actions
    ) == (101,)
    query = result.value.query
    assert query is not None
    assert query.round_event is not None
    assert query.round_event.value == 102


def _open_trick(
    *, lead_actor: Seat, current_actor: Seat
) -> TrickSnapshot:
    lead_card = card("hearts", "2")
    slots = tuple(
        TrickSlotSnapshot(
            actor=seat,
            cards=(lead_card,)
            if seat == lead_actor and lead_actor != current_actor
            else (),
        )
        for seat in seats()
    )
    return TrickSnapshot(
        lead_actor=lead_actor,
        current_actor=current_actor,
        slots=slots,
    )
