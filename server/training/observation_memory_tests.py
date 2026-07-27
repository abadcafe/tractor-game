"""Black-box tests for contiguous training observation memory."""

from __future__ import annotations

from server.foundation.result import Ok, Rejected
from server.game import Seat
from server.game.rules.cards import Card
from server.game.snapshots.events import BidEventSnapshot
from server.game.snapshots.tricks import (
    CompletedTrickSnapshot,
    TrickSlotSnapshot,
    TrickSnapshot,
)
from server.training.observation_memory import ObservationMemory
from tests.support import card, seat_values, snapshot


def test_memory_records_pass_and_reveal_at_deal_ordinals() -> None:
    revealed = card("hearts", "2")
    states = (
        snapshot(
            phase="DEAL_BID",
            awaiting_action="bid",
            hand=[revealed],
            remaining_cards=seat_values(1, 0, 0, 0),
        ),
        snapshot(
            phase="DEAL_BID",
            awaiting_action=None,
            hand=[revealed],
            remaining_cards=seat_values(1, 1, 0, 0),
        ),
        snapshot(
            phase="DEAL_BID",
            awaiting_action=None,
            hand=[revealed],
            remaining_cards=seat_values(1, 1, 1, 0),
            bid_events=[
                BidEventSnapshot(
                    actor=Seat.B,
                    cards=(revealed,),
                    kind="trump_rank",
                    suit=revealed.suit,
                    joker_type=None,
                    count=1,
                    deal_ordinal=2,
                )
            ],
        ),
    )
    memory = ObservationMemory()

    assert isinstance(memory.observe(seq=4, snapshot=states[0]), Ok)
    assert isinstance(memory.observe(seq=5, snapshot=states[1]), Ok)
    result = memory.observe(seq=6, snapshot=states[2])

    assert isinstance(result, Ok)
    actions = result.value.bid_actions
    assert tuple(action.disposition for action in actions) == (
        "pass",
        "reveal",
    )
    assert tuple(action.deal_ordinal.value for action in actions) == (
        1,
        2,
    )


def test_memory_rejects_missing_state_sequence() -> None:
    state = snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        hand=[card("hearts", "2")],
        remaining_cards=seat_values(1, 0, 0, 0),
    )
    memory = ObservationMemory()
    assert isinstance(memory.observe(seq=8, snapshot=state), Ok)

    result = memory.observe(seq=10, snapshot=state)

    assert isinstance(result, Rejected)
    assert "missed state sequence 9" in result.reason


def test_memory_accepts_identical_duplicate_delivery() -> None:
    state = snapshot(phase="WAITING", awaiting_action="next_round")
    memory = ObservationMemory()
    first = memory.observe(seq=3, snapshot=state)
    duplicate = memory.observe(seq=3, snapshot=state)

    assert isinstance(first, Ok)
    assert isinstance(duplicate, Ok)
    assert duplicate.value == first.value


def test_memory_records_each_completed_trick_once() -> None:
    played = card("spades", "A")
    trick = CompletedTrickSnapshot(
        lead_actor=Seat.A,
        slots=(
            TrickSlotSnapshot(
                actor=Seat.A,
                cards=(played,),
            ),
        ),
        winner=Seat.A,
        points=0,
        failed_throw=None,
    )
    first = snapshot(
        phase="WAITING",
        awaiting_action="next_round",
        last_completed_trick=trick,
    )
    second = snapshot(
        phase="WAITING",
        awaiting_action=None,
        last_completed_trick=trick,
    )
    memory = ObservationMemory()

    assert isinstance(memory.observe(seq=1, snapshot=first), Ok)
    result = memory.observe(seq=2, snapshot=second)

    assert isinstance(result, Ok)
    assert result.value.completed_tricks == (trick,)


def test_memory_ignores_error_at_current_sequence() -> None:
    state = snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        hand=[card("hearts", "2")],
        remaining_cards=seat_values(1, 0, 0, 0),
    )
    memory = ObservationMemory()
    first = memory.observe(seq=4, snapshot=state)

    error = memory.observe(
        seq=4,
        snapshot=state,
        error="invalid bid",
    )

    assert isinstance(first, Ok)
    assert isinstance(error, Ok)
    assert error.value == first.value


def test_memory_rejects_error_for_unknown_sequence() -> None:
    state = snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        hand=[card("hearts", "2")],
        remaining_cards=seat_values(1, 0, 0, 0),
    )
    memory = ObservationMemory()

    result = memory.observe(
        seq=4,
        snapshot=state,
        error="invalid bid",
    )

    assert isinstance(result, Rejected)
    assert "unknown-state error" in result.reason


def test_memory_rejects_mid_round_initial_snapshot() -> None:
    memory = ObservationMemory()

    result = memory.observe(
        seq=9,
        snapshot=snapshot(
            phase="PLAYING",
            awaiting_action="play",
        ),
    )

    assert isinstance(result, Rejected)
    assert "did not observe round start" in result.reason


def test_memory_keeps_consecutive_equal_face_tricks() -> None:
    first_card = card("hearts", "A", 1)
    second_card = card("hearts", "A", 2)
    first = _completed_trick(first_card)
    second = _completed_trick(second_card)
    deal = snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        hand=[card("spades", "2")],
        remaining_cards=seat_values(1, 0, 0, 0),
    )
    playing = snapshot(
        phase="PLAYING",
        awaiting_action=None,
        remaining_cards=seat_values(25, 25, 25, 25),
    )
    open_second = TrickSnapshot(
        lead_actor=Seat.A,
        current_actor=Seat.B,
        slots=(
            TrickSlotSnapshot(
                actor=Seat.A,
                cards=(second_card,),
            ),
        ),
    )
    memory = ObservationMemory()

    assert isinstance(memory.observe(seq=1, snapshot=deal), Ok)
    assert isinstance(memory.observe(seq=2, snapshot=playing), Ok)
    assert isinstance(
        memory.observe(
            seq=3,
            snapshot=playing.model_copy(
                update={"last_completed_trick": first}
            ),
        ),
        Ok,
    )
    assert isinstance(
        memory.observe(
            seq=4,
            snapshot=playing.model_copy(
                update={
                    "last_completed_trick": first,
                    "trick": open_second,
                }
            ),
        ),
        Ok,
    )
    result = memory.observe(
        seq=5,
        snapshot=playing.model_copy(
            update={"last_completed_trick": second}
        ),
    )

    assert isinstance(result, Ok)
    assert result.value.completed_tricks == (first, second)


def test_memory_clears_history_at_new_round_deal() -> None:
    trick = _completed_trick(card("spades", "A"))
    waiting = snapshot(
        phase="WAITING",
        awaiting_action="next_round",
        last_completed_trick=trick,
    )
    next_deal = snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        hand=[card("hearts", "2")],
        remaining_cards=seat_values(1, 0, 0, 0),
    )
    memory = ObservationMemory()
    assert isinstance(memory.observe(seq=1, snapshot=waiting), Ok)

    result = memory.observe(seq=2, snapshot=next_deal)

    assert isinstance(result, Ok)
    assert result.value.completed_tricks == ()
    assert result.value.bid_actions == ()


def test_memory_fork_continues_without_sharing_mutation() -> None:
    waiting = snapshot(
        phase="WAITING",
        awaiting_action="next_round",
    )
    deal = snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        hand=[card("hearts", "2")],
        remaining_cards=seat_values(1, 0, 0, 0),
    )
    original = ObservationMemory()
    assert isinstance(original.observe(seq=1, snapshot=waiting), Ok)
    forked = original.fork()

    assert isinstance(forked.observe(seq=2, snapshot=deal), Ok)

    assert forked.view() == original.view()
    assert isinstance(original.observe(seq=2, snapshot=deal), Ok)


def _completed_trick(played: Card) -> CompletedTrickSnapshot:
    return CompletedTrickSnapshot(
        lead_actor=Seat.A,
        slots=(
            TrickSlotSnapshot(
                actor=Seat.A,
                cards=(played,),
            ),
        ),
        winner=Seat.A,
        points=played.points,
    )
