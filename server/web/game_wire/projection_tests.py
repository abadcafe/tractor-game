"""Black-box tests for domain-to-browser state projection."""

from server.game import Seat
from server.game.snapshots.events import BidEventSnapshot
from server.game.snapshots.tricks import (
    TrickSlotSnapshot,
    TrickSnapshot,
)
from server.web.game_wire import StateMessage, encode_state
from tests.support import (
    card,
    partnership_levels,
    seat_values,
    snapshot,
)


def test_encode_state_preserves_complete_browser_contract() -> None:
    hand = (card("spades", "A"),)
    current = snapshot(
        hand=hand,
        remaining_cards=seat_values(1, 2, 3, 4),
        declarer=Seat.C,
        defender_points=15,
        partnership_levels=partnership_levels("5", "10"),
        next_round_confirmed=frozenset((Seat.A, Seat.C)),
    )

    message = encode_state(
        viewer=Seat.A,
        seq=7,
        snapshot=current,
        error="rejected",
    )

    assert message.type == "state"
    assert message.seq == 7
    assert message.error == "rejected"
    assert tuple(item.id for item in message.state.hand) == tuple(
        item.id for item in hand
    )
    assert message.state.remaining_cards.model_dump() == {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
    }
    assert message.state.declarer == "c"
    assert message.state.defender_points == 15
    assert message.state.partnership_levels.first.value == "5"
    assert message.state.partnership_levels.second.value == "10"
    assert message.state.next_round_confirmed == ("a", "c")
    assert not hasattr(current, "action_hints")


def test_encode_state_generates_complete_bid_hints_in_web_layer() -> (
    None
):
    single = card("spades", "2")
    pair = (
        card("hearts", "2", 1),
        card("hearts", "2", 2),
    )
    current = snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        hand=(single, *pair),
        remaining_cards=seat_values(3, 0, 0, 0),
    )

    message = encode_state(
        viewer=Seat.A,
        seq=1,
        snapshot=current,
        error=None,
    )

    assert _hint_ids(message) == (
        (pair[0].id,),
        (single.id,),
        tuple(card.id for card in pair),
    )


def test_encode_state_filters_stir_hints_to_pairs() -> None:
    single = card("spades", "2")
    pair = (
        card("hearts", "2", 1),
        card("hearts", "2", 2),
    )
    current = snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        hand=(single, *pair),
    )

    message = encode_state(
        viewer=Seat.A,
        seq=2,
        snapshot=current,
        error=None,
    )

    assert _hint_ids(message) == (tuple(card.id for card in pair),)


def test_encode_state_generates_follow_hints_from_current_trick() -> (
    None
):
    lead = card("spades", "5")
    follow = card("spades", "7")
    off_suit = card("hearts", "A")
    trick = TrickSnapshot(
        lead_actor=Seat.A,
        current_actor=Seat.B,
        slots=(
            TrickSlotSnapshot(
                actor=Seat.A,
                cards=(lead,),
            ),
        ),
    )
    current = snapshot(
        awaiting_action="play",
        hand=(follow, off_suit),
        remaining_cards=seat_values(1, 2, 1, 1),
        trick=trick,
    )

    message = encode_state(
        viewer=Seat.B,
        seq=3,
        snapshot=current,
        error=None,
    )

    assert _hint_ids(message) == ((follow.id,),)


def test_encode_state_respects_current_bid_winner() -> None:
    pair = (
        card("spades", "2", 1),
        card("spades", "2", 2),
    )
    event = BidEventSnapshot(
        actor=Seat.A,
        cards=pair,
        kind="trump_rank",
        suit=pair[0].suit,
        joker_type=None,
        count=2,
        deal_ordinal=20,
    )
    current = snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        hand=pair,
        remaining_cards=seat_values(5, 5, 5, 5),
        bid_events=(event,),
        bid_winner=event,
    )

    message = encode_state(
        viewer=Seat.A,
        seq=4,
        snapshot=current,
        error=None,
    )

    assert message.state.action_hints == ()
    dumped = message.model_dump(mode="json")
    assert dumped["state"]["bid_events"][0]["deal_ordinal"] == 20


def test_encode_state_hides_overcall_hint_from_current_bid_winner() -> (
    None
):
    single = card("spades", "2")
    pair = (
        card("hearts", "2", 1),
        card("hearts", "2", 2),
    )
    event = BidEventSnapshot(
        actor=Seat.A,
        cards=(single,),
        kind="trump_rank",
        suit=single.suit,
        joker_type=None,
        count=1,
        deal_ordinal=20,
    )
    current = snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        hand=(single, *pair),
        remaining_cards=seat_values(3, 0, 0, 0),
        bid_events=(event,),
        bid_winner=event,
    )

    own_view = encode_state(
        viewer=Seat.A,
        seq=5,
        snapshot=current,
        error=None,
    )
    opponent_view = encode_state(
        viewer=Seat.B,
        seq=5,
        snapshot=current,
        error=None,
    )

    assert own_view.state.action_hints == ()
    assert _hint_ids(opponent_view) == (
        tuple(card.id for card in pair),
    )


def _hint_ids(
    message: StateMessage,
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(card.id for card in hint)
        for hint in message.state.action_hints
    )
