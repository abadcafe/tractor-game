"""Black-box tests for domain-to-browser state projection."""

from server.game import Seat
from server.game.snapshots.events import BidEventSnapshot
from server.game.snapshots.tricks import (
    TrickSlotSnapshot,
    TrickSnapshot,
)
from server.web.game_wire import encode_state
from tests.support import card, snapshot


def test_encode_state_preserves_complete_browser_contract() -> None:
    hand = (card("spades", "A"),)
    current = snapshot(
        player_hand=hand,
        player_hand_counts=(1, 2, 3, 4),
        declarer_player=2,
        defender_points=15,
        team0_level="5",
        team1_level="10",
        next_round_confirmed=(0, 2),
    )

    message = encode_state(
        viewer=Seat.NORTH,
        seq=7,
        snapshot=current,
        error="rejected",
    )

    assert message.type == "state"
    assert message.seq == 7
    assert message.error == "rejected"
    assert message.state.player_hand == hand
    assert message.state.player_hand_counts == (1, 2, 3, 4)
    assert message.state.declarer_player == 2
    assert message.state.declarer_team == 0
    assert message.state.defender_points == 15
    assert message.state.team0_level.value == "5"
    assert message.state.team1_level.value == "10"
    assert message.state.next_round_confirmed == (0, 2)
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
        player_hand=(single, *pair),
        player_hand_counts=(3, 0, 0, 0),
    )

    message = encode_state(
        viewer=Seat.NORTH,
        seq=1,
        snapshot=current,
        error=None,
    )

    assert message.state.action_hints == (
        (pair[0],),
        (single,),
        pair,
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
        player_hand=(single, *pair),
    )

    message = encode_state(
        viewer=Seat.NORTH,
        seq=2,
        snapshot=current,
        error=None,
    )

    assert message.state.action_hints == (pair,)


def test_encode_state_generates_follow_hints_from_current_trick() -> (
    None
):
    lead = card("spades", "5")
    follow = card("spades", "7")
    off_suit = card("hearts", "A")
    trick = TrickSnapshot(
        lead_player=Seat.NORTH,
        current_player=Seat.WEST,
        slots=(
            TrickSlotSnapshot(
                player=Seat.NORTH,
                cards=(lead,),
            ),
        ),
    )
    current = snapshot(
        awaiting_action="play",
        player_hand=(follow, off_suit),
        player_hand_counts=(1, 2, 1, 1),
        trick=trick,
    )

    message = encode_state(
        viewer=Seat.WEST,
        seq=3,
        snapshot=current,
        error=None,
    )

    assert message.state.action_hints == ((follow,),)


def test_encode_state_respects_current_bid_winner() -> None:
    pair = (
        card("spades", "2", 1),
        card("spades", "2", 2),
    )
    event = BidEventSnapshot(
        player=Seat.NORTH,
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
        player_hand=pair,
        player_hand_counts=(5, 5, 5, 5),
        bid_events=(event,),
        bid_winner=event,
    )

    message = encode_state(
        viewer=Seat.NORTH,
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
        player=Seat.NORTH,
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
        player_hand=(single, *pair),
        player_hand_counts=(3, 0, 0, 0),
        bid_events=(event,),
        bid_winner=event,
    )

    own_view = encode_state(
        viewer=Seat.NORTH,
        seq=5,
        snapshot=current,
        error=None,
    )
    opponent_view = encode_state(
        viewer=Seat.WEST,
        seq=5,
        snapshot=current,
        error=None,
    )

    assert own_view.state.action_hints == ()
    assert opponent_view.state.action_hints == (pair,)
