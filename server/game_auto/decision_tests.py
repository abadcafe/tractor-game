"""Black-box tests for the rule-driven automatic strategy."""

from __future__ import annotations

import random

from server.game import Seat, commands
from server.game.snapshots.events import BidEventSnapshot
from server.game_auto import choose_auto_command
from tests.support import card, snapshot


def test_choose_auto_command_current_bid_winner_passes() -> None:
    first = card("spades", "2", 1)
    second = card("spades", "2", 2)
    current = BidEventSnapshot(
        actor=Seat.A,
        cards=(first,),
        kind="trump_rank",
        suit=first.suit,
        joker_type=None,
        count=1,
        deal_ordinal=1,
    )
    state = snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        hand=(first, second),
        bid_events=(current,),
        bid_winner=current,
    )

    command = choose_auto_command(
        actor=Seat.A,
        snapshot=state,
        random_source=random.Random(1),
    )

    assert isinstance(command, commands.PassBid)


def test_choose_auto_command_other_seat_can_raise_bid() -> None:
    current_card = card("hearts", "2", 1)
    first = card("spades", "2", 1)
    second = card("spades", "2", 2)
    current = BidEventSnapshot(
        actor=Seat.A,
        cards=(current_card,),
        kind="trump_rank",
        suit=current_card.suit,
        joker_type=None,
        count=1,
        deal_ordinal=1,
    )
    state = snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        hand=(first, second),
        bid_events=(current,),
        bid_winner=current,
    )

    command = choose_auto_command(
        actor=Seat.B,
        snapshot=state,
        random_source=random.Random(1),
    )

    assert isinstance(command, commands.RevealBid)
    assert command.card_ids
    assert set(command.card_ids) <= {first.id, second.id}
