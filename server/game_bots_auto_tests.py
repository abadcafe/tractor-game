"""Black-box tests for the rule-driven automatic policy."""

from __future__ import annotations

import random

from server.foundation.result import Ok
from server.game import Seat, commands
from server.game.rules.cards import CardId
from server.game.snapshots import PlayerSnapshot
from server.game.snapshots.tricks import (
    TrickSlotSnapshot,
    TrickSnapshot,
)
from server.game_bots import (
    AutoPolicy,
    DecisionRequest,
    StrategicAction,
)
from server.game_runtime.player import PlayerView
from tests.support import card, seat_values, snapshot


def _request(
    state: PlayerSnapshot,
    action: StrategicAction,
) -> DecisionRequest:
    return DecisionRequest(
        view=PlayerView(
            viewer=Seat.A,
            seq=1,
            snapshot=state,
            error=None,
        ),
        action=action,
    )


def test_policy_observes_every_view_without_state() -> None:
    policy = AutoPolicy(random.Random(0))
    view = PlayerView(
        viewer=Seat.A,
        seq=1,
        snapshot=snapshot(awaiting_action=None),
        error="previous rejection",
    )

    result = policy.observe(view)

    assert isinstance(result, Ok)


async def test_policy_passes_bid_without_legal_reveal() -> None:
    policy = AutoPolicy(random.Random(1))
    state = snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        hand=[card("spades", "3")],
        remaining_cards=seat_values(1, 0, 0, 0),
    )

    result = await policy.decide(_request(state, "bid"))

    assert isinstance(result, Ok)
    assert isinstance(result.value, commands.PassBid)


async def test_policy_can_choose_complete_bid_reveal() -> None:
    policy = AutoPolicy(random.Random(1))
    level = card("spades", "2")
    state = snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        hand=[level],
        remaining_cards=seat_values(1, 0, 0, 0),
    )

    result = await policy.decide(_request(state, "bid"))

    assert isinstance(result, Ok)
    assert isinstance(result.value, commands.RevealBid)
    assert result.value.card_ids == (CardId(level.id),)


async def test_policy_buries_exact_bottom_count() -> None:
    policy = AutoPolicy(random.Random(3))
    hand = (
        card("hearts", "2"),
        card("hearts", "3"),
        card("hearts", "4"),
        card("hearts", "5"),
        card("hearts", "6"),
        card("hearts", "7"),
        card("hearts", "8"),
        card("hearts", "9"),
        card("hearts", "10"),
    )
    state = snapshot(
        phase="STIRRING",
        awaiting_action="discard",
        hand=hand,
        bottom_cards=hand[:8],
    )

    result = await policy.decide(_request(state, "discard"))

    assert isinstance(result, Ok)
    assert isinstance(result.value, commands.Bury)
    assert len(result.value.card_ids) == 8
    assert len(set(result.value.card_ids)) == 8
    assert set(result.value.card_ids) <= {
        CardId(item.id) for item in hand
    }


async def test_policy_stirs_only_with_a_complete_pair() -> None:
    policy = AutoPolicy(random.Random(1))
    first = card("spades", "2", 1)
    second = card("spades", "2", 2)
    state = snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        hand=(first, second),
    )

    result = await policy.decide(_request(state, "stir"))

    assert isinstance(result, Ok)
    assert isinstance(result.value, commands.Stir)
    assert set(result.value.card_ids) == {
        CardId(first.id),
        CardId(second.id),
    }


async def test_policy_passes_stir_without_pair() -> None:
    policy = AutoPolicy(random.Random(1))
    state = snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        hand=(card("spades", "2", 1),),
    )

    result = await policy.decide(_request(state, "stir"))

    assert isinstance(result, Ok)
    assert isinstance(result.value, commands.PassStir)


async def test_policy_leads_a_card_from_hand() -> None:
    policy = AutoPolicy(random.Random(0))
    hand = (
        card("hearts", "A"),
        card("spades", "3"),
    )
    state = snapshot(
        awaiting_action="play",
        hand=hand,
        remaining_cards=seat_values(2, 2, 2, 2),
    )

    result = await policy.decide(_request(state, "play"))

    assert isinstance(result, Ok)
    assert isinstance(result.value, commands.Play)
    assert len(result.value.card_ids) == 1
    assert result.value.card_ids[0] in {
        CardId(item.id) for item in hand
    }


async def test_policy_follows_pair_structure() -> None:
    policy = AutoPolicy(random.Random(0))
    lead_pair = (
        card("spades", "5", 1),
        card("spades", "5", 2),
    )
    follow_pair = (
        card("spades", "7", 1),
        card("spades", "7", 2),
    )
    off_suit = card("hearts", "A")
    state = snapshot(
        awaiting_action="play",
        hand=(*follow_pair, off_suit),
        remaining_cards=seat_values(3, 3, 3, 3),
        trick=TrickSnapshot(
            lead_actor=Seat.A,
            current_actor=Seat.B,
            slots=(
                TrickSlotSnapshot(
                    actor=Seat.A,
                    cards=lead_pair,
                ),
            ),
        ),
    )

    result = await policy.decide(_request(state, "play"))

    assert isinstance(result, Ok)
    assert isinstance(result.value, commands.Play)
    assert set(result.value.card_ids) == {
        CardId(follow_pair[0].id),
        CardId(follow_pair[1].id),
    }
