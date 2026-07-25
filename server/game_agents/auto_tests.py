"""Black-box tests for the rule-driven automatic agent."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field

from server.foundation.result import Ok
from server.game import Seat, commands
from server.game.rules.cards import CardId
from server.game.snapshots.tricks import (
    TrickSlotSnapshot,
    TrickSnapshot,
)
from server.game_agents.auto import AutoAgent, automatic_command
from server.game_runtime.session import CommandDecoder, Delivery
from tests.support import card, seat_values, snapshot


def test_automatic_command_ignores_unrequested_action() -> None:
    assert (
        automatic_command(
            snapshot(awaiting_action=None),
            random.Random(0),
        )
        is None
    )


def test_automatic_command_confirms_next_round() -> None:
    command = automatic_command(
        snapshot(phase="WAITING", awaiting_action="next_round"),
        random.Random(0),
    )

    assert isinstance(command, commands.ConfirmRound)


def test_automatic_command_passes_bid_without_legal_reveal() -> None:
    command = automatic_command(
        snapshot(
            phase="DEAL_BID",
            awaiting_action="bid",
            hand=[card("spades", "3")],
            remaining_cards=seat_values(1, 0, 0, 0),
        ),
        random.Random(1),
    )

    assert isinstance(command, commands.PassBid)


def test_automatic_command_can_choose_complete_bid_reveal() -> None:
    level = card("spades", "2")
    command = automatic_command(
        snapshot(
            phase="DEAL_BID",
            awaiting_action="bid",
            hand=[level],
            remaining_cards=seat_values(1, 0, 0, 0),
        ),
        random.Random(1),
    )

    assert isinstance(command, commands.RevealBid)
    assert command.card_ids == (CardId(level.id),)


def test_automatic_command_buries_exact_bottom_count() -> None:
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
    command = automatic_command(
        snapshot(
            phase="STIRRING",
            awaiting_action="discard",
            hand=hand,
            bottom_cards=hand[:8],
        ),
        random.Random(3),
    )

    assert isinstance(command, commands.Bury)
    assert len(command.card_ids) == 8
    assert len(set(command.card_ids)) == 8
    assert set(command.card_ids) <= {CardId(item.id) for item in hand}


def test_automatic_command_stirs_only_with_a_complete_pair() -> None:
    first = card("spades", "2", 1)
    second = card("spades", "2", 2)
    command = automatic_command(
        snapshot(
            phase="STIRRING",
            awaiting_action="stir",
            hand=(first, second),
        ),
        random.Random(1),
    )

    assert isinstance(command, commands.Stir)
    assert set(command.card_ids) == {
        CardId(first.id),
        CardId(second.id),
    }


def test_automatic_command_passes_stir_without_pair() -> None:
    command = automatic_command(
        snapshot(
            phase="STIRRING",
            awaiting_action="stir",
            hand=(card("spades", "2", 1),),
        ),
        random.Random(1),
    )

    assert isinstance(command, commands.PassStir)


def test_automatic_command_leads_a_card_from_hand() -> None:
    hand = (
        card("hearts", "A"),
        card("spades", "3"),
    )
    command = automatic_command(
        snapshot(
            hand=hand,
            remaining_cards=seat_values(2, 2, 2, 2),
        ),
        random.Random(0),
    )

    assert isinstance(command, commands.Play)
    assert len(command.card_ids) == 1
    assert command.card_ids[0] in {CardId(item.id) for item in hand}


def test_automatic_command_follows_pair_structure() -> None:
    lead_pair = (
        card("spades", "5", 1),
        card("spades", "5", 2),
    )
    follow_pair = (
        card("spades", "7", 1),
        card("spades", "7", 2),
    )
    off_suit = card("hearts", "A")
    command = automatic_command(
        snapshot(
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
        ),
        random.Random(0),
    )

    assert isinstance(command, commands.Play)
    assert set(command.card_ids) == {
        CardId(follow_pair[0].id),
        CardId(follow_pair[1].id),
    }


def _submissions() -> list[tuple[int, commands.Command]]:
    return []


@dataclass(slots=True)
class _Target:
    submissions: list[tuple[int, commands.Command]] = field(
        default_factory=_submissions
    )

    async def submit(
        self,
        seq: int,
        decoder: CommandDecoder,
    ) -> None:
        result = decoder.decode()
        assert isinstance(result, Ok)
        self.submissions.append((seq, result.value))


async def test_agent_requests_state_then_processes_delivery() -> None:
    target = _Target()
    agent = AutoAgent(target, random.Random(0))
    starting = asyncio.create_task(agent.start())
    await asyncio.sleep(0)
    await agent.offer(
        Delivery(
            viewer=Seat.A,
            seq=1,
            snapshot=snapshot(awaiting_action=None),
            error=None,
        )
    )
    await starting
    await agent.offer(
        Delivery(
            viewer=Seat.A,
            seq=2,
            snapshot=snapshot(
                phase="WAITING",
                awaiting_action="next_round",
            ),
            error=None,
        )
    )
    await asyncio.sleep(0)
    await agent.stop()

    assert target.submissions[0][0] == 0
    assert isinstance(target.submissions[0][1], commands.ConfirmRound)
    assert target.submissions[1][0] == 2
    assert isinstance(target.submissions[1][1], commands.ConfirmRound)
