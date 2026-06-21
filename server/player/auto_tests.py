"""Tests for AutoPlayer behavior."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from server.protocol import (
    StirringStateSnapshot,
    TrickSlotSnapshot,
    TrickSnapshot,
)
from server.rules.cards import Card

from . import auto
from .test_helpers import (
    card,
    is_object_list,
    make_game,
    make_snapshot,
    make_state_message,
)


@pytest.mark.asyncio
async def test_auto_player_play_when_current() -> None:
    """
    AutoPlayer submits a play action when it's their turn in PLAYING
    phase.
    """
    test_card = card("spades", "A", 1)
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[test_card],
    )
    game = make_game(snap)
    player = auto.AutoPlayer(index=1)
    await player.on_state(game, make_state_message(snap))
    await asyncio.sleep(0.05)
    game.receive.assert_awaited()


@pytest.mark.asyncio
async def test_auto_player_play_from_action_hints() -> None:
    """
    AutoPlayer picks from the same action_hints visible to human
    players.
    """
    card1 = card("spades", "A", 1)
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[card1],
        action_hints=[[card1]],
    )
    game = make_game(snap)
    player = auto.AutoPlayer(index=0)
    await player.on_state(game, make_state_message(snap))
    await asyncio.sleep(0.05)
    game.receive.assert_awaited()
    call_args = game.receive.call_args
    assert call_args[0][0] == 0


@pytest.mark.asyncio
async def test_auto_follow_fallback_uses_rules_for_pair_lead() -> None:
    """
    Without action_hints, AutoPlayer fallback still honors pair-follow
    rules.
    """
    lead1 = card("hearts", "3", 1)
    lead2 = card("hearts", "3", 2)
    heart_ace1 = card("hearts", "A", 1)
    heart_ace2 = card("hearts", "A", 2)
    heart_king = card("hearts", "K", 1)
    spade_queen = card("spades", "Q", 1)
    trick = TrickSnapshot(
        lead_player=0,
        current_player=1,
        slots=[
            TrickSlotSnapshot(player=0, cards=[lead1, lead2]),
            TrickSlotSnapshot(player=1, cards=[]),
            TrickSlotSnapshot(player=2, cards=[]),
            TrickSlotSnapshot(player=3, cards=[]),
        ],
    )
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[heart_ace1, heart_ace2, heart_king, spade_queen],
        action_hints=[],
        trick=trick,
    )
    game = make_game(snap)
    player = auto.AutoPlayer(index=1)

    await player.on_state(game, make_state_message(snap))
    await asyncio.sleep(0.05)

    game.receive.assert_awaited()
    message = game.receive.call_args[0][1]
    assert message.raw == {
        "type": "play",
        "cards": [heart_ace1["id"], heart_ace2["id"]],
    }


@pytest.mark.asyncio
async def test_auto_player_error_skips_failed_hint_candidate() -> None:
    """
    A rejected card action is not repeated for the same player-facing
    state.
    """
    card1 = card("spades", "A", 1)
    card2 = card("hearts", "A", 1)
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[card1, card2],
        action_hints=[[card1], [card2]],
    )
    game = make_game(snap)
    player = auto.AutoPlayer(index=0)

    def choose_first(seq: list[list[Card]]) -> list[Card]:
        return seq[0]

    with patch.object(auto.random, "choice", side_effect=choose_first):
        await player.on_state(game, make_state_message(snap))
        await asyncio.sleep(0.05)
        first_message = game.receive.call_args[0][1]
        assert first_message.raw == {
            "type": "play",
            "cards": [card1["id"]],
        }

        game.receive.reset_mock()
        await player.on_state(
            game, make_state_message(snap, error="rejected")
        )
        await asyncio.sleep(0.05)
        second_message = game.receive.call_args[0][1]
        assert second_message.raw == {
            "type": "play",
            "cards": [card2["id"]],
        }


@pytest.mark.asyncio
async def test_auto_player_ignores_when_not_awaiting() -> None:
    """AutoPlayer does not act when awaiting_action is None."""
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action=None,
    )
    game = make_game(snap)
    player = auto.AutoPlayer(index=0)
    await player.on_state(game, make_state_message(snap))
    await asyncio.sleep(0.05)
    game.receive.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_player_ignores_stirring_when_not_awaiting() -> None:
    """
    AutoPlayer does not stir when awaiting_action is None in STIRRING.
    """
    snap = make_snapshot(
        phase="STIRRING",
        awaiting_action=None,
    )
    game = make_game(snap)
    player = auto.AutoPlayer(index=0)
    await player.on_state(game, make_state_message(snap))
    await asyncio.sleep(0.05)
    game.receive.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_player_ignores_discard_when_not_awaiting() -> None:
    """
    AutoPlayer does not discard when awaiting_action is None in
    STIRRING.
    """
    snap = make_snapshot(
        phase="STIRRING",
        awaiting_action=None,
    )
    game = make_game(snap)
    player = auto.AutoPlayer(index=0)
    await player.on_state(game, make_state_message(snap))
    await asyncio.sleep(0.05)
    game.receive.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_player_next_round() -> None:
    """AutoPlayer submits NextRoundAction when awaiting next_round."""
    snap = make_snapshot(
        phase="WAITING",
        awaiting_action="next_round",
    )
    game = make_game(snap)
    player = auto.AutoPlayer(index=0)
    await player.on_state(game, make_state_message(snap))
    await asyncio.sleep(0.05)
    game.receive.assert_awaited()
    call_args = game.receive.call_args
    message = call_args[0][1]
    assert message.raw == {"type": "next_round"}


@pytest.mark.asyncio
async def test_auto_submits_next_round_when_other_player() -> None:
    """
    AutoPlayer submits NextRoundAction whenever awaiting_action is
    next_round.
    """
    snap = make_snapshot(
        phase="WAITING",
        awaiting_action="next_round",
    )
    game = make_game(snap)
    player = auto.AutoPlayer(index=0)
    await player.on_state(game, make_state_message(snap))
    await asyncio.sleep(0.05)
    game.receive.assert_awaited()


@pytest.mark.asyncio
async def test_auto_player_discard_when_current() -> None:
    """
    AutoPlayer submits DiscardAction when awaiting discard and it's
    their turn.
    """
    card1 = card("diamonds", "3", 1)
    card2 = card("clubs", "4", 1)
    snap = make_snapshot(
        phase="STIRRING",
        awaiting_action="discard",
        stirring_state=StirringStateSnapshot(
            phase="EXCHANGING",
            trump_suit=None,
            current_player=0,
            declarer_player=0,
            exchanging_player=0,
            exchange_count=2,
        ),
        player_hand=[card1, card2],
    )
    game = make_game(snap)
    player = auto.AutoPlayer(index=0)
    await player.on_state(game, make_state_message(snap))
    await asyncio.sleep(0.05)
    game.receive.assert_awaited()
    call_args = game.receive.call_args
    message = call_args[0][1]
    assert message.raw["type"] == "discard"


@pytest.mark.asyncio
async def test_auto_player_stir_when_current() -> None:
    """
    AutoPlayer can stir from the same action_hints visible to human
    players.
    """
    card1 = card("hearts", "2", 1)
    card2 = card("hearts", "2", 2)
    snap = make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        action_hints=[[card1, card2]],
    )
    game = make_game(snap)
    player = auto.AutoPlayer(index=0)
    with patch.object(auto.random, "random", return_value=0.4):
        await player.on_state(game, make_state_message(snap))
    await asyncio.sleep(0.05)
    game.receive.assert_awaited()
    message = game.receive.call_args[0][1]
    assert message.raw == {
        "type": "stir",
        "cards": [card1["id"], card2["id"]],
    }


@pytest.mark.asyncio
async def test_auto_player_stir_pass() -> None:
    """AutoPlayer passes during STIRRING when action_hints is empty."""
    snap = make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        player_hand=[],
    )
    game = make_game(snap)
    player = auto.AutoPlayer(index=0)
    await player.on_state(game, make_state_message(snap))
    await asyncio.sleep(0.05)
    game.receive.assert_awaited()
    message = game.receive.call_args[0][1]
    assert message.raw == {"type": "stir", "pass": True}


@pytest.mark.asyncio
async def test_auto_player_stir_randomly_skips_hint() -> None:
    """
    AutoPlayer keeps the old optional-stir behavior by skipping half the
    time.
    """
    card1 = card("hearts", "2", 1)
    card2 = card("hearts", "2", 2)
    snap = make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        action_hints=[[card1, card2]],
    )
    game = make_game(snap)
    player = auto.AutoPlayer(index=0)
    with patch.object(auto.random, "random", return_value=0.6):
        await player.on_state(game, make_state_message(snap))
    await asyncio.sleep(0.05)
    game.receive.assert_awaited()
    message = game.receive.call_args[0][1]
    assert message.raw == {"type": "stir", "pass": True}


@pytest.mark.asyncio
async def test_auto_player_bid_during_dealing() -> None:
    """AutoPlayer bids with the first server-provided hint."""
    trump_card = card("hearts", "2", 1)
    trump_pair_card_1 = card("spades", "2", 1)
    trump_pair_card_2 = card("spades", "2", 2)
    snap = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand=[trump_card, trump_pair_card_1, trump_pair_card_2],
        action_hints=[
            [trump_card],
            [trump_pair_card_1, trump_pair_card_2],
        ],
        trump_rank="2",
    )
    game = make_game(snap)
    player = auto.AutoPlayer(index=0)
    await player.on_state(game, make_state_message(snap))
    await asyncio.sleep(0.05)
    game.receive.assert_awaited()
    call_args = game.receive.call_args
    message = call_args[0][1]
    assert message.raw == {"type": "bid", "cards": [trump_card["id"]]}


@pytest.mark.asyncio
async def test_auto_player_ignores_dealing_if_no_trump_rank() -> None:
    """
    AutoPlayer sends SkipBidAction during DEAL_BID if hand has no trump
    rank cards.
    """
    non_trump = card("spades", "3", 1)
    snap = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand=[non_trump],
        trump_rank="2",
    )
    game = make_game(snap)
    player = auto.AutoPlayer(index=0)
    await player.on_state(game, make_state_message(snap))
    await asyncio.sleep(0.05)
    game.receive.assert_awaited()
    call_args = game.receive.call_args
    message = call_args[0][1]
    assert message.raw == {"type": "bid", "pass": True}


@pytest.mark.asyncio
async def test_auto_player_stir_only_uses_same_suit_pairs() -> None:
    """AutoPlayer stirs only from server-provided action_hints."""
    card_hearts_2_d1 = card("hearts", "2", 1)
    card_spades_2_d1 = card("spades", "2", 1)
    card_hearts_2_d2 = card("hearts", "2", 2)

    snap = make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        player_hand=[
            card_hearts_2_d1,
            card_spades_2_d1,
            card_hearts_2_d2,
        ],
        action_hints=[[card_hearts_2_d1, card_hearts_2_d2]],
        trump_rank="2",
    )
    game = make_game(snap)
    player = auto.AutoPlayer(index=0)

    with patch.object(auto.random, "random", return_value=0.4):
        await player.on_state(game, make_state_message(snap))

    await asyncio.sleep(0.05)
    game.receive.assert_awaited()
    call_args = game.receive.call_args
    message = call_args[0][1]

    assert message.raw == {
        "type": "stir",
        "cards": [card_hearts_2_d1["id"], card_hearts_2_d2["id"]],
    }


@pytest.mark.asyncio
async def test_auto_player_discard_with_stirring_exchange_count() -> (
    None
):
    """
    AutoPlayer._handle_discard uses
    StirringStateSnapshot.exchange_count.
    """
    card1 = card("diamonds", "3", 1)
    card2 = card("clubs", "4", 1)
    card3 = card("spades", "5", 1)

    snap = make_snapshot(
        phase="STIRRING",
        awaiting_action="discard",
        player_hand=[card1, card2, card3],
        stirring_state=StirringStateSnapshot(
            phase="EXCHANGING",
            trump_suit=None,
            current_player=0,
            declarer_player=0,
            exchanging_player=0,
            exchange_count=3,
        ),
    )

    game = make_game(snap)
    player = auto.AutoPlayer(index=0)

    await player.on_state(game, make_state_message(snap))
    await asyncio.sleep(0.05)

    game.receive.assert_awaited()
    call_args = game.receive.call_args
    message = call_args[0][1]
    assert message.raw["type"] == "discard"
    cards = message.raw["cards"]
    assert is_object_list(cards)
    assert len(cards) == 3
