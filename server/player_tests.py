"""Tests for server/player.py -- Player, AutoPlayer, HumanPlayer, PlayerAction types."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from server.player import (
    AutoPlayer, HumanPlayer,
    BidAction, StirAction, SkipStirAction,
    DiscardAction, PlayAction, NextRoundAction,
)


def _make_snapshot(
    phase="PLAYING",
    awaiting_action="play",
    current_player=0,
    legal_actions=None,
    trump_rank="2",
    player_hand=None,
):
    """Create a mock StateSnapshot."""
    snap = MagicMock()
    snap.phase = phase
    snap.awaiting_action = awaiting_action
    snap.current_player = current_player
    snap.legal_actions = legal_actions or []
    snap.trump_rank = trump_rank
    snap.player_hand = player_hand if player_hand is not None else []
    return snap


def _make_game(snapshot=None):
    """Create a mock Game that returns the given snapshot."""
    game = MagicMock()
    game.snapshot = MagicMock(return_value=snapshot or _make_snapshot())
    game.act = AsyncMock()
    return game


# ---- PlayerAction types ----


def test_bid_action_fields():
    action = BidAction(cards=["c1", "c2"], count=2)
    assert action.cards == ["c1", "c2"]
    assert action.count == 2


def test_play_action_fields():
    action = PlayAction(cards=["c1"])
    assert action.cards == ["c1"]


def test_stir_action_fields():
    action = StirAction(cards=["c1", "c2"])
    assert action.cards == ["c1", "c2"]


def test_skip_stir_action_fields():
    action = SkipStirAction()
    assert isinstance(action, SkipStirAction)


def test_discard_action_fields():
    action = DiscardAction(cards=["c1", "c2", "c3"])
    assert action.cards == ["c1", "c2", "c3"]


def test_next_round_action_fields():
    action = NextRoundAction()
    assert isinstance(action, NextRoundAction)


# ---- AutoPlayer ----


@pytest.mark.asyncio
async def test_auto_player_play_when_current():
    """AutoPlayer submits a play action when it's their turn in PLAYING phase."""
    legal = [MagicMock(cards=[MagicMock(id="c1")])]
    snap = _make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        current_player=1,
        legal_actions=legal,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=1)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_awaited()


@pytest.mark.asyncio
async def test_auto_player_play_from_legal_actions():
    """AutoPlayer picks from legal_actions when playing."""
    card1 = MagicMock(id="c1")
    legal_play = MagicMock(cards=[card1])
    snap = _make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        current_player=0,
        legal_actions=[legal_play],
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_awaited()
    call_args = game.act.call_args
    assert call_args[0][0] == 0  # player_index


@pytest.mark.asyncio
async def test_auto_player_ignores_wrong_player():
    """AutoPlayer does not act when current_player != self.index in non-dealing phase."""
    snap = _make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        current_player=2,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_player_ignores_wrong_player_stirring():
    """AutoPlayer does not stir when current_player != self.index in STIRRING."""
    snap = _make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        current_player=2,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_player_ignores_wrong_player_discard():
    """AutoPlayer does not discard when current_player != self.index in EXCHANGE."""
    snap = _make_snapshot(
        phase="EXCHANGE",
        awaiting_action="discard",
        current_player=2,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_player_next_round():
    """AutoPlayer submits NextRoundAction when awaiting next_round."""
    snap = _make_snapshot(
        phase="COMPLETE",
        awaiting_action="next_round",
        current_player=0,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_awaited()
    call_args = game.act.call_args
    assert isinstance(call_args[0][1], NextRoundAction)


@pytest.mark.asyncio
async def test_auto_player_ignores_wrong_player_next_round():
    """AutoPlayer does not submit NextRoundAction when current_player != self.index."""
    snap = _make_snapshot(
        phase="COMPLETE",
        awaiting_action="next_round",
        current_player=2,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_player_discard_when_current():
    """AutoPlayer submits DiscardAction when awaiting discard and it's their turn."""
    card1 = MagicMock(id="c1")
    card2 = MagicMock(id="c2")
    snap = _make_snapshot(
        phase="EXCHANGE",
        awaiting_action="discard",
        current_player=0,
        player_hand=[card1, card2],
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_awaited()
    call_args = game.act.call_args
    assert isinstance(call_args[0][1], DiscardAction)


@pytest.mark.asyncio
async def test_auto_player_stir_when_current():
    """AutoPlayer acts during STIRRING when it's their turn."""
    snap = _make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        current_player=0,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_awaited()


@pytest.mark.asyncio
async def test_auto_player_stir_pass():
    """AutoPlayer can pass during STIRRING if no valid stir cards."""
    snap = _make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        current_player=0,
        player_hand=[],
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_awaited()


@pytest.mark.asyncio
async def test_auto_player_bid_during_dealing():
    """AutoPlayer can bid during DEAL_BID phase if hand has trump rank cards."""
    trump_card = MagicMock(id="tc1", rank="2", suit="hearts")
    trump_card.rank = "2"
    snap = _make_snapshot(
        phase="DEAL_BID",
        awaiting_action=None,
        current_player=0,
        player_hand=[trump_card],
        trump_rank="2",
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    # May or may not bid (random), but should not error


@pytest.mark.asyncio
async def test_auto_player_ignores_dealing_if_no_trump_rank():
    """AutoPlayer does not bid during DEAL_BID if hand has no trump rank cards."""
    non_trump = MagicMock(id="nt1")
    non_trump.rank = "3"
    snap = _make_snapshot(
        phase="DEAL_BID",
        awaiting_action=None,
        current_player=0,
        player_hand=[non_trump],
        trump_rank="2",
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_not_awaited()


# ---- HumanPlayer ----


@pytest.mark.asyncio
async def test_human_player_sends_state_on_push():
    """HumanPlayer sends state JSON via WebSocket on on_state."""
    ws = AsyncMock()
    snap = _make_snapshot()
    snap.to_dict.return_value = {"phase": "PLAYING"}
    game = _make_game(snap)
    player = HumanPlayer(index=0, ws=ws)
    await player.on_state(game)
    ws.send_json.assert_awaited_once()
    sent_data = ws.send_json.call_args[0][0]
    assert sent_data["type"] == "state"
    assert sent_data["state"] == {"phase": "PLAYING"}
    snap.to_dict.assert_called_once()


@pytest.mark.asyncio
async def test_human_player_set_ws_replaces_reference():
    """HumanPlayer.set_ws replaces the WebSocket reference."""
    ws1 = AsyncMock()
    ws2 = AsyncMock()
    snap = _make_snapshot()
    game = _make_game(snap)
    player = HumanPlayer(index=0, ws=ws1)
    await player.on_state(game)
    ws1.send_json.assert_awaited_once()

    player.set_ws(ws2)
    await player.on_state(game)
    ws2.send_json.assert_awaited_once()
    assert ws1.send_json.await_count == 1  # not called again


@pytest.mark.asyncio
async def test_human_player_does_not_send_when_no_ws():
    """HumanPlayer does nothing when ws is None."""
    snap = _make_snapshot()
    game = _make_game(snap)
    player = HumanPlayer(index=0, ws=None)
    # Should not raise
    await player.on_state(game)


def test_human_player_is_connected_true():
    """HumanPlayer.is_connected() returns True when ws is set."""
    ws = AsyncMock()
    player = HumanPlayer(index=0, ws=ws)
    assert player.is_connected() is True


def test_human_player_is_connected_false():
    """HumanPlayer.is_connected() returns False when ws is None."""
    player = HumanPlayer(index=0, ws=None)
    assert player.is_connected() is False


def test_human_player_is_connected_false_after_set_ws_none():
    """HumanPlayer.is_connected() returns False after set_ws(None)."""
    ws = AsyncMock()
    player = HumanPlayer(index=0, ws=ws)
    assert player.is_connected() is True
    player.set_ws(None)
    assert player.is_connected() is False
