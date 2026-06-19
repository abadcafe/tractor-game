"""Tests for HumanPlayer behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import WebSocketDisconnect

from . import human
from .test_helpers import make_game, make_snapshot, make_state_message


@pytest.mark.asyncio
async def test_human_player_handle_connection_accepts_ws() -> None:
    """HumanPlayer.handle_connection accepts WS and waits for player messages."""
    ws = AsyncMock()
    ws.receive_json = AsyncMock(side_effect=WebSocketDisconnect())
    snap = make_snapshot()
    game = make_game(snap)
    player = human.HumanPlayer(index=0)

    await player.handle_connection(ws, game)
    ws.accept.assert_awaited_once()


@pytest.mark.asyncio
async def test_human_player_connection_takeover() -> None:
    """HumanPlayer.handle_connection closes old WS and binds new one."""
    old_ws = AsyncMock()
    new_ws = AsyncMock()
    snap = make_snapshot()
    game = make_game(snap)
    old_ws.receive_json = AsyncMock(side_effect=WebSocketDisconnect())
    player = human.HumanPlayer(index=0)
    await player.handle_connection(old_ws, game)
    assert player.is_connected() is False

    new_ws.receive_json = AsyncMock(side_effect=WebSocketDisconnect())
    await player.handle_connection(new_ws, game)
    new_ws.accept.assert_awaited_once()


@pytest.mark.asyncio
async def test_human_player_does_not_send_when_no_ws() -> None:
    """HumanPlayer.on_state does nothing when no WS is bound."""
    snap = make_snapshot()
    game = make_game(snap)
    player = human.HumanPlayer(index=0)
    await player.on_state(game, make_state_message(snap))


def test_human_player_is_connected_false() -> None:
    """HumanPlayer.is_connected() returns False when no WS is bound."""
    player = human.HumanPlayer(index=0)
    assert player.is_connected() is False
